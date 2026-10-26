# -*- coding: utf-8 -*-
"""f2b_mixed.py — H1 检验（难分区是否富集混交）+ 用邻域证据推导 45/46 混交候选

* 输入
    F:/lc_work/nb_national_2026/forest6/forest_pool_v1.parquet   472,307 点 × 103 列
    Z:/nb_national_2026/fcs10_dl/tiles/{TILE}_fcs10.tif          FCS10 2023 全国瓦片（10 m, uint8, 0=nodata）
    F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif     30 m 植被类型（EPSG:3857）+ .vat.dbf
    F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet            仅用于训练 AEF 森林/非林判别器 p_forest
* 规则源
    技术文档/84_全国六类林地样本汇总.md §3.2（H1 判据）、§4.2（2026-10-25 修正规则）
      —— 45/46 按**邻域内针叶成分的码**（常绿 71/72 vs 落叶 81/82）拆分，
         不用混交点自身 amp2023/djf2023（原规则已被 f2a 实测推翻）
* 门槛 / 预注册
    H1：难分 = p_needle ∈ [0.35, 0.65]；富集比 ≥3.0 且 95% CI 下界 >1.0（分层版为裁决口径）
        分层 = (1°格 × 生态区)，Mantel-Haenszel 合并 RR（Greenland-Robins 方差）
        + 层内超几何随机化检验（200 次）作 p 值
    B ：5×5 @10m 窗口；needle={71,72,81,82}，broad={51,52,61,62}
        frac_needle ∈ [0.15,0.85] 且 needle/broad 各 ≥3 像元
        45 = 窗口针叶码中常绿{71,72} >60%；46 = 落叶{81,82} >60%
    CV：RF 300 树、n_jobs=8、10 km 空间块 5 折（block 种子 20261025）
        —— 混交点从不进训练；其 p_needle 取**与自身块所属折对应**的那个模型（单模型，
           与纯林点 OOF 的方差结构一致，避免 5 折平均把概率压向 0.5 而虚高难分率）
* 输出（均在 forest6/）
    H1_result.json / f2b_stats.json / mixed_candidates.parquet（全 472,307 行）
* 用法
    python f2b_mixed.py [--skip-pforest] [--n-perm 200]
"""
import argparse
import gc
import json
import math
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

POOL = r'F:/lc_work/nb_national_2026/forest6/forest_pool_v1.parquet'
TILEDIR = r'Z:/nb_national_2026/fcs10_dl/tiles'
VEG30 = r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif'
VEG30_VAT = r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif.vat.dbf'
TILES_CSV = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv'
R7 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
OUT = r'F:/lc_work/nb_national_2026/forest6'

A = ['A%02d' % i for i in range(64)]
NEEDLE = [71, 72, 81, 82]
BROAD = [51, 52, 61, 62]
NEEDLE_EV = [71, 72]
NEEDLE_DEC = [81, 82]
FOREST10 = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92]
PURE = [41, 42, 43, 44]
MIXED = [45, 46]

HARD_LO, HARD_HI = 0.35, 0.65
FRAC_LO, FRAC_HI = 0.15, 0.85
MIN_PX = 3
EV_DOM = 0.60
BLOCK_DEG = 0.1          # 0.1° ≈ 11 km（纬向）；10 km 量级空间块
N_FOLD = 5
SEED_BLOCK = 20261025
SEED_MODEL = 20261025
Z95 = 1.959963984540054

LUT_NEEDLE = np.zeros(256, bool); LUT_NEEDLE[NEEDLE] = True
LUT_BROAD = np.zeros(256, bool); LUT_BROAD[BROAD] = True
LUT_EV = np.zeros(256, bool); LUT_EV[NEEDLE_EV] = True
LUT_DEC = np.zeros(256, bool); LUT_DEC[NEEDLE_DEC] = True
LUT_FOREST = np.zeros(256, bool); LUT_FOREST[FOREST10] = True
LUT_MIXCODE = np.zeros(256, bool); LUT_MIXCODE[[91, 92]] = True


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def jclean(o):
    """递归把 numpy 标量/键转成 JSON 原生类型。"""
    if isinstance(o, dict):
        return {str(jclean(k)): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ---------------------------------------------------------------- 统计工具
def wilson(k, nn, z=Z95):
    """Wilson score 95% CI for a binomial proportion."""
    if nn <= 0:
        return (float('nan'),) * 3
    p_ = k / nn
    d = 1.0 + z * z / nn
    c = (p_ + z * z / (2 * nn)) / d
    h = z * math.sqrt(p_ * (1 - p_) / nn + z * z / (4 * nn * nn)) / d
    return p_, max(0.0, c - h), min(1.0, c + h)


def log_rr_ci(a, n1, b, n0, z=Z95):
    """未分层风险比 + Katz(log) 95% CI。a/n1 = 混交组难分率，b/n0 = 纯林组难分率。"""
    if min(a, b, n1, n0) <= 0:
        return (float('nan'),) * 3
    rr = (a / n1) / (b / n0)
    se = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / b - 1.0 / n0)
    return rr, math.exp(math.log(rr) - z * se), math.exp(math.log(rr) + z * se)


def mh_rr(a, n1, b, n0):
    """Mantel-Haenszel 合并风险比 + Greenland-Robins 方差。仅 n1>0 且 n0>0 的层参与。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1 = np.asarray(n1, float); n0 = np.asarray(n0, float)
    keep = (n1 > 0) & (n0 > 0)
    a, b, n1, n0 = a[keep], b[keep], n1[keep], n0[keep]
    if len(a) == 0:
        return float('nan'), (float('nan'), float('nan')), 0
    N = n1 + n0
    R = float((a * n0 / N).sum())
    S = float((b * n1 / N).sum())
    if R <= 0 or S <= 0:
        return float('nan'), (float('nan'), float('nan')), int(keep.sum())
    rr = R / S
    num = (n1 * n0 * (a + b) - a * b * N) / (N * N)
    var = float(num.sum()) / (R * S)
    if not np.isfinite(var) or var <= 0:
        return rr, (float('nan'), float('nan')), int(keep.sum())
    se = math.sqrt(var)
    return rr, (math.exp(math.log(rr) - Z95 * se), math.exp(math.log(rr) + Z95 * se)), int(keep.sum())


def strata_table(key, hard, is_mixed):
    """按分层键汇总 a(混交难分)/n1(混交) / b(纯林难分)/n0(纯林)。"""
    df = pd.DataFrame({'k': key, 'h': hard.astype(np.int32), 'm': is_mixed.astype(np.int32)})
    g = df.groupby('k').apply(lambda s: pd.Series({
        'n1': int((s.m == 1).sum()), 'n0': int((s.m == 0).sum()),
        'a': int(s.h[s.m == 1].sum()), 'b': int(s.h[s.m == 0].sum())}), include_groups=False)
    return g.reset_index(drop=True)


# ---------------------------------------------------------------- 植被类型 VAT
def read_vat_group_map():
    b = open(VEG30_VAT, 'rb').read()
    nrec, hlen, rlen = struct.unpack('<IHH', b[4:12])
    off, fields = 32, []
    while b[off] != 0x0d:
        fd = b[off:off + 32]; off += 32
        fields.append((fd[:11].split(b'\x00')[0].decode('utf-8', 'ignore'), fd[16]))
    out = {}
    for r in range(nrec):
        rec = b[hlen + r * rlen:hlen + (r + 1) * rlen]
        pos, d = 1, {}
        for name, ln in fields:
            d[name] = rec[pos:pos + ln].decode('utf-8', 'ignore').strip()
            pos += ln
        try:
            v = int(d['Value'])
        except Exception:
            continue
        out[v] = d.get('类型组', '')

    def canon(g):
        if g.startswith('1'):
            return '针叶林'
        if g.startswith('2'):
            return '针阔混交林'
        if g.startswith('3'):
            return '阔叶林'
        return '其他'
    return {k: canon(v) for k, v in out.items()}


def veg30_sample(lon, lat):
    """30 m 植被类型点采样（栅格为 EPSG:3857，先投影）。"""
    R = 6378137.0
    x = np.asarray(lon, float) * math.pi / 180.0 * R
    y = np.log(np.tan((90.0 + np.asarray(lat, float)) * math.pi / 360.0)) * R
    vals = np.empty(len(x), np.int64)
    with rasterio.open(VEG30) as s:
        chunk = 40000
        for i in range(0, len(x), chunk):
            j = min(i + chunk, len(x))
            vals[i:j] = np.fromiter((p[0] for p in s.sample(zip(x[i:j], y[i:j]))),
                                    dtype=np.int64, count=j - i)
    return vals


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-pforest', action='store_true')
    ap.add_argument('--pforest-n', type=int, default=200000, help='每类抽样上限')
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--force-models', action='store_true', help='忽略缓存，重训两个 RF')
    args = ap.parse_args()
    t00 = time.time()

    # ---------- 读池 ----------
    emit('读 forest_pool_v1.parquet')
    cols = ['uid', 'lon', 'lat', 'cls6', 'eco', 'prov', 'tier', 'src_primary',
            'amp2023', 'djf2023'] + A
    p = pd.read_parquet(POOL, columns=cols)
    n = len(p)
    emit('池 %d 点；cls6=%s' % (n, p.cls6.value_counts().sort_index().to_dict()))
    X = np.ascontiguousarray(p[A].values, dtype=np.float32)
    y_nb = p.cls6.isin([41, 42]).values.astype(np.int8)
    is_pure = p.cls6.isin(PURE).values
    is_mixed_lbl = p.cls6.isin(MIXED).values
    emit('纯林 %d，混交标签 %d' % (is_pure.sum(), is_mixed_lbl.sum()))

    # ---------- 10 km 空间块 5 折 ----------
    lon = p.lon.values.astype('float64'); lat = p.lat.values.astype('float64')
    blk = (np.floor(lon / BLOCK_DEG).astype(np.int64) * 1000000
           + np.floor(lat / BLOCK_DEG).astype(np.int64))
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(SEED_BLOCK)
    fold = rng.integers(0, N_FOLD, size=len(ub))[inv]
    emit('空间块 %d 个（%.1f°×%.1f° ≈10 km）；纯林各折点数 %s'
         % (len(ub), BLOCK_DEG, BLOCK_DEG, np.bincount(fold[is_pure], minlength=N_FOLD).tolist()))

    # ---------- 模型阶段（带断点缓存，避免重跑重训） ----------
    CACHE = os.path.join(OUT, 'code', '_f2b_model_cache.npz')
    p_needle = np.full(n, np.nan, np.float32)
    p_forest = np.full(n, np.nan, np.float32)
    pforest_info = {'skipped': True}
    mixed_idx = np.where(is_mixed_lbl)[0]
    if os.path.exists(CACHE) and not args.force_models:
        z = np.load(CACHE)
        if len(z['p_needle']) == n:
            p_needle = z['p_needle']; p_forest = z['p_forest']
            fold = z['fold']; fold_auc = z['fold_auc'].tolist()
            oof_auc = float(z['oof_auc'])
            pforest_info = json.loads(str(z['pforest_info']))
            emit('命中模型缓存 %s：OOF AUC=%.4f' % (CACHE, oof_auc))
        else:
            emit('缓存长度不符，重训')
            args.force_models = True
    if not (os.path.exists(CACHE) and not args.force_models):
        # ---------- 针阔二分类（仅纯林训练，OOF；混交点用同折模型） ----------
        p_needle = np.full(n, np.nan, np.float32)
        fold_auc = []
        for f in range(N_FOLD):
            tr = np.where(is_pure & (fold != f))[0]
            te = np.where(is_pure & (fold == f))[0]
            t0 = time.time()
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                       n_jobs=args.n_jobs, random_state=SEED_MODEL + f)
            m.fit(X[tr], y_nb[tr])
            p_needle[te] = m.predict_proba(X[te])[:, 1]
            auc = float(roc_auc_score(y_nb[te], p_needle[te]))
            fold_auc.append(auc)
            # 混交点：按自身块所属折取对应模型（该模型从未见过任何混交点）
            mi = mixed_idx[fold[mixed_idx] == f]
            if len(mi):
                p_needle[mi] = m.predict_proba(X[mi])[:, 1]
            emit('  折%d：train %d / test %d / mixed %d，AUC=%.4f，%.0fs'
                 % (f, len(tr), len(te), len(mi), auc, time.time() - t0))
            del m; gc.collect()
        oof_auc = float(roc_auc_score(y_nb[is_pure], p_needle[is_pure]))
        emit('针阔 OOF AUC（纯林 %d 点）= %.4f' % (int(is_pure.sum()), oof_auc))

        # ---------- p_forest（AEF 森林/非林，doc84 §4.2(d)） ----------
        p_forest = np.full(n, np.nan, np.float32)
        if not args.skip_pforest:
            emit('训练 p_forest（r7_train_2023 抽样）')
            pf = pd.read_parquet(R7, columns=['class_new'] + A)
            isf = pf.class_new.isin(FOREST10).values
            pos = np.where(isf)[0].copy(); neg = np.where(~isf)[0].copy()
            rp = np.random.default_rng(SEED_MODEL); rp.shuffle(pos); rp.shuffle(neg)
            pos = pos[:args.pforest_n]; neg = neg[:args.pforest_n]
            idx = np.concatenate([pos, neg])
            yy = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
            ho = np.random.default_rng(SEED_MODEL + 2).random(len(idx)) < 0.1
            Xp = pf[A].values.astype(np.float32)
            t0 = time.time()
            mf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features='sqrt',
                                        class_weight='balanced', n_jobs=args.n_jobs,
                                        random_state=SEED_MODEL)
            mf.fit(Xp[idx[~ho]], yy[~ho])
            ho_auc = float(roc_auc_score(yy[ho], mf.predict_proba(Xp[idx[ho]])[:, 1]))
            p_forest[:] = mf.predict_proba(X)[:, 1].astype(np.float32)
            pforest_info = {'skipped': False, 'n_pos': int(len(pos)), 'n_neg': int(len(neg)),
                            'holdout_auc_random10pct': round(ho_auc, 4), 'fit_sec': round(time.time() - t0, 1),
                            'note': '正=森林码51..92，负=其余24类；随机10%留出（非空间）仅作健全性检查'}
            emit('  p_forest holdout AUC=%.4f，%.0fs；池内 p_forest 中位 %.3f'
                 % (ho_auc, time.time() - t0, float(np.nanmedian(p_forest))))
            del pf, Xp, mf; gc.collect()
        np.savez_compressed(CACHE, p_needle=p_needle, p_forest=p_forest, fold=fold,
                            fold_auc=np.array(fold_auc), oof_auc=np.float64(oof_auc),
                            pforest_info=np.array(json.dumps(pforest_info)))
        emit('模型缓存已写 %s' % CACHE)

    # ---------- Task A：H1 ----------
    emit('Task A：H1 检验')
    hard = (p_needle >= HARD_LO) & (p_needle <= HARD_HI)
    a_m = int((hard & is_mixed_lbl).sum()); n_m = int(is_mixed_lbl.sum())
    b_p = int((hard & is_pure).sum()); n_p = int(is_pure.sum())
    pm, pm_lo, pm_hi = wilson(a_m, n_m)
    pp, pp_lo, pp_hi = wilson(b_p, n_p)
    rr_raw, rr_raw_lo, rr_raw_hi = log_rr_ci(a_m, n_m, b_p, n_p)
    emit('  未分层：混交难分 %d/%d=%.3f，纯林 %d/%d=%.5f，RR=%.2f (%.2f-%.2f)'
         % (a_m, n_m, pm, b_p, n_p, pp, rr_raw, rr_raw_lo, rr_raw_hi))

    c1 = (np.floor(lon).astype(np.int64) * 1000 + (np.floor(lat).astype(np.int64) + 100)).astype(np.int64)
    eco_code = pd.factorize(p.eco.values)[0].astype(np.int64)
    key_ce = c1 * 1000 + eco_code
    key_cell = c1
    key_eco = eco_code
    key_c2 = (np.floor(lon / 2).astype(np.int64) * 1000
              + (np.floor(lat / 2).astype(np.int64) + 100)).astype(np.int64) * 1000 + eco_code

    def do_strat(key, min_n1=1, min_n0=1):
        t = strata_table(key, hard, is_mixed_lbl)
        t2 = t[(t.n1 >= min_n1) & (t.n0 >= min_n0)]
        rr, ci, k_use = mh_rr(t2.a.values, t2.n1.values, t2.b.values, t2.n0.values)
        return t, t2, rr, ci, k_use

    tab_ce, tab_ce_use, rr_mh, ci_mh, k_mh = do_strat(key_ce)
    _, _, rr_mh_cell, ci_mh_cell, k_cell = do_strat(key_cell)
    _, _, rr_mh_eco, ci_mh_eco, k_eco = do_strat(key_eco)
    _, _, rr_mh_c2, ci_mh_c2, k_c2 = do_strat(key_c2)
    _, tab_ce5, rr_mh5, ci_mh5, k_mh5 = do_strat(key_ce, min_n1=5)

    # 分层随机化检验：层内固定难分总数，超几何重抽混交组难分数
    emit('  分层随机化检验（%d 次）' % args.n_perm)
    K = tab_ce_use.a.values + tab_ce_use.b.values
    Nl = tab_ce_use.n1.values + tab_ce_use.n0.values
    n1l = tab_ce_use.n1.values
    n0l = tab_ce_use.n0.values
    rr_null = np.full(args.n_perm, np.nan)
    prng = np.random.default_rng(SEED_BLOCK + 7)
    for r in range(args.n_perm):
        a_r = prng.hypergeometric(K.astype(np.int64), (Nl - K).astype(np.int64), n1l.astype(np.int64))
        rr_r, _, _ = mh_rr(a_r, n1l, K - a_r, n0l)
        rr_null[r] = rr_r
    perm_p = float((np.sum(rr_null >= rr_mh) + 1) / (args.n_perm + 1))

    def sub_rr(mask):
        am = int((hard & is_mixed_lbl & mask).sum()); nm = int((is_mixed_lbl & mask).sum())
        bp = int((hard & is_pure & mask).sum()); np_ = int((is_pure & mask).sum())
        rr, lo, hi = log_rr_ci(am, nm, bp, np_)
        return {'n_mixed': nm, 'n_pure': np_,
                'rate_mixed': round(am / nm, 4) if nm else None,
                'rate_pure': round(bp / np_, 5) if np_ else None,
                'RR': None if math.isnan(rr) else round(rr, 3),
                'RR_ci95': None if math.isnan(lo) else [round(lo, 3), round(hi, 3)]}

    def rate_by(mask, col):
        out = {}
        sub = p.loc[mask, [col]]
        hs = hard[mask]; ms = is_mixed_lbl[mask]
        for v in sub[col].unique():
            sel = (sub[col] == v).values
            out[str(v)] = {'n': int(sel.sum()), 'n_mixed': int((sel & ms).sum()),
                           'hard_rate_mixed': round(float(hs[sel & ms].mean()), 4) if (sel & ms).any() else None,
                           'hard_rate_pure': round(float(hs[sel & ~ms].mean()), 5) if (sel & ~ms).any() else None}
        return out

    ne_eco = (p.eco == '东北').values
    hard_cls = p.loc[hard, 'cls6'].value_counts().sort_index()
    hard_by_cls = {int(k): {'n': int((p.cls6 == k).sum()),
                            'hard_rate': round(float(hard[(p.cls6 == k).values].mean()), 4)}
                   for k in sorted(p.cls6.unique())}
    # 分轴对照：混交 vs 纯针叶 / 纯阔叶
    pn_pure = (p.cls6.isin([41, 42])).values
    pb_pure = (p.cls6.isin([43, 44])).values
    h1 = {
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'prereg': {'hard_zone': [HARD_LO, HARD_HI], 'enrich_thr': 3.0,
                   'criterion': 'RR_MH >= 3.0 且 95%CI 下界 > 1.0（分层版为裁决口径）',
                   'model': 'AEF64 针阔二分类 RF300，仅纯林 41/42/43/44 训练，10km 块 5 折 OOF',
                   'mixed_scoring': '混交点从不进训练；取自身块所属折的单一模型（与纯林 OOF 方差结构一致）'},
        'model_qc': {'oof_auc_pure': round(oof_auc, 4), 'fold_auc': [round(x, 4) for x in fold_auc],
                     'n_blocks': int(len(ub)), 'block_deg': BLOCK_DEG,
                     'n_pure_train_total': int(is_pure.sum())},
        'hard_zone': {'n_total': n, 'n_hard': int(hard.sum()),
                      'share_of_all': round(float(hard.mean()), 5),
                      'cls6_composition': {int(k): int(v) for k, v in hard_cls.items()},
                      'mixed_label_share_within_hard': round(float(is_mixed_lbl[hard].mean()), 4)
                      if hard.any() else None,
                      'hard_rate_by_cls6': hard_by_cls},
        'unstratified': {'n_mixed': n_m, 'n_pure': n_p,
                         'P_hard_given_mixed': round(pm, 4), 'wilson_ci_mixed': [round(pm_lo, 4), round(pm_hi, 4)],
                         'P_hard_given_pure': round(pp, 6), 'wilson_ci_pure': [round(pp_lo, 6), round(pp_hi, 6)],
                         'RR': round(rr_raw, 3), 'RR_ci95': [round(rr_raw_lo, 3), round(rr_raw_hi, 3)]},
        'secondary_axis': {
            'P_hard|pure_needle': round(float(hard[pn_pure].mean()), 5),
            'P_hard|pure_broad': round(float(hard[pb_pure].mean()), 5),
            'RR_mixed_vs_pure_needle': round(pm / float(hard[pn_pure].mean()), 2) if hard[pn_pure].any() else None,
            'RR_mixed_vs_pure_broad': round(pm / float(hard[pb_pure].mean()), 2) if hard[pb_pure].any() else None,
        },
        'stratified': {
            'strata_def_primary': '1°格 × 生态区',
            'RR_MH': None if math.isnan(rr_mh) else round(rr_mh, 3),
            'RR_MH_ci95': None if math.isnan(ci_mh[0]) else [round(ci_mh[0], 3), round(ci_mh[1], 3)],
            'n_strata_used': k_mh, 'n_strata_total': int(len(tab_ce)),
            'n_mixed_in_used_strata': int(tab_ce_use.n1.sum()),
            'perm_test_p': perm_p, 'n_perm': args.n_perm,
            'rr_null_median': round(float(np.nanmedian(rr_null)), 3) if np.isfinite(rr_null).any() else None,
            'sensitivity': {
                '1°格 only': {'RR_MH': None if math.isnan(rr_mh_cell) else round(rr_mh_cell, 3),
                              'ci': None if math.isnan(ci_mh_cell[0]) else [round(ci_mh_cell[0], 3), round(ci_mh_cell[1], 3)],
                              'n_strata': k_cell},
                '生态区 only': {'RR_MH': None if math.isnan(rr_mh_eco) else round(rr_mh_eco, 3),
                                'ci': None if math.isnan(ci_mh_eco[0]) else [round(ci_mh_eco[0], 3), round(ci_mh_eco[1], 3)],
                                'n_strata': k_eco},
                '2°格×生态区': {'RR_MH': None if math.isnan(rr_mh_c2) else round(rr_mh_c2, 3),
                                'ci': None if math.isnan(ci_mh_c2[0]) else [round(ci_mh_c2[0], 3), round(ci_mh_c2[1], 3)],
                                'n_strata': k_c2},
                '1°格×生态区(混交≥5)': {'RR_MH': None if math.isnan(rr_mh5) else round(rr_mh5, 3),
                                        'ci': None if math.isnan(ci_mh5[0]) else [round(ci_mh5[0], 3), round(ci_mh5[1], 3)],
                                        'n_strata': k_mh5, 'n_mixed_in_used_strata': int(tab_ce5.n1.sum())},
            }},
        'confound': {
            'mixed_eco_share': {k: round(v / n_m, 3) for k, v in p.loc[is_mixed_lbl, 'eco'].value_counts().items()},
            'pure_eco_share': {k: round(v / n_p, 3) for k, v in p.loc[is_pure, 'eco'].value_counts().items()},
            'src_by_group': {g: p.loc[mm, 'src_primary'].value_counts().head(5).to_dict()
                             for g, mm in [('mixed', is_mixed_lbl), ('pure', is_pure)]},
            'within_东北': sub_rr(ne_eco),
            'outside_东北': sub_rr(~ne_eco),
            'by_eco': rate_by(np.ones(n, bool), 'eco'),
            'top10_cells_mixed': (np.floor(lon[is_mixed_lbl]).astype(int).astype(str) + 'E'
                                  + np.floor(lat[is_mixed_lbl]).astype(int).astype(str) + 'N'
                                  ).astype(object).tolist(),
        },
    }
    tc = pd.Series(h1['confound']['top10_cells_mixed']).value_counts().head(10).to_dict()
    h1['confound']['top10_cells_mixed'] = tc
    h1['verdict'] = {
        'H1_supported': bool((not math.isnan(rr_mh)) and rr_mh >= 3.0 and (not math.isnan(ci_mh[0])) and ci_mh[0] > 1.0),
        'reason': ('RR_MH=%.2f (95%%CI %.2f–%.2f)，判据 ≥3.0 且下界>1.0' % (rr_mh, ci_mh[0], ci_mh[1]))
                  if not math.isnan(rr_mh) else '分层 RR 不可计算',
        'unstratified_RR_reference': None if math.isnan(rr_raw) else round(rr_raw, 2),
    }
    emit('  H1 分层 RR_MH=%.2f（CI %.2f-%.2f），perm p=%.4f → supported=%s'
         % (rr_mh, ci_mh[0], ci_mh[1], perm_p, h1['verdict']['H1_supported']))
    with open(os.path.join(OUT, 'H1_result.json'), 'w', encoding='utf-8') as f:
        json.dump(jclean(h1), f, ensure_ascii=False, indent=1)

    # ---------- Task B：FCS10 5×5 窗口 ----------
    emit('Task B：FCS10 5×5 窗口（逐瓦读，读后即释放）')
    ix = np.floor((lon - 73) / 2).astype(np.int64)
    iy = np.floor((lat - 18) / 2).astype(np.int64)
    tile_id = np.array(['T%02d%02d' % (a, b) for a, b in zip(ix, iy)])
    avail = set(f[:-len('_fcs10.tif')] for f in os.listdir(TILEDIR) if f.endswith('_fcs10.tif'))
    bad = sorted(set(tile_id.tolist()) - avail)
    if bad:
        emit('  !! 缺失瓦片 %d 个：%s' % (len(bad), bad[:10]))
    else:
        emit('  瓦片齐备：需要 %d 个，全部可用' % len(set(tile_id.tolist())))

    n_needle = np.zeros(n, np.int16); n_broad = np.zeros(n, np.int16)
    n_ev = np.zeros(n, np.int16); n_dec = np.zeros(n, np.int16)
    n_mixcode = np.zeros(n, np.int16); n_valid = np.zeros(n, np.int16)
    center_code = np.zeros(n, np.uint8); n_edge = 0
    order = np.argsort(tile_id, kind='stable')
    uniq_tiles = sorted(set(tile_id.tolist()))
    bounds = np.searchsorted(tile_id[order], np.array(uniq_tiles), side='left')
    ends = np.append(bounds[1:], len(order))
    done = 0
    for tname, s, e in zip(uniq_tiles, bounds, ends):
        if tname not in avail:
            continue
        idx = order[s:e]
        with rasterio.open(os.path.join(TILEDIR, tname + '_fcs10.tif')) as ds:
            H, W = ds.height, ds.width
            inv_t = ~ds.transform
            cc, rr = inv_t * (lon[idx], lat[idx])
            r = np.floor(np.asarray(rr)).astype(np.int64)
            c = np.floor(np.asarray(cc)).astype(np.int64)
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            if not ok.all():
                emit('    %s: %d 点越界被剔' % (tname, int((~ok).sum())))
                idx = idx[ok]; r = r[ok]; c = c[ok]
            # 仅在点的外包窗口内读，避免整瓦 496 MB 常驻
            r0, r1 = max(0, int(r.min()) - 2), min(H, int(r.max()) + 3)
            c0, c1 = max(0, int(c.min()) - 2), min(W, int(c.max()) + 3)
            a = ds.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
            hh, ww = a.shape
            r = r - r0; c = c - c0
            center_code[idx] = a[r, c]
            rs = r[:, None] + np.arange(-2, 3)[None, :]
            cs = c[:, None] + np.arange(-2, 3)[None, :]
            n_edge += int(((rs < 0) | (rs >= hh) | (cs < 0) | (cs >= ww)).any(axis=1).sum())
            rs = np.clip(rs, 0, hh - 1); cs = np.clip(cs, 0, ww - 1)
            patch = a[rs[:, :, None], cs[:, None, :]]
            n_needle[idx] = LUT_NEEDLE[patch].sum((1, 2))
            n_broad[idx] = LUT_BROAD[patch].sum((1, 2))
            n_ev[idx] = LUT_EV[patch].sum((1, 2))
            n_dec[idx] = LUT_DEC[patch].sum((1, 2))
            n_mixcode[idx] = LUT_MIXCODE[patch].sum((1, 2))
            n_valid[idx] = (patch != 0).sum((1, 2))
            del a, patch
        done += 1
        if done % 40 == 0 or done == len(uniq_tiles):
            emit('  瓦片 %d/%d（%s）累计 %.0fs' % (done, len(uniq_tiles), tname, time.time() - t00))
        del idx
        gc.collect()

    den = n_needle.astype(np.int32) + n_broad
    frac_needle = np.where(den > 0, n_needle / np.maximum(den, 1), np.nan)
    ev_frac = np.where(n_needle > 0, n_ev / np.maximum(n_needle, 1), np.nan)
    center_forest = LUT_FOREST[center_code]
    has_cov = n_valid > 0
    both_present = (n_needle >= 1) & (n_broad >= 1)
    both_ge3 = (n_needle >= MIN_PX) & (n_broad >= MIN_PX)
    frac_ok = both_ge3 & (frac_needle >= FRAC_LO) & (frac_needle <= FRAC_HI)
    d45 = frac_ok & (ev_frac > EV_DOM)
    d46 = frac_ok & (ev_frac < 1.0 - EV_DOM)
    derived = np.where(d45, 45, np.where(d46, 46, 0)).astype(np.int8)
    cand_status = np.where(d45, '45', np.where(d46, '46', np.where(
        frac_ok, 'uncertain_mixed', np.where(both_present, 'both_present_frac_out', 'not_mixed')))).astype(object)

    emit('  采样 30m 植被类型')
    v30 = veg30_sample(lon, lat)
    vmap = read_vat_group_map()
    v30_group = np.array(['NA' if int(v) == 65535 else vmap.get(int(v), '其他') for v in v30], dtype=object)

    # ---------- Task C：分布 + 与既有混交点重合 ----------
    emit('Task C：分布与重合')
    der = derived > 0
    R = 6378137.0

    def unit3(lonv, latv):
        lo = np.radians(lonv); la = np.radians(latv)
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    dist_nearest = np.full(n, np.nan)
    if is_mixed_lbl.any():
        tree = cKDTree(unit3(lon[is_mixed_lbl], lat[is_mixed_lbl]))
        ch, _ = tree.query(unit3(lon, lat), k=1)
        dist_nearest = 2 * R * np.arcsin(np.clip(ch / 2, 0, 1))
    near = {('<=%dm' % t): int((dist_nearest[der] <= t).sum()) for t in [50, 100, 250, 500, 1000, 5000]}

    def cellstr(mask):
        return (np.floor(lon[mask]).astype(int).astype(str) + 'E' + np.floor(lat[mask]).astype(int).astype(str) + 'N')

    cell_der = pd.Series(cellstr(der)); cell_mix = pd.Series(cellstr(is_mixed_lbl))
    overlap_cells = set(cell_der) & set(cell_mix)

    funnel = {
        'step0_total_pool': n,
        'step1_fcs10_covered': int(has_cov.sum()),
        'step2_window_both_present(needle>=1 & broad>=1)': int(both_present.sum()),
        'step3_window_both_ge3': int(both_ge3.sum()),
        'step4_frac_needle_in_0.15_0.85': int(frac_ok.sum()),
        'step5_derived_45': int(d45.sum()),
        'step5_derived_46': int(d46.sum()),
        'step5_derived_total': int(der.sum()),
        'note_center_forest_is_not_a_gate': {
            'center_is_forest_code': int(center_forest.sum()),
            'derived_and_center_forest': int((der & center_forest).sum()),
            'derived_but_center_not_forest': int((der & ~center_forest).sum())},
        'uncertain(EV/DEC 均不>60%)': int((frac_ok & ~der).sum()),
        'cand_status_counts': pd.Series(cand_status).value_counts().to_dict(),
        'edge_clamped_points': int(n_edge),
        'missing_tiles': len(bad),
    }
    strict_gate = der & center_forest & (np.nan_to_num(p_forest, nan=1.0) >= 0.5) & np.isin(v30_group, ['针叶林', '阔叶林', '针阔混交林'])
    funnel['strict_variant(doc84 a+d+e 全门)'] = {
        'n': int(strict_gate.sum()), '45': int((strict_gate & (derived == 45)).sum()),
        '46': int((strict_gate & (derived == 46)).sum()),
        'def': 'derived & center_is_forest & p_forest>=0.5 & v30_group∈{针叶林,阔叶林,针阔混交林}'}

    def dist_table(mask):
        return {'eco': pd.Series(p.eco.values[mask]).value_counts().to_dict(),
                'prov': pd.Series(p.prov.values[mask]).value_counts().to_dict(),
                'cells_1deg': int(pd.Series(cellstr(mask)).nunique()),
                'top15_cells': pd.Series(cellstr(mask)).value_counts().head(15).to_dict()}

    ctab = pd.crosstab(pd.Series(v30_group[der], name='v30'), pd.Series(derived[der], name='der'))
    mix_win = {
        'n_mixed_label': int(is_mixed_lbl.sum()),
        'their_frac_needle_median': round(float(np.nanmedian(frac_needle[is_mixed_lbl])), 4)
        if np.isfinite(frac_needle[is_mixed_lbl]).any() else None,
        'their_center_code_top': pd.Series(center_code[is_mixed_lbl]).value_counts().head(6).to_dict(),
        'their_both_present': int(both_present[is_mixed_lbl].sum()),
        'their_frac_ok': int(frac_ok[is_mixed_lbl].sum()),
        'their_derived_45': int(d45[is_mixed_lbl].sum()),
        'their_derived_46': int(d46[is_mixed_lbl].sum()),
        'their_window_mixcode_px_median': float(np.median(n_mixcode[is_mixed_lbl])),
        'interpretation': 'FCS10 91/92 像元本身不计入 needle/broad 分母；混交标签点窗口内 needle+broad 常不足，说明规则的“混交”=纯针叶斑块与纯阔叶斑块相邻，与 FCS10 的 91/92 混交斑块是两种不同表达',
    }
    disc_by_cls = {int(k): {'n': int((p.cls6 == k).sum()),
                            'derived_45': int((d45 & (p.cls6 == k).values).sum()),
                            'derived_46': int((d46 & (p.cls6 == k).values).sum()),
                            'derived_any_rate': round(float(der[(p.cls6 == k).values].mean()), 4)}
                   for k in PURE + MIXED}
    mixcode_maj = has_cov & (n_mixcode >= MIN_PX)
    pheno_incon = der & np.isfinite(p.amp2023.values) & (p.amp2023.values > 500)

    stats = {
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'rule': {'window': '5x5 @10m (50m x 50m)', 'needle_codes': NEEDLE, 'broad_codes': BROAD,
                 'frac_range': [FRAC_LO, FRAC_HI], 'min_px_each': MIN_PX, 'ev_dom': EV_DOM,
                 'source': '技术文档/84 §4.2（2026-10-25 修正：按邻域针叶码分常绿/落叶，不用混交点自身物候）'},
        'funnel': funnel,
        'derived_distribution': {'all': dist_table(der), '45': dist_table(derived == 45),
                                 '46': dist_table(derived == 46)},
        'v30_crosstab': {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in ctab.to_dict('index').items()},
        'v30_group_whole_pool': pd.Series(v30_group).value_counts().to_dict(),
        'existing_6864_mixed_under_rule': mix_win,
        'discovered_from_each_cls6': disc_by_cls,
        'variant_mixcode_9192_as_evidence': {
            'points_window_mixcode_ge3': int(mixcode_maj.sum()),
            'of_which_not_derived': int((mixcode_maj & ~der).sum()),
            'of_which_center_9192': int((mixcode_maj & ~der & LUT_MIXCODE[center_code]).sum())},
        'overlap_with_existing_mixed': {
            'n_derived': int(der.sum()),
            'nearest_mixed_distance_m': near,
            'median_dist_to_mixed_derived_points': round(float(np.nanmedian(dist_nearest[der])), 1)
            if np.isfinite(np.nanmedian(dist_nearest[der])) else None,
            'median_dist_to_mixed_all_points': round(float(np.nanmedian(dist_nearest)), 1)
            if np.isfinite(np.nanmedian(dist_nearest)) else None,
            'cells_derived': int(cell_der.nunique()), 'cells_mixed': int(cell_mix.nunique()),
            'cells_overlap': len(overlap_cells),
            'share_of_derived_in_mixed_cells': round(float(cell_der.isin(overlap_cells).mean()), 4)},
        'pheno_inconsistent_flag': {'n': int(pheno_incon.sum()),
                                    'share_of_derived': round(float(pheno_incon[der].mean()), 4) if der.any() else None,
                                    'note': 'amp2023>500 但被判 45；按 doc84 §4.2 只标记不剔除'},
        'eco_balance': {str(e): {'pool': int((p.eco == e).sum()), 'derived': int((der & (p.eco == e).values).sum()),
                                 '45': int(((derived == 45) & (p.eco == e).values).sum()),
                                 '46': int(((derived == 46) & (p.eco == e).values).sum())}
                        for e in p.eco.unique()},
    }
    with open(os.path.join(OUT, 'f2b_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(jclean(stats), f, ensure_ascii=False, indent=1)

    out = pd.DataFrame({
        'uid': p.uid.values, 'lon': lon, 'lat': lat,
        'cls6_orig': p.cls6.values.astype(np.int16),
        'derived_as': derived, 'cand_status': cand_status,
        'frac_needle': frac_needle.astype(np.float32),
        'n_needle_win': n_needle, 'n_broad_win': n_broad,
        'n_evergreen_needle_win': n_ev, 'n_decid_needle_win': n_dec,
        'needle_evergreen_frac': ev_frac.astype(np.float32),
        'n_mixedcode_win': n_mixcode, 'n_valid_win': n_valid,
        'center_code': center_code, 'center_is_forest': center_forest,
        'v30_group': v30_group, 'v30_value': v30.astype(np.int32),
        'p_needle': p_needle, 'p_forest': p_forest,
        'eco': p.eco.values, 'prov': p.prov.values,
        'tier': p.tier.values, 'src_primary': p.src_primary.values,
        'amp2023': p.amp2023.values.astype(np.float32),
        'pheno_inconsistent': pheno_incon, 'derived_gate_strict': strict_gate,
        'dist_nearest_mixed_m': dist_nearest.astype(np.float32),
        'fold10km': fold.astype(np.int8),
    })
    outp = os.path.join(OUT, 'mixed_candidates.parquet')
    out.to_parquet(outp, index=False)
    emit('写出 %s（%d 行 × %d 列，%.1f MB）' % (outp, len(out), out.shape[1], os.path.getsize(outp) / 1e6))
    emit('完成，总耗时 %.1f 分钟' % ((time.time() - t00) / 60))
    print(json.dumps(jclean({'verdict': h1['verdict'],
                             'funnel': {k: v for k, v in funnel.items() if k.startswith('step')}}),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
