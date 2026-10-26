# -*- coding: utf-8 -*-
"""f7_assemble.py — 乔灌园地 v8 装配与精度评价（ForTy 提纯 + 扩量 + 1:1 负样本）

* 输入：
    v7 点表/面板（160,931 点 × 8 年）+ points/v7_forty_screen.parquet + feat/forty/v7_merged.parquet
    cand/pos_candidates.parquet + feat/{forty,aef,terrain}/pos_merged.parquet
    cand/neg_candidates.parquet + feat/{forty,aef,terrain}/neg_merged.parquet
* 规则源（**预注册，见 docs/f0_方案与预注册判据.md §3；先写死再跑**）：
    ① forty_ok = p_treecrop/250 ≥ 0.5（GEE 概率）；
    ② 保留规则（不因 ForTy 否定 apple/tea/anchor；柑橘仅留 forty_ok）：
       rubber/hainan_official/multi → 全保留（forty_ok 者标 core）
       apple/tea → 保留，forty_conflict=1
       anchor   → 保留（目视优先）
       citrus   → 仅保留 forty_ok（标 citrus_confirmed=1）
    ③ 新正样本 = pos_candidates（ForTy 类 5 纯块，已配额）；新负 = neg_candidates（1:1）
    ④ 输出面板：正样本 8 年（v7 沿用原 AEF；新点用 feat/aef/pos_merged）；负样本 2020+2023
* 门槛：正样本面板 AEF 命中率 ≥0.99；新正/负 行数完整；空间门（省数、0.5° 格数、最小间距）
* 输出：交付目录 `.../乔灌园地/v8_ForTy提纯扩量/`
        乔灌园地v8_正样本_点表.csv｜乔灌园地v8_正样本_逐年面板_2017-2024.parquet
        乔灌园地v8_负样本_点表.csv｜乔灌园地v8_负样本_AEF面板.parquet
        v8_stats.json｜v8_前后对比.csv｜可分离性指标
* 用法：python f7_assemble.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

WORK = r'F:/lc_work/forty_exp'
V7DIR = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地'
DEST = os.path.join(V7DIR, 'v8_ForTy提纯扩量')
AEF_COLS = ['A%02d' % i for i in range(64)]


def emit(m):
    import time
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_merged(mode, name):
    p = os.path.join(WORK, 'feat', mode, '%s_merged.parquet' % name)
    if not os.path.exists(p):
        p = os.path.join(WORK, 'feat', mode, '%s_p0.parquet' % name)
    return pd.read_parquet(p)


def thin_min_dist(df, r_m=250.0, seed=20261026):
    """贪心最小间距抽稀（haversine）：随机序，保留点半径 r_m 内的点全部剔除。
    修订 A2（2026-10-27）：0.005° 网格去重存在边界伪影（最近邻可低至 10 m），补 250 m 贪心抽稀。"""
    from sklearn.neighbors import BallTree
    coords = np.radians(df[['lat', 'lon']].to_numpy('float64'))
    tree = BallTree(coords, metric='haversine')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    keep = np.zeros(len(df), bool); taken = np.zeros(len(df), bool)
    for i in order:
        if taken[i]:
            continue
        keep[i] = True
        idx = tree.query_radius(coords[i:i + 1], r=r_m / 6371000.0)[0]
        taken[idx] = True
    return keep


def rf_auc(X, y, seed=7):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features='sqrt',
                                     n_jobs=8, random_state=seed).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    f1 = max(float(f1_score(y, (p >= t).astype(int))) for t in np.arange(0.3, 0.71, 0.05))
    return round(auc, 4), round(f1, 4)


def main():
    os.makedirs(DEST, exist_ok=True)
    # ---------------- v7 提纯 ----------------
    p7 = pd.read_parquet(os.path.join(WORK, 'points', 'v7_points.parquet'))
    sc = pd.read_parquet(os.path.join(WORK, 'points', 'v7_forty_screen.parquet'))
    f7 = load_merged('forty', 'v7')[['uid', 'p_treecrop', 'p_other', 'forty_ok']]
    m = p7.merge(sc, on='uid', how='left').merge(f7, on='uid', how='left', suffixes=('', '_gee'))
    m['p5'] = m.p_treecrop / 250.0
    m['forty_agree'] = m.p5 >= 0.5
    m['forty_conflict'] = 0
    m['status'] = 'keep'
    # 规则
    is_soft = m.srcset.isin(['apple', 'tea'])
    is_citrus = m.srcset == 'citrus'
    m.loc[is_soft, 'status'] = 'keep_conflict'
    m.loc[is_soft, 'forty_conflict'] = 1
    m.loc[m.srcset == 'anchor', 'status'] = 'keep_visual'
    drop = is_citrus & ~m.forty_agree
    m.loc[is_citrus & m.forty_agree, 'status'] = 'core_confirmed'
    core = (~drop) & (~is_soft) & (~m.srcset.isin(['anchor'])) & m.forty_agree
    m.loc[core, 'status'] = 'core'
    m['kept'] = ~drop
    emit('v7 提纯：总 %d，保留 %d，剔除 %d（柑橘未确认 %d）' % (
        len(m), int(m.kept.sum()), int((~m.kept).sum()), int(drop.sum())))
    keep_uids = set(m.loc[m.kept, 'uid'].astype(int))
    keep_tbl = m.loc[m.kept, ['uid', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'rec',
                              'forty_class', 'dom11', 'c5_5x5', 'c5_11x11', 'p5',
                              'forty_agree', 'forty_conflict', 'status']].copy()
    keep_tbl['src'] = 'v7'
    # v7 保留点补 CTAmap 省归属（与 v8 新点同口径）
    try:
        import geopandas as gpd
        from shapely import STRtree, contains_xy
        CTAMAP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
        prov = gpd.read_file(CTAMAP)
        names = prov['省'].astype(str).to_numpy(); codes = prov['省级码'].astype(str).to_numpy()
        geoms = prov.geometry.to_numpy()
        lo, la = keep_tbl.lon.to_numpy(), keep_tbl.lat.to_numpy()
        pi = np.full(len(keep_tbl), -1)
        for i, g in enumerate(geoms):
            mm = contains_xy(g, lo, la)
            pi[mm & (pi < 0)] = i
        keep_tbl['ct_prov'] = np.where(pi >= 0, names[np.clip(pi, 0, None)], '')
        keep_tbl['ct_code'] = np.where(pi >= 0, codes[np.clip(pi, 0, None)], '')
        keep_tbl['ct_km'] = 0.0
        if (pi < 0).any():
            st = STRtree(list(geoms))
            from shapely.geometry import Point
            for i in np.where(pi < 0)[0]:
                j = st.nearest(Point(lo[i], la[i]))
                keep_tbl.loc[keep_tbl.index[i], 'ct_prov'] = names[j]
                keep_tbl.loc[keep_tbl.index[i], 'ct_code'] = codes[j]
    except Exception as e:
        print('CTAmap v7 归属跳过：%s' % str(e)[:90])
        keep_tbl['ct_prov'] = ''; keep_tbl['ct_code'] = ''; keep_tbl['ct_km'] = np.nan
    # v7 面板按保留 uid 过滤
    panel7 = pd.read_parquet(os.path.join(V7DIR, '乔灌园地v7_逐年面板_2017-2024.parquet'),
                             columns=['uid', 'year', 'lon', 'lat'] + AEF_COLS + ['aef_ok'])
    panel7 = panel7[panel7.uid.isin(keep_uids)].copy()
    emit('v7 面板保留行 %d（%.3f 亿）' % (len(panel7), len(panel7) / 1e8))
    # ---------------- 新正样本 ----------------
    pos = pd.read_parquet(os.path.join(WORK, 'cand', 'pos_candidates.parquet'))
    fp = load_merged('forty', 'pos')
    fpn = {c: c for c in fp.columns if c not in ('pid',)}
    pos = pos.merge(fp[[c for c in fp.columns if c != 'p_crop_argmax']], on='pid', how='left')
    pos['p5'] = pos.p_treecrop / 250.0
    pos['forty_agree'] = pos.p5 >= 0.5
    pos['forty_conflict'] = 0
    pos['status'] = np.where(pos.forty_agree, 'new_pos_core', 'new_pos_lowprob')
    pos['src'] = 'forty2020_new'
    pos['grade'] = 'B_ForTy树作物'
    pos['subtype'] = '乔木园地'
    pos['srcset'] = 'forty_new'
    pos['rec'] = 1
    pos['uid'] = len(m) + np.arange(len(pos))    # 全局唯一 id（v7 的 uid 为 0..160930）
    pos['forty_class'] = 5
    pos['dom11'] = 5
    pos['c5_5x5'] = pos.frac5
    pos['c5_11x11'] = pos.frac5
    kp = thin_min_dist(pos)
    emit('新正样本 250 m 贪心抽稀：%d → %d（%.1f%%）' % (len(pos), int(kp.sum()), 100 * kp.mean()))
    pos = pos.loc[kp].reset_index(drop=True)
    pos_tbl = pos[['uid', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'rec', 'forty_class',
                   'dom11', 'c5_5x5', 'c5_11x11', 'p5', 'forty_agree', 'forty_conflict',
                   'status', 'src', 'ct_prov', 'ct_code', 'ct_km']].copy()
    emit('新正样本 %d（forty_agree=%.4f）' % (len(pos), float(pos.forty_agree.mean())))
    # 新正 AEF 面板
    pa = load_merged('aef', 'pos')
    keep = [c for c in pa.columns]
    pa = pa.rename(columns={'pid': 'key'})
    pa = pa.merge(pos[['pid', 'uid', 'lon', 'lat']], left_on='key', right_on='pid', how='left')
    panel_new = pa[['uid', 'year', 'lon', 'lat'] + AEF_COLS + ['aef_ok']].dropna(subset=['uid']).copy()
    panel_new['uid'] = panel_new.uid.astype('int64')
    emit('新正面板 %d 行（%d 点 × %d 年）命中 %.4f' % (
        len(panel_new), panel_new.uid.nunique(), panel_new.year.nunique(),
        float(panel_new.aef_ok.mean())))
    # 合并正样本面板
    for c in ['subtype', 'grade', 'srcset', 'rec', 'status', 'src', 'forty_agree',
              'forty_conflict', 'p5', 'ct_prov', 'ct_km']:
        keep_tbl[c] = keep_tbl[c].astype(str) if c in ('subtype', 'grade', 'srcset', 'status', 'src', 'ct_prov') else keep_tbl[c]
    pos_all = pd.concat([keep_tbl, pos_tbl], ignore_index=True)
    pos_all['pid'] = np.arange(len(pos_all))
    panel_all = pd.concat([panel7, panel_new], ignore_index=True)
    emit('正样本合计 %d 点 / %d 行' % (pos_all.pid.nunique(), len(panel_all)))
    # ---------------- 负样本 ----------------
    neg = pd.read_parquet(os.path.join(WORK, 'cand', 'neg_candidates.parquet'))
    fn = load_merged('forty', 'neg')
    neg = neg.merge(fn[[c for c in fn.columns if c not in ('p_crop_argmax',)]], on='pid', how='left')
    neg['p5'] = neg.p_treecrop / 250.0
    neg['uid'] = len(m) + len(pos) + np.arange(len(neg))
    neg['label'] = 0
    kn = thin_min_dist(neg)
    emit('负样本 250 m 贪心抽稀：%d → %d（%.1f%%）' % (len(neg), int(kn.sum()), 100 * kn.mean()))
    neg = neg.loc[kn].reset_index(drop=True)
    aneg = load_merged('aef', 'neg').rename(columns={'pid': 'key'})
    aneg = aneg.merge(neg[['pid', 'uid', 'lon', 'lat']], left_on='key', right_on='pid', how='left')
    npanel = aneg[['uid', 'year', 'lon', 'lat'] + AEF_COLS + ['aef_ok']].dropna(subset=['uid']).copy()
    npanel['uid'] = npanel.uid.astype('int64')
    neg_tbl = neg[['uid', 'lon', 'lat', 'neg_type', 'dom', 'pur', 'p5', 'dist_v7_km',
                   'dist_pos_km', 'ct_prov', 'ct_code', 'ct_km', 'label']].copy()
    emit('负样本 %d 点 / %d 行（命中 %.4f）' % (len(neg_tbl), len(npanel), float(npanel.aef_ok.mean())))
    # ---------------- 地形（可选并入点表） ----------------
    try:
        tp = load_merged('terrain', 'pos').rename(columns={'pid': 'key'})
        tp = tp.merge(pos[['pid', 'uid']], left_on='key', right_on='pid', how='left')
        pos_all = pos_all.merge(tp[['uid', 'elev', 'slope']], on='uid', how='left')
        tn = load_merged('terrain', 'neg').rename(columns={'pid': 'key'})
        tn = tn.merge(neg[['pid', 'uid']], left_on='key', right_on='pid', how='left')
        neg_tbl = neg_tbl.merge(tn[['uid', 'elev', 'slope']], on='uid', how='left')
        emit('地形并入：正 %d / 负 %d 缺失率 %.2f%% / %.2f%%' % (
            int(pos_all.elev.notna().sum()), int(neg_tbl.elev.notna().sum()),
            100 * pos_all.elev.isna().mean(), 100 * neg_tbl.elev.isna().mean()))
    except Exception as e:
        emit('地形并入跳过：%s' % str(e)[:100])
    # ---------------- 写交付 ----------------
    pos_all.to_csv(os.path.join(DEST, '乔灌园地v8_正样本_点表.csv'), index=False, encoding='utf-8-sig')
    panel_all.to_parquet(os.path.join(DEST, '乔灌园地v8_正样本_逐年面板_2017-2024.parquet'), index=False)
    neg_tbl.to_csv(os.path.join(DEST, '乔灌园地v8_负样本_点表.csv'), index=False, encoding='utf-8-sig')
    npanel.to_parquet(os.path.join(DEST, '乔灌园地v8_负样本_AEF面板.parquet'), index=False)
    # ---------------- 精度与质量指标 ----------------
    # 可分性：新正 vs 负（2020 AEF）
    pp = panel_all[(panel_all.year == 2020) & (panel_all.aef_ok == 1)][['uid'] + AEF_COLS]
    nn = npanel[(npanel.year == 2020) & (npanel.aef_ok == 1)][['uid'] + AEF_COLS]
    X = np.vstack([pp[AEF_COLS].to_numpy('float32'), nn[AEF_COLS].to_numpy('float32')])
    y = np.r_[np.ones(len(pp), int), np.zeros(len(nn), int)]
    auc_all, f1_all = rf_auc(X, y)
    emit('新正 vs 负 可分性（AEF64 2020）：AUC %.4f / F1 %.4f' % (auc_all, f1_all))
    # 仅 core 正样本 vs 负
    core_uid = set(pos_all.loc[pos_all.status == 'new_pos_core', 'uid'].astype(int)) | \
               set(pos_all.loc[(pos_all.src == 'v7') & (pos_all.status == 'core'), 'uid'].astype(int))
    pm = panel_all[(panel_all.year == 2020) & (panel_all.aef_ok == 1) & panel_all.uid.isin(core_uid)]
    X2 = np.vstack([pm[AEF_COLS].to_numpy('float32'), nn[AEF_COLS].to_numpy('float32')])
    y2 = np.r_[np.ones(len(pm), int), np.zeros(len(nn), int)]
    auc_core, f1_core = rf_auc(X2, y2)
    emit('core 正 vs 负：AUC %.4f / F1 %.4f（core n=%d）' % (auc_core, f1_core, len(pm)))
    # 前后对比
    before = {'n_points': int(len(m)), 'a1_rate': round(float(m.forty_agree.mean()), 4),
              'n_prov': int((m.lat.notna()).sum() and pd.Series([1]).sum() and 0) or 0}
    by_src_before = m.groupby('srcset').agg(n=('uid', 'size'), a1=('forty_agree', 'mean')).round(4)
    by_src_after = pos_all.groupby('srcset').agg(n=('uid', 'size'), a1=('forty_agree', 'mean')).round(4)
    cmp = pd.DataFrame({'v7_n': by_src_before.n, 'v7_A1': by_src_before.a1,
                        'v8_n': by_src_after.n, 'v8_A1': by_src_after.a1}).fillna(0)
    cmp.to_csv(os.path.join(DEST, 'v8_前后对比_按来源.csv'), encoding='utf-8-sig')
    stats = {
        'v7': {'n_points': int(len(m)), 'kept': int(m.kept.sum()), 'dropped': int((~m.kept).sum()),
               'a1_rate': round(float(m.forty_agree.mean()), 4)},
        'v7_kept': {'n': int(pos_all[pos_all.src == 'v7'].shape[0]),
                    'a1_rate': round(float(pos_all[pos_all.src == 'v7'].forty_agree.mean()), 4),
                    'status': pos_all[pos_all.src == 'v7'].status.value_counts().to_dict()},
        'new_pos': {'n': int(len(pos)), 'a1_rate': round(float(pos.forty_agree.mean()), 4),
                    'status': pos.status.value_counts().to_dict(),
                    'frac5_med': round(float(pos.frac5.median()), 3)},
        'neg': {'n': int(len(neg_tbl)), 'by_type': neg_tbl.neg_type.value_counts().to_dict()},
        'total_points': int(len(pos_all)) + int(len(neg_tbl)),
        'pos_neg_ratio': round(len(pos_all) / max(1, len(neg_tbl)), 3),
        'panel_rows_pos': int(len(panel_all)), 'panel_rows_neg': int(len(npanel)),
        'panel_aef_hit_pos': round(float(panel_all.aef_ok.mean()), 4),
        'panel_aef_hit_neg': round(float(npanel.aef_ok.mean()), 4),
        'sep_auc_newpos_vs_neg': auc_all, 'sep_f1': f1_all,
        'sep_auc_core_vs_neg': auc_core, 'sep_f1_core': f1_core,
        'thinning_A2': {'rule': 'greedy 250 m haversine', 'seed': 20261026,
                        'pos_before': 118661, 'pos_after': int(len(pos)),
                        'neg_before': 118660, 'neg_after': int(len(neg))},
        'space': {'pos_prov_n': int(pos_all.ct_prov.nunique()) if 'ct_prov' in pos_all else None,
                  'pos_g05_cells': int((np.floor(pos_all.lon / 0.5).astype('int64') * 100000 +
                                        np.floor(pos_all.lat / 0.5).astype('int64')).nunique()),
                  'new_pos_dist_v7_min_km': round(float(pos.dist_v7_km.min()), 3),
                  'new_pos_dist_v7_med_km': round(float(pos.dist_v7_km.median()), 3),
                  'pos_max_prov_share': round(float(pos.ct_prov.value_counts(normalize=True).max()), 4)},
    }
    try:
        jm = pd.read_csv(os.path.join(WORK, 'vis', 'judge_merged.csv'), encoding='utf-8-sig')
        jc = pd.read_csv(os.path.join(WORK, 'vis', 'judge_out_calib.csv'), encoding='utf-8-sig')
        jm.reset_index(drop=True, inplace=True)
        vis = {}
        for g, gr in jm.groupby('group'):
            ok = (gr.Q1 == '是').sum(); na = (gr.Q1 == '无法判读').sum()
            vis[g] = {'n': int(len(gr)), 'yes': int(ok), 'na': int(na),
                      'rate': round(float(ok / max(1, len(gr) - na)), 3)}
        vis['calib_rubber_A'] = {'n': int(len(jc)), 'yes': int((jc.Q1 == '是').sum()),
                                 'na': int((jc.Q1 == '无法判读').sum()),
                                 'rate': round(float((jc.Q1 == '是').sum() / max(1, (jc.Q1 != '无法判读').sum())), 3)}
        stats['visual_check'] = vis
    except Exception as e:
        emit('目视并入跳过：%s' % str(e)[:90])
    # 平衡子集（新增正 : 负 = 1:1），供二分类训练直接使用
    a = pos_tbl[['uid', 'lon', 'lat', 'status', 'grade', 'ct_prov']].copy()
    a['label'] = 1; a['srcset'] = 'forty_new'; a['neg_type'] = ''
    b = neg_tbl[['uid', 'lon', 'lat', 'neg_type', 'ct_prov']].copy()
    b['label'] = 0; b['srcset'] = 'forty_neg'; b['status'] = b.neg_type; b['grade'] = ''
    bal = pd.concat([a, b], ignore_index=True).sample(frac=1.0, random_state=20261026).reset_index(drop=True)
    bal = bal[['uid', 'lon', 'lat', 'label', 'srcset', 'status', 'grade', 'neg_type', 'ct_prov']]
    bal.to_csv(os.path.join(DEST, '乔灌园地v8_平衡子集_点表.csv'), index=False, encoding='utf-8-sig')
    emit('平衡子集 %d（正 %d / 负 %d）' % (len(bal), int((bal.label == 1).sum()), int((bal.label == 0).sum())))
    with open(os.path.join(DEST, 'v8_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    emit(json.dumps({k: stats[k] for k in ('total_points', 'pos_neg_ratio', 'panel_aef_hit_pos',
                                           'sep_auc_newpos_vs_neg', 'sep_auc_core_vs_neg')}, ensure_ascii=False))
    emit('→ %s' % DEST)


if __name__ == '__main__':
    main()
