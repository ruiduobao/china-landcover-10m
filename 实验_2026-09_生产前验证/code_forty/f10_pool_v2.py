# -*- coding: utf-8 -*-
"""f10_pool_v2.py — 样本集 v2 构建 + 本地训练对比小实验（小实验通过才铺开）

* 输入：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（池 v1，2,060,577 点，含 AEF64/src/class_new）
        data/m3/refset_all.csv + gap_imagery/judge_*.csv + gsnx_ref_judged.csv（1,520 判读真值点）
        data/m3/layer_precision.json（层精度证据）
* 规则源（预注册；全部由已有实测证据支撑）：
  v2 相对 v1 的处置：
   (1) **删除 FCS10 灌丛层**（src='glc_fcs10_2023_shrub'，521,400 点）——干/湿区判读精度 13.9% / 16.5%
   (2) **删除 木本沼泽(14) 标签**（2,435 点）——三江判读精度 6.7%（1/15）
   (3) **生态硬约束筛选**（区域化 V 档，见 ECO_RULES）——删"生理不可能"的点
   (4) 其余类**原样保留**（含 4/5/6/7/8 针阔细分——用户明确不合并）
  评价：本地 RF(100树/leaf2/nodes5000/sqrt) 分别在 v1、v2 上训练 → 在 1,520 判读真值点上测
        24 类 OA / 9 大类 OA / 灌草合并 OA；**判据：v2 的 9 大类 OA 不低于 v1 且 24 类 OA 提升 ≥0**
* 门槛：训练子样本固定（每策略 60 万点，种子 7），评价集只用有 AEF 的判读点
* 输出：data/m3/pool_v2_delta.json（删除/改标签点表统计）、results/d2/pool_v2_exp.json
* 用法：python f10_pool_v2.py
"""
import collections
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
M3 = os.path.join(WORK, 'data', 'm3')
OUTD = os.path.join(M3, 'pool_v2')
OUTJ = os.path.join(WORK, 'results', 'd2', 'pool_v2_exp.json')
SHRUB_SRC = 'glc_fcs10_2023_shrub'
DROP_CLASSES = [14]                      # 木本沼泽
N_TRAIN = 600000
SEED = 7
NAMES = VC.V31_NAMES()

# 生态硬约束（区域化；与 doc55 V 档同源，但**限定适用区域**，避免 R07 全国误杀——见 doc55 §11）
ECO_RULES = [
    ('elev>4800 & cls in (5,6,8)', '高山林线以上'),
    ('elev>3800 & cls==4 & in_west', '常绿阔叶只在川滇西部受 3800m 限制'),
    ('elev>4500 & cls in (2,3)', '农耕上限'),
    ('elev>5000 & cls==9', '灌丛上限'),
    ('elev<2500 & cls==24', '冰雪下限'),
    ('elev>4800 & cls==7', '落叶针叶上限'),
    ('basin_sichuan & cls==7', '落叶针叶不入四川盆地'),      # R07 区域化
    ('elev<1500 & basin_sichuan & cls==12', '地衣苔藓不入盆地'),
]
SICHUAN_BASIN = (102.5, 28.5, 108.5, 32.5)
WEST = (73.0, 25.0, 105.0, 45.0)


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_pool():
    cols = ['row_id', 'lon', 'lat', 'class_new', 'src'] + VC.FEATS
    df = pd.read_parquet(POOL, columns=cols)
    df['cls'] = VC.to_v31(df['class_new'].to_numpy(int))
    return df


def in_box(lon, lat, box):
    x0, y0, x1, y1 = box
    return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)


def build_v2(df):
    """返回 (v2 索引掩膜, 删除原因统计)。"""
    keep = np.ones(len(df), bool)
    why = collections.Counter()
    m = df['src'].fillna('').astype(str) == SHRUB_SRC
    keep &= ~m
    why['删FCS10灌丛层'] = int(m.sum())
    m = df.cls.isin(DROP_CLASSES).to_numpy()
    keep &= ~m
    why['删木本沼泽'] = int(m.sum())
    el = pd.to_numeric(df.get('elev_src', np.nan), errors='coerce')
    return keep, why


def add_elev(df):
    """给池点贴 1 km DEM/bio（本地栅格，避免 GEE 逐点取数）。"""
    import rasterio
    dem_fp = os.path.join(WORK, 'data', 'eco_gate', '_dem', 'srtm1000_chn.tif')
    bio_fp = os.path.join(WORK, 'data', 'eco_gate', '_dem', 'bio1000_chn.tif')
    lon, lat = df.lon.to_numpy(), df.lat.to_numpy()
    out = {}
    for tag, fp, bands in (('elev', dem_fp, [1]), ('bio01', bio_fp, [1]), ('bio12', bio_fp, [2])):
        if not os.path.exists(fp):
            out[tag] = np.full(len(df), np.nan, 'float32')
            continue
        with rasterio.open(fp) as ds:
            arr = ds.read()
            tfm = ds.transform
        col = np.floor((lon - tfm.c) / tfm.a).astype('int64')
        row = np.floor((tfm.f - lat) / abs(tfm.e)).astype('int64')
        ok = (col >= 0) & (col < arr.shape[2]) & (row >= 0) & (row < arr.shape[1])
        v = np.full(len(df), np.nan, 'float32')
        v[ok] = arr[bands[0] - 1][row[ok], col[ok]]
        out[tag] = v
    return out


def eco_mask(df, elev):
    bad = np.zeros(len(df), bool)
    cls = df.cls.to_numpy()
    lon, lat = df.lon.to_numpy(), df.lat.to_numpy()
    basin = in_box(lon, lat, SICHUAN_BASIN)
    west = in_box(lon, lat, WEST)
    e = elev
    checks = [
        (e > 4800) & np.isin(cls, [5, 6, 8]),
        (e > 3800) & (cls == 4) & west,
        (e > 4500) & np.isin(cls, [2, 3]),
        (e > 5000) & (cls == 9),
        (e < 2500) & (cls == 24),
        (e > 4800) & (cls == 7),
        basin & (cls == 7),
        (e < 1500) & basin & (cls == 12),
    ]
    for c in checks:
        bad |= np.nan_to_num(c, nan=False).astype(bool)
    return bad


def load_eval():
    """判读真值点（有 AEF 的）。"""
    frames = []
    for f in ('expA_judged.csv', 'expA2_judged.csv', 'expB_judged.csv', 'expC_judged.csv',
              'expC2_judged.csv', 'expC3_judged.csv', 'expC4_judged.csv'):
        p = os.path.join(M3, f)
        if os.path.exists(p):
            d = pd.read_csv(p, encoding='utf-8-sig')
            frames.append(d[['point_id', 'lon', 'lat', 'truth_code'] + VC.FEATS])
    R = pd.read_csv(os.path.join(M3, 'refset_all.csv'), encoding='utf-8-sig')
    frames.append(R[['point_id', 'lon', 'lat', 'truth_code'] + VC.FEATS])
    gp = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    gj = [pd.read_csv(os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i), encoding='utf-8-sig')
          for i in range(1, 6) if os.path.exists(os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i))]
    G = gp.merge(pd.concat(gj, ignore_index=True), on='point_id')
    G = G[G.Q1 != '无法判读'].copy()
    name2code = {v: int(k) for k, v in NAMES.items()}
    G['truth_code'] = G.Q2.map(lambda s: name2code.get(str(s), 0))
    # gap 点缺 AEF → 单独取（若已有 gap AEF 文件则用之）
    gaef = os.path.join(M3, 'gap_aef.csv')
    if os.path.exists(gaef):
        G = G.merge(pd.read_csv(gaef, encoding='utf-8-sig'), on='point_id', how='left')
        frames.append(G[['point_id', 'lon', 'lat', 'truth_code'] + VC.FEATS])
    gs = pd.read_csv(os.path.join(M3, 'gsnx_ref_judged.csv'), encoding='utf-8-sig')
    gs = gs[gs.Q1 != '无法判读'].copy()
    gs['truth_code'] = gs.apply(lambda r: 10 if r.Q1 == '是' else name2code.get(str(r.Q2), 0), axis=1)
    e = pd.read_csv(os.path.join(M3, 'gsnx_ref_embed.csv'), encoding='utf-8-sig')
    gs = gs.merge(e, on='point_id', how='left')
    frames.append(gs[['point_id', 'lon', 'lat', 'truth_code'] + VC.FEATS])
    df = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates('point_id')
    df = df[df.truth_code > 0]
    df = df[df[VC.FEATS].notna().all(axis=1)]
    return df


def main():
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)
    emit('载入池 …')
    pool = load_pool()
    emit('池 %d 点；贴地形/气候 …' % len(pool))
    aux = add_elev(pool)
    elev = aux['elev']
    emit('  高程覆盖 %.1f%%（nan %d）' % (100 * float(np.isfinite(elev).mean()), int(np.isnan(elev).sum())))
    keep = np.ones(len(pool), bool)
    why = collections.Counter()
    shrub = pool['src'].fillna('').astype(str) == SHRUB_SRC
    keep &= ~shrub.to_numpy()
    why['删FCS10灌丛层'] = int(shrub.sum())
    m14 = (pool.cls == 14).to_numpy()
    keep &= ~m14
    why['删木本沼泽'] = int(m14.sum())
    eco = eco_mask(pool, elev)
    keep &= ~eco
    why['生态硬约束筛选'] = int(eco.sum())
    emit('v2 删除：%s → 保留 %d / %d（%.1f%%）' % (
        dict(why), int(keep.sum()), len(pool), 100 * keep.mean()))
    # 删类分布
    why_cls = collections.Counter(NAMES.get(str(c), str(c))
                                 for c in pool.cls.to_numpy()[eco])
    emit('  生态筛选按类：%s' % dict(why_cls.most_common(6)))
    pd.DataFrame([dict(reason=k, n=v) for k, v in why.items()]).to_csv(
        os.path.join(OUTD, 'pool_v2_delta.csv'), index=False, encoding='utf-8-sig')

    # ---- 本地训练对比 ----
    emit('载入判读真值 …')
    ev = load_eval()
    emit('评价集 %d 点：%s' % (len(ev), dict(collections.Counter(
        NAMES.get(str(c), c) for c in ev.truth_code).most_common(8))))
    Xev = ev[VC.FEATS].to_numpy('float32')
    t24 = ev.truth_code.to_numpy(int)
    rng = np.random.RandomState(SEED)
    res = {}
    for tag, mask in (('v1', np.ones(len(pool), bool)), ('v2', keep)):
        idx = np.where(mask)[0]
        if len(idx) > N_TRAIN:
            idx = rng.choice(idx, N_TRAIN, replace=False)
        X = pool.iloc[idx][VC.FEATS].to_numpy('float32')
        y = pool.iloc[idx]['cls'].to_numpy(int)
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                     max_features='sqrt', n_jobs=15, random_state=SEED).fit(X, y)
        p = clf.predict(Xev)
        tm, pm = VC.to_macro(t24), VC.to_macro(p)
        mer_t = np.where(np.isin(t24, (9, 10, 11)), -1, t24)
        mer_p = np.where(np.isin(p, (9, 10, 11)), -1, p)
        rec = dict(n_train=int(len(idx)), OA24=round(float((t24 == p).mean()), 4),
                   OA9=round(float((tm == pm).mean()), 4),
                   OA_sg=round(float((mer_t == mer_p).mean()), 4),
                   pred_shrub=int((p == 10).sum() + (p == 9).sum()),
                   pred_14=int((p == 14).sum()), pred_16=int((p == 16).sum()),
                   pred_23=int((p == 23).sum()))
        res[tag] = rec
        emit('  %s：n_train=%d → OA24 %.4f  OA9 %.4f  灌草合并 %.4f  [%.1f min]' % (
            tag, len(idx), rec['OA24'], rec['OA9'], rec['OA_sg'], (time.time() - t0) / 60))
    ok = (res['v2']['OA9'] >= res['v1']['OA9'] - 1e-9) and (res['v2']['OA24'] >= res['v1']['OA24'] - 1e-9)
    verdict = ('通过：v2 OA24 %.4f→%.4f（+%.2fpp）、OA9 %.4f→%.4f' % (
        res['v1']['OA24'], res['v2']['OA24'], 100 * (res['v2']['OA24'] - res['v1']['OA24']),
        res['v1']['OA9'], res['v2']['OA9'])) if ok else '未通过，维持 v1'
    out = dict(why=dict(why), why_cls=dict(why_cls), n_keep=int(keep.sum()), n_pool=int(len(pool)),
               results=res, verdict=verdict, n_eval=int(len(ev)))
    VC.jsave(out, OUTJ)
    emit('判定：%s' % verdict)
    # 保存 v2 点表（row_id 列表，供 GEE 侧筛除）
    np.save(os.path.join(OUTD, 'pool_v2_dropped_rowids.npy'), pool['row_id'].to_numpy()[~keep])
    emit('→ %s；删除点 row_id 已存 pool_v2/' % OUTJ)


if __name__ == '__main__':
    main()
