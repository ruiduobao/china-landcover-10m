# -*- coding: utf-8 -*-
"""r2_veg30_sample.py — 30 m 植被类型栅格 → 高几何质量针叶/阔叶候选样本点（逐瓦、可续跑）

* 输入 :
    F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif     (EPSG:3857, uint16, nodata=65535, 30 m)
    F:/lc_work/nb_national_2026/opt/r1_code_map.csv                (863 行 Value→映射类/群系 码表)
    F:/lc_work/nb_national_2026/opt/r1_by_tile.csv                 (313 瓦 30m 像元统计)
    F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv   (313 瓦经纬度边界)
    F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json (DataV 省界, 35 feature)
    F:/lc_work/nb_national_2026/opt/r1_points.parquet              (v2 现有 110,653 点，用于去重)
* 规则源 : 用户 2026-10-24 指令 L1–L7（逐条实现）
    L1 取类  映射类 in {needle, broad}
    L2 腐蚀  binary_erosion 5x5；读窗带 2 px halo，避免瓦边人为削边
    L3 连通  ndimage.label 8 连通，保留 >= 20 px（≈1.8 ha 名义）
    L4 撒点  3x3 像元块（全局对齐，块内随机偏移）→ 每块最多 1 点；同类最小间距 0.015°；
             每瓦每类 <= 500 点
    L5 去重  v2 点 550 m（=0.005°）内有则弃
    L6 境内  DataV 省界逐 feature 判定（STRtree，不 unary_union）
    L7 群系  按 code30 取 群系/植被型/类型组
* 门槛 : 见 L2–L5；单进程、内存 <= 12 GB（逐瓦窗口读取，绝不整幅读）
* 输出 :
    F:/lc_work/nb_national_2026/opt/needle_broad_veg30_v1.parquet
    F:/lc_work/nb_national_2026/opt/veg30_v1_stats.json
    F:/lc_work/nb_national_2026/opt/veg30_v1_work/tiles/<tile>.parquet|.json   (checkpoint)
* 用法 : python r2_veg30_sample.py [--force] [--tiles T0001,T0002] [--limit N] [--assemble-only]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- 路径常量
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
BASE = r'F:/lc_work/nb_national_2026'
OPT = os.path.join(BASE, 'opt')
VEG30 = os.path.join(BASE, 'veg30', '中国植被类型分布_30m.tif')
CMAP = os.path.join(OPT, 'r1_code_map.csv')
BYTILE = os.path.join(OPT, 'r1_by_tile.csv')
TILES = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
BOUND = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
V2PTS = os.path.join(OPT, 'r1_points.parquet')
OUT_PQ = os.path.join(OPT, 'needle_broad_veg30_v1.parquet')
OUT_JSON = os.path.join(OPT, 'veg30_v1_stats.json')
WORK = os.path.join(OPT, 'veg30_v1_work')
WORK_TILES = os.path.join(WORK, 'tiles')

# ---------------------------------------------------------------- 参数（冻结）
HALO = 2                 # 读窗 halo 像元（防瓦边人为腐蚀）
ERODE_K = 5              # L2 腐蚀窗口
MIN_COMP_PX = 20         # L3 最小连通域
BLOCK = 3                # L4 3x3 像元块 ≈ 90 m
CAP_PER_CLASS_TILE = 500 # L4 每瓦每类上限
CAND_MAX = 400000        # L4 每瓦每类进入贪心的候选块上限（大瓦抽稀，防 Python 级长循环）
MINSEP_DEG = 0.015       # L4 同类最小间距（度）> 0.01° 格对角线 0.01414°
GRID_DEDUP = 0.01        # L4 全局 0.01° 同类 1 点（跨瓦兜底）
V2_DEDUP_DEG = 0.005     # L5 去重盒半宽（度，L∞）；≈550 m（赤道 556 m，北纬 53° 经向 334 m）
R_EARTH = 6371008.8
CLS_NAMES = {1: 'needle', 2: 'broad'}
NODATA = 65535


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- 码表
def read_code_map():
    m = pd.read_csv(CMAP, encoding='utf-8-sig')
    m['Value'] = m['Value'].astype(int)
    lut = np.zeros(NODATA + 1, dtype=np.uint8)
    for i, c in enumerate(['needle', 'broad', 'mixed', 'bamboo', 'nonforest'], start=1):
        lut[m.loc[m['映射类'] == c, 'Value'].to_numpy()] = i
    meta = {}
    for r in m.itertuples(index=False):
        meta[int(r.Value)] = (str(r.类型组), str(r.植被型),
                              '' if pd.isna(r.群系) else str(r.群系),
                              int(r.歧义级别) if not pd.isna(r.歧义级别) else 0)
    mapcls = dict(zip(m.Value, m['映射类']))
    return m, lut, meta, mapcls


# ---------------------------------------------------------------- L4 撒点：同类间距贪心
def pick_spreaded(lon, lat, cap, minsep, rng):
    """随机顺序贪心：同类点（经纬度平面、圆心半径 minsep）互相排斥，直到 cap。

    minsep=0.015° > 0.01°² 的对角线 0.01414°，故该规则**必然**保证
    "0.01°×0.01° 格内同类最多 1 点"（L4 的硬要求），且不需要另设网格。
    """
    n = lon.size
    if n == 0 or cap <= 0:
        return np.empty(0, dtype=np.int64)
    order = rng.permutation(n)
    inv = 1.0 / minsep
    ms2 = minsep * minsep
    grid = {}
    acc = []
    for i in order:
        lo, la = lon[i], lat[i]
        ci, cj = int(np.floor(lo * inv)), int(np.floor(la * inv))
        ok = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                lst = grid.get((ci + di, cj + dj))
                if not lst:
                    continue
                for k in lst:
                    d1 = lon[k] - lo
                    d2 = lat[k] - la
                    if d1 * d1 + d2 * d2 < ms2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((ci, cj), []).append(i)
            acc.append(i)
            if len(acc) >= cap:
                break
    return np.asarray(acc, dtype=np.int64)


# ---------------------------------------------------------------- 单瓦
def process_tile(tile, x0, y0, x1, y1, ds, T, LUT, seed):
    tr = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
    tb = transform_bounds('EPSG:4326', 'EPSG:3857', x0, y0, x1, y1)
    w = from_bounds(*tb, transform=T)
    c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
    c1 = min(ds.width, int(np.ceil(w.col_off + w.width)))
    r1 = min(ds.height, int(np.ceil(w.row_off + w.height)))
    c0, r0 = max(0, c0), max(0, r0)
    cnt = {'tile': tile, 'x0': float(x0), 'y0': float(y0), 'x1': float(x1), 'y1': float(y1)}
    if c1 <= c0 or r1 <= r0:
        cnt.update({'note': 'empty window', 'win_px': 0, 'valid_px': 0})
        return pd.DataFrame(columns=['lon', 'lat', 'tile', 'leaf']), cnt

    # 带 halo 读取（halo 用于腐蚀正确性；腐蚀在 halo 上做，核心区不受人为削边影响）
    oy, ox = min(HALO, r0), min(HALO, c0)
    rr0, cc0 = r0 - oy, c0 - ox
    rr1, cc1 = min(ds.height, r1 + HALO), min(ds.width, c1 + HALO)
    a = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
    h, wd = r1 - r0, c1 - c0
    cnt['win_px'] = int(h * wd)
    a_core = a[oy:oy + h, ox:ox + wd]
    cnt['valid_px'] = int((a_core != NODATA).sum())
    cls_f = LUT[a]              # 全 halo 分类图
    del a
    cls_c = LUT[a_core]

    rng = np.random.default_rng(seed)
    rows = []

    for cv, cname in CLS_NAMES.items():
        cpx = int((cls_c == cv).sum())
        cnt['%s_L1_px' % cname] = cpx
        for k in ('L2_px', 'L3_px', 'L3_ncomp', 'L4_blocks', 'L4_pts'):
            cnt['%s_%s' % (cname, k)] = 0
        if cpx == 0:
            continue

        # L2 腐蚀（halo 上做，再裁核心）
        mask_f = cls_f == cv
        me = ndimage.binary_erosion(mask_f, structure=np.ones((ERODE_K, ERODE_K), bool), border_value=0)
        del mask_f
        me = me[oy:oy + h, ox:ox + wd]
        n2 = int(me.sum())
        cnt['%s_L2_px' % cname] = n2
        if n2 == 0:
            continue

        # L3 8 连通 >= 20 px
        lab, _ = ndimage.label(me, structure=np.ones((3, 3), np.uint8))
        sizes = np.bincount(lab.ravel())
        keep = sizes >= MIN_COMP_PX
        keep[0] = False
        cnt['%s_L3_ncomp' % cname] = int(keep.sum())
        mf = keep[lab]
        del lab, me, sizes, keep
        n3 = int(mf.sum())
        cnt['%s_L3_px' % cname] = n3
        if n3 == 0:
            continue

        # L4 3x3 块：按**全局像元网格**对齐（跨瓦一致），块内 9 像元必须全保留
        trow, tcol = (-r0) % BLOCK, (-c0) % BLOCK   # 首行/首列到下一个全局块边界的距离
        mfa = mf[trow:, tcol:]
        del mf
        H, W = mfa.shape
        if H < BLOCK or W < BLOCK:
            cnt['%s_L4_pts' % cname] = 0
            continue
        H -= H % BLOCK
        W -= W % BLOCK
        sub = mfa[:H, :W].reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK)
        block_ok = sub.all(axis=(1, 3))
        del sub, mfa
        br, bc = np.nonzero(block_ok)
        del block_ok
        nb = int(br.size)
        cnt['%s_L4_blocks' % cname] = nb
        if nb == 0:
            continue

        # 候选抽稀：大瓦可用块可达数百万，先随机取 CAND_MAX 个候选再贪心（避免 Python 级长循环）
        if nb > CAND_MAX:
            pick = rng.choice(nb, size=CAND_MAX, replace=False)
            br, bc = br[pick], bc[pick]
            nb = CAND_MAX

        # 块中心全局像元坐标
        grow = (r0 + trow) + br * BLOCK + 1.0
        gcol = (c0 + tcol) + bc * BLOCK + 1.0
        jit = rng.uniform(-1.0, 1.0, size=(nb, 2))
        rowf = grow + jit[:, 1]
        colf = gcol + jit[:, 0]
        lon, lat = tr.transform(T.c + colf * T.a, T.f + rowf * T.e)
        sel = pick_spreaded(np.asarray(lon), np.asarray(lat), CAP_PER_CLASS_TILE, MINSEP_DEG, rng)
        cnt['%s_L4_pts' % cname] = int(sel.size)
        if sel.size:
            # 直接回取该点所在像元的 30m 值（像元索引 floor == rasterio.sample 口径）
            rr = np.clip(np.floor(np.asarray(rowf)[sel]).astype(int), r0, r1 - 1)
            ccol = np.clip(np.floor(np.asarray(colf)[sel]).astype(int), c0, c1 - 1)
            code30 = a_core[rr - r0, ccol - c0]
            rows.append(pd.DataFrame({'lon': np.asarray(lon)[sel], 'lat': np.asarray(lat)[sel],
                                      'tile': tile, 'leaf': cname,
                                      'code30': code30.astype('int64')}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['lon', 'lat', 'tile', 'leaf'])
    return df, cnt


# ---------------------------------------------------------------- 边界
def build_boundary():
    import shapely
    from shapely.geometry import shape
    d = json.load(open(BOUND, encoding='utf-8'))
    geoms, names = [], []
    for f in d['features']:
        geoms.append(shape(f['geometry']))
        names.append(f['properties'].get('name') or '南海诸岛(JD)')
    for g in geoms:
        shapely.prepare(g)
    return shapely, shapely.STRtree(geoms), np.array(names, dtype=object)


def nearest_v2(lon, lat, v2lon, v2lat, k=32):
    """返回 (最近 v2 点真实距离 m, 最近 v2 点 max(|dlon|,|dlat|) 度)。

    度空间 k 近邻取候选（k=32 足以覆盖 0.005° 盒内的所有候选），再分别算
    haversine 真距与 L∞ 度距（后者用于 L5 的 0.005° 盒判据）。
    """
    tree = cKDTree(np.column_stack([v2lon, v2lat]))
    k = min(k, len(v2lon))
    dd, ii = tree.query(np.column_stack([lon, lat]), k=k, workers=1)
    ii = np.atleast_2d(ii)
    la1 = np.deg2rad(lat)[:, None]
    lo1 = np.deg2rad(lon)[:, None]
    lon1 = lon[:, None]
    lat1 = lat[:, None]
    dmin = np.full(len(lon), np.inf)
    linf = np.full(len(lon), np.inf)
    for j in range(ii.shape[1]):
        jlon = v2lon[ii[:, j]][:, None]
        jlat = v2lat[ii[:, j]][:, None]
        la2 = np.deg2rad(jlat)
        lo2 = np.deg2rad(jlon)
        hh = (np.sin((la2 - la1) / 2) ** 2
              + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
        dmin = np.minimum(dmin, (2 * R_EARTH * np.arcsin(np.sqrt(np.clip(hh, 0, 1))))[:, 0])
        linf = np.minimum(linf, np.maximum(np.abs(jlon - lon1), np.abs(jlat - lat1))[:, 0])
    return dmin, linf


def add_purity(pts, lut, radius=2):
    """逐点算 5x5 邻域纯度：pur_leaf（同类叶型占比）与 pur_gen（同 code30 占比）。

    这是**诊断列**，不参与筛选（L2 的类级 5x5 腐蚀已保证 pur_leaf=1；
    pur_gen<1 说明该点位于类内不同群系的交界，属 30m 图斑内部群系混写）。
    按瓦分组顺序读小窗，块缓存命中率高。
    """
    pur_leaf = np.ones(len(pts), dtype='float32')
    pur_gen = np.ones(len(pts), dtype='float32')
    r = int(radius)
    with rasterio.open(VEG30) as ds:
        T = ds.transform
        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        by_tile = pts.groupby('tile').indices
        for tile, idx in by_tile.items():
            X, Y = tr.transform(pts.lon.to_numpy()[idx], pts.lat.to_numpy()[idx])
            col = np.floor((X - T.c) / T.a).astype(int)
            row = np.floor((Y - T.f) / T.e).astype(int)
            for j, k in enumerate(idx):
                rr, cc = row[j], col[j]
                if rr - r < 0 or cc - r < 0 or rr + r >= ds.height or cc + r >= ds.width:
                    continue
                w = ds.read(1, window=Window(cc - r, rr - r, 2 * r + 1, 2 * r + 1))
                n = w.size
                pur_leaf[k] = float((lut[w] == lut[pts.code30.iat[k]]).sum()) / n
                pur_gen[k] = float((w == pts.code30.iat[k]).sum()) / n
    pts['pur_leaf'] = pur_leaf
    pts['pur_gen'] = pur_gen
    return pts


def cellkey(d):
    return (np.floor(d.lon.to_numpy() / GRID_DEDUP).astype('int64') * 100000
            + np.floor(d.lat.to_numpy() / GRID_DEDUP).astype('int64'))


def seam_thin(df, minsep_deg, seed=17):
    """同类最小距离 < minsep_deg 的点对，随机保留其一（迭代到收敛）。

    独立于 0.01° 网格：跨瓦接缝处的近邻对落在相邻格，网格去重管不到。
    """
    out = df.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    out['_pri'] = rng.random(len(out))
    while True:
        drop = np.zeros(len(out), bool)
        for leaf in ('needle', 'broad'):
            m = (out.leaf == leaf).to_numpy()
            if m.sum() < 2:
                continue
            sub = out[m]
            tree = cKDTree(np.column_stack([sub.lon, sub.lat]))
            pairs = tree.query_pairs(minsep_deg, output_type='ndarray')
            if not len(pairs):
                continue
            a, b = pairs[:, 0], pairs[:, 1]
            pri = sub._pri.to_numpy()
            loser = np.where(pri[a] < pri[b], b, a)   # 优先级低者被弃
            idx = np.nonzero(m)[0]
            drop[idx[np.unique(loser)]] = True
        if not drop.any():
            break
        out = out[~drop].reset_index(drop=True)
    return out.drop(columns=['_pri'])


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--tiles', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--assemble-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(WORK_TILES, exist_ok=True)
    t_start = time.time()
    _, LUT, meta, mapcls = read_code_map()
    by = pd.read_csv(BYTILE, encoding='utf-8-sig').set_index('tile')
    tl = pd.read_csv(TILES, encoding='utf-8-sig').set_index('tile')
    forest_tiles = [t for t in tl.index
                    if t in by.index and (by.loc[t, 'needle_px'] + by.loc[t, 'broad_px']) > 0]
    log('码表 %d 行；瓦 %d；有针/阔像元的瓦 %d' % (len(meta), len(tl), len(forest_tiles)))
    if args.tiles:
        want = set(s for s in args.tiles.split(',') if s.strip())
        forest_tiles = [t for t in forest_tiles if t in want]
    if args.limit:
        forest_tiles = forest_tiles[:args.limit]

    if not args.assemble_only:
        ds = rasterio.open(VEG30)
        T = ds.transform
        log('栅格 %dx%d crs=%s nodata=%s' % (ds.width, ds.height, ds.crs, ds.nodata))
        for i, t in enumerate(forest_tiles, 1):
            pq = os.path.join(WORK_TILES, '%s.parquet' % t)
            js = os.path.join(WORK_TILES, '%s.json' % t)
            if not args.force and os.path.exists(pq) and os.path.exists(js):
                continue
            r = tl.loc[t]
            t0 = time.time()
            seed = 20261024 + int(t[1:])
            df, cnt = process_tile(t, float(r.x0), float(r.y0), float(r.x1), float(r.y1), ds, T, LUT, seed)
            cnt['secs'] = round(time.time() - t0, 2)
            df.to_parquet(pq, index=False)
            json.dump(cnt, open(js, 'w', encoding='utf-8'), ensure_ascii=False, default=jsonable)
            log('%3d/%3d %s  L1 %7d/%7d  L2 %7d/%7d  L3comp %5d/%4d  pts %3d/%3d  %.1fs' % (
                i, len(forest_tiles), t,
                cnt.get('needle_L1_px', 0), cnt.get('broad_L1_px', 0),
                cnt.get('needle_L2_px', 0), cnt.get('broad_L2_px', 0),
                cnt.get('needle_L3_ncomp', 0), cnt.get('broad_L3_ncomp', 0),
                cnt.get('needle_L4_pts', 0), cnt.get('broad_L4_pts', 0), cnt['secs']))
        ds.close()

    # ---------------- 组装
    log('组装 ...')
    pqs = sorted(glob.glob(os.path.join(WORK_TILES, '*.parquet')))
    pts = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    pts = pts.drop_duplicates(subset=['lon', 'lat', 'leaf']).reset_index(drop=True)
    n_sampled = len(pts)
    log('撒点合计 %d（needle %d / broad %d）' % (
        n_sampled, int((pts.leaf == 'needle').sum()), int((pts.leaf == 'broad').sum())))

    ds = rasterio.open(VEG30)
    # 抽样自证：随机 2000 点用 rasterio.sample 复核 code30（注意 rasterio.sample 吃数据集 CRS 坐标=3857）
    chk = np.random.default_rng(11).choice(len(pts), size=min(2000, len(pts)), replace=False)
    sub = pts.iloc[chk]
    tr_chk = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    Xc, Yc = tr_chk.transform(sub.lon.to_numpy(), sub.lat.to_numpy())
    chk_raw = np.fromiter((v[0] for v in ds.sample(list(zip(Xc, Yc)), indexes=1)),
                          dtype='int64', count=len(sub))
    ds.close()
    agree = float((chk_raw == sub.code30.to_numpy()).mean() * 100)
    log('code30 抽样复核（n=%d，rasterio.sample vs 阶段内回取）：一致率 %.3f%%' % (len(sub), agree))
    raw = pts.code30.to_numpy()
    pts['ctype'] = [meta.get(int(v), ('', '', '', 0))[0] for v in raw]
    pts['vtype'] = [meta.get(int(v), ('', '', '', 0))[1] for v in raw]
    pts['gen'] = [meta.get(int(v), ('', '', '', 0))[2] for v in raw]
    pts['ambig_level'] = [meta.get(int(v), ('', '', '', 0))[3] for v in raw]
    pts['code_cls'] = [mapcls.get(int(v), 'nodata') for v in raw]
    bad = int((pts.code_cls != pts.leaf).sum())
    log('code30 映射类与 leaf 不一致 %d（应为 0）' % bad)

    # L5 v2 去重：0.005° 盒（|dlon|<=0.005 且 |dlat|<=0.005）内有 v2 点则弃
    v2 = pd.read_parquet(V2PTS, columns=['lon', 'lat', 'leaf'])
    d2, linf = nearest_v2(pts.lon.to_numpy(), pts.lat.to_numpy(), v2.lon.to_numpy(), v2.lat.to_numpy())
    pts['dist_v2_m'] = d2
    n_before_l5 = len(pts)
    pts5 = pts[linf > V2_DEDUP_DEG].reset_index(drop=True)
    n_after_v2 = len(pts5)
    log('L5 v2 去重（%.3f° 盒，真距中位 %.0f m）：%d → %d（弃 %d）' % (
        V2_DEDUP_DEG, float(np.median(d2[linf > V2_DEDUP_DEG])) if n_after_v2 else 0,
        n_before_l5, n_after_v2, n_before_l5 - n_after_v2))

    # L4 全局 0.01° 同类 1 点（随机优先）
    rng = np.random.default_rng(7)
    tmp = pts5.assign(_cell=cellkey(pts5), _pri=rng.random(len(pts5)))
    tmp = tmp.sort_values('_pri').drop_duplicates(subset=['leaf', '_cell'], keep='first')
    pts6 = tmp.drop(columns=['_cell', '_pri']).sort_values(['tile', 'leaf', 'lon']).reset_index(drop=True)
    n_after_grid = len(pts6)
    log('L4 全局 0.01° 格去重：%d → %d' % (n_after_v2, n_after_grid))

    # L4b 跨瓦接缝兜底：0.01° 格去重对"跨越格边界的近邻对"无效（两瓦各出一对，
    # 分属相邻格）→ 再做一次同类最小间距 0.01° 的全局贪心（只删，不增）
    pts6 = seam_thin(pts6, 0.01)
    n_after_seam = len(pts6)
    log('L4b 跨瓦接缝薄化（0.01° 同类间距）：%d → %d（弃 %d）' % (
        n_after_grid, n_after_seam, n_after_grid - n_after_seam))

    # L6 境内
    shapely, tree, names = build_boundary()
    pairs = tree.query(shapely.points(pts6.lon.to_numpy(), pts6.lat.to_numpy()), predicate='intersects')
    inside = np.zeros(len(pts6), bool)
    prov = np.array([''] * len(pts6), dtype=object)
    inside[pairs[0]] = True
    prov[pairs[0]] = names[pairs[1]]
    n_multi = int((np.bincount(pairs[0], minlength=len(pts6)) > 1).sum())
    n_before_l6 = len(pts6)
    pts7 = pts6[inside].reset_index(drop=True)
    pts7['prov'] = prov[inside]
    pts7['in_china'] = True
    pts7['point_id'] = np.arange(1, len(pts7) + 1, dtype='int64')
    log('L6 境内：%d → %d（境外弃 %d；多省重叠 %d）' % (n_before_l6, len(pts7), n_before_l6 - len(pts7), n_multi))

    # 诊断：5x5 邻域纯度（不筛选，只记录）
    log('计算 5x5 邻域纯度诊断（%d 点，逐点小窗）...' % len(pts7))
    pts7 = add_purity(pts7, LUT, 2)

    pts7 = pts7[['point_id', 'lon', 'lat', 'tile', 'leaf', 'code30', 'ctype', 'vtype', 'gen',
                 'prov', 'in_china', 'dist_v2_m', 'ambig_level', 'pur_leaf', 'pur_gen']]
    pts7.to_parquet(OUT_PQ, index=False)
    log('写入 %s（%d 行）' % (OUT_PQ, len(pts7)))

    stats = build_stats(pts7, {'sampled': n_sampled, 'after_v2': n_after_v2,
                               'after_grid': n_after_grid, 'after_seam': n_after_seam,
                               'before_l6': n_before_l6,
                               'final': len(pts7), 'code30_agree_pct': agree,
                               'n_code30_mismatch_leaf': bad}, by, tl, forest_tiles, v2, n_multi,
                        time.time() - t_start)
    json.dump(stats, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=jsonable)
    log('写入 %s' % OUT_JSON)
    log('DONE %.1f s' % (time.time() - t_start))


def build_stats(pts, cnts, by, tl, forest_tiles, v2, n_multi, secs):
    fl = pd.DataFrame([json.load(open(p, encoding='utf-8'))
                       for p in sorted(glob.glob(os.path.join(WORK_TILES, '*.json')))])
    done = set(fl.tile) if len(fl) else set()
    miss = []
    for t in tl.index:
        if t in done:
            continue
        r = tl.loc[t]
        b = by.loc[t] if t in by.index else None
        row = {'tile': t, 'x0': r.x0, 'y0': r.y0, 'x1': r.x1, 'y1': r.y1, 'secs': 0.0,
               'win_px': int(b.width_px * b.height_px) if b is not None else 0,
               'valid_px': int(b.valid_px) if b is not None else 0}
        for c in ('needle', 'broad'):
            row['%s_L1_px' % c] = int(b['%s_px' % c]) if b is not None else 0
            for k in ('L2_px', 'L3_px', 'L3_ncomp', 'L4_blocks', 'L4_pts'):
                row['%s_%s' % (c, k)] = 0
        miss.append(row)
    flall = pd.concat([fl, pd.DataFrame(miss)], ignore_index=True) if len(fl) else pd.DataFrame(miss)
    for c in flall.columns:
        if c not in ('tile', 'x0', 'y0', 'x1', 'y1', 'secs', 'note'):
            flall[c] = pd.to_numeric(flall[c], errors='coerce').fillna(0).astype('int64')

    S = lambda k: int(flall[k].sum()) if k in flall else 0
    funnel = {
        'n_tiles_total': int(len(tl)),
        'n_tiles_scanned': int(len(flall)),
        'n_tiles_with_forest': int(len(forest_tiles)),
        'n_tiles_scanned_with_pts': int((flall['needle_L4_pts'] + flall['broad_L4_pts'] > 0).sum()),
        'L0_window_px': S('win_px'),
        'L0_valid_px': S('valid_px'),
        'L1_needle_px': S('needle_L1_px'),
        'L1_broad_px': S('broad_L1_px'),
        'L1_forest_px': S('needle_L1_px') + S('broad_L1_px'),
        'L2_needle_px': S('needle_L2_px'),
        'L2_broad_px': S('broad_L2_px'),
        'L2_forest_px': S('needle_L2_px') + S('broad_L2_px'),
        'L3_needle_px': S('needle_L3_px'),
        'L3_broad_px': S('broad_L3_px'),
        'L3_needle_ncomp': S('needle_L3_ncomp'),
        'L3_broad_ncomp': S('broad_L3_ncomp'),
        'L4_needle_blocks': S('needle_L4_blocks'),
        'L4_broad_blocks': S('broad_L4_blocks'),
        'L4_needle_pts': S('needle_L4_pts'),
        'L4_broad_pts': S('broad_L4_pts'),
        'L4_pts_total': S('needle_L4_pts') + S('broad_L4_pts'),
        'dedup_exact_or_seam': int(S('needle_L4_pts') + S('broad_L4_pts') - cnts['sampled']),
        'after_v2_dedup': cnts['after_v2'],
        'after_grid_dedup': cnts['after_grid'],
        'after_seam_thin': cnts['after_seam'],
        'before_china_filter': cnts['before_l6'],
        'final_in_china': cnts['final'],
        'code30_selfcheck_sample_agree_pct': round(cnts['code30_agree_pct'], 3),
        'code30_vs_leaf_mismatch': cnts['n_code30_mismatch_leaf'],
    }
    L1 = max(1, funnel['L1_forest_px'])
    funnel['retention_pct'] = {
        'L2_over_L1_px': round(100.0 * funnel['L2_forest_px'] / L1, 2),
        'L3_over_L1_px': round(100.0 * (funnel['L3_needle_px'] + funnel['L3_broad_px']) / L1, 2),
        'L4pts_over_L4blocks_pct': round(100.0 * funnel['L4_pts_total']
                                         / max(1, funnel['L4_needle_blocks'] + funnel['L4_broad_blocks']), 2),
        'final_over_sampled_pct': round(100.0 * cnts['final'] / max(1, cnts['sampled']), 2),
        'note': ('L1/L2/L3 为像元级漏斗；L4 撒点数与像元不同量纲，'
                 'L4pts/L4blocks 是"可用 3x3 块→实际保留点"的收敛率（受 500/瓦/类上限与 0.015° 间距约束）'),
    }

    out = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'secs': round(secs, 1),
        'params': {'erode': 'binary_erosion 5x5', 'halo_px': HALO, 'min_comp_px': MIN_COMP_PX,
                   'block_px': BLOCK, 'cap_per_class_tile': CAP_PER_CLASS_TILE,
                   'minsep_deg': MINSEP_DEG, 'grid_dedup_deg': GRID_DEDUP, 'v2_dedup_deg': V2_DEDUP_DEG,
                   'raster': VEG30, 'crs': 'EPSG:3857', 'nodata': NODATA},
        'funnel': funnel,
        'by_leaf': {k: int(v) for k, v in pts.leaf.value_counts().items()},
        'by_leaf_final_pct': {k: round(100.0 * v / max(1, len(pts)), 2) for k, v in pts.leaf.value_counts().items()},
        'n_tiles_covered': int(pts.tile.nunique()),
        'n_prov_covered': int(pts.prov.nunique()),
        'by_prov': {k: int(v) for k, v in pts.prov.value_counts().items()},
        'by_prov_leaf': {f'{a}|{b}': int(v) for (a, b), v in pts.groupby(['prov', 'leaf']).size().items()},
        'by_tile': {k: int(v) for k, v in pts.tile.value_counts().items()},
        'gen_top20': {k: int(v) for k, v in pts.gen.replace('', '(空)').value_counts().head(20).items()},
        'gen_top20_share_pct': round(100.0 * pts.gen.replace('', '(空)').value_counts().head(20).sum()
                                     / max(1, len(pts)), 2),
        'n_gen_distinct': int(pts.gen.nunique()),
        'gen_top10_needle': {k: int(v) for k, v in
                             pts[pts.leaf == 'needle'].gen.replace('', '(空)').value_counts().head(10).items()},
        'gen_top10_broad': {k: int(v) for k, v in
                            pts[pts.leaf == 'broad'].gen.replace('', '(空)').value_counts().head(10).items()},
        'ambig_level_counts': {str(k): int(v) for k, v in pts.ambig_level.value_counts().sort_index().items()},
        'ambig_level_pct': {str(k): round(100.0 * v / max(1, len(pts)), 2)
                            for k, v in pts.ambig_level.value_counts().sort_index().items()},
        'pur_leaf': {
            'mean': round(float(pts.pur_leaf.mean()), 4),
            'pct_eq_1': round(100.0 * float((pts.pur_leaf >= 0.999).mean()), 2),
            'pct_lt_0.8': round(100.0 * float((pts.pur_leaf < 0.8).mean()), 2),
            'note': '5x5 邻域内同叶型像元占比；L2 类级腐蚀应保证 =1',
        },
        'pur_gen': {
            'mean': round(float(pts.pur_gen.mean()), 4),
            'pct_eq_1': round(100.0 * float((pts.pur_gen >= 0.999).mean()), 2),
            'pct_ge_0.9': round(100.0 * float((pts.pur_gen >= 0.9).mean()), 2),
            'pct_lt_0.8': round(100.0 * float((pts.pur_gen < 0.8).mean()), 2),
            'note': ('5x5 邻域内同 code30（同群系）像元占比。类级腐蚀不保证群系级纯；'
                     '<1 的点位于同一叶型内不同群系交界，按群系分层核查时应留意这一小部分'),
        },
        'dist_v2_m': {k: round(float(v), 1) for k, v in {
            'min': pts.dist_v2_m.min(), 'p05': pts.dist_v2_m.quantile(.05),
            'p25': pts.dist_v2_m.quantile(.25), 'median': pts.dist_v2_m.median(),
            'mean': pts.dist_v2_m.mean(), 'p75': pts.dist_v2_m.quantile(.75),
            'p95': pts.dist_v2_m.quantile(.95), 'max': pts.dist_v2_m.max()}.items()},
        'n_multi_prov_points': int(n_multi),
    }
    c_new, c_v2 = cellkey(pts), cellkey(v2)
    ov = {}
    ov['any_leaf'] = round(100.0 * np.isin(c_new, np.unique(c_v2)).sum() / max(1, len(pts)), 2)
    for leaf in ('needle', 'broad'):
        m = (pts.leaf == leaf).to_numpy()
        s = np.unique(c_v2[(v2.leaf == leaf).to_numpy()])
        ov[leaf + '_same_leaf'] = round(100.0 * np.isin(c_new[m], s).sum() / max(1, m.sum()), 2)
    out['v2_grid_0p01_overlap_pct'] = ov
    # 上限饱和诊断：多少瓦×类触到 500 上限（触顶说明该瓦实际可采块远超配额，
    # 采样强度由配额而非空间可用性决定 → 空间均匀性依赖随机候选，非穷尽）
    per = pts.groupby(['tile', 'leaf']).size()
    out['cap_diagnostics'] = {
        'n_tile_class_groups': int(len(per)),
        'n_at_or_above_495': int((per >= 495).sum()),
        'n_at_cap_exact': int((per >= CAP_PER_CLASS_TILE).sum()),
        'max_per_group': int(per.max()),
        'median_per_group': int(per.median()),
        'note': ('L4 每瓦每类 500 上限：%d/%d 个瓦×类组触顶（≥495 点）。'
                 '未触顶的组由可用 3x3 块数决定，触顶组由配额决定——'
                 '因此点数不完全反映森林面积分布，做面积加权统计时须自行重加权'
                 % (int((per >= 495).sum()), len(per))),
    }
    return out


if __name__ == '__main__':
    main()
