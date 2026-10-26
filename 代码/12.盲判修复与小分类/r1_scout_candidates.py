# -*- coding: utf-8 -*-
"""r1_scout_candidates.py — 候选点踏查：林型坡向配对 + 园地/森林/耕地争议 + R/V 角色

* 输入（全部只读）：v2 训练池 train2023_v2.parquet、两版 10 m 瓦片、本地 SRTM 60 m DEM、
        母库 r7_train.parquet（仅用于 V 的空间隔离距离守卫）、tiles_secondrun.csv
* 规则源：技术文档/72 协议（主 agent 冻结）；doc69 §5.1 坡向口径；doc70 §6.1 空间块
* 角色：
    R 点一律取自 **v2 训练池成员**（可进 candidate 训练、绝不进评价）；
    V 点一律**不取自训练池**：由 V 块内随机位置 / 两版分歧位置产生，且与母库原点、
    v2 池点均保持 ≥ V_MIN_DIST_KM，V 所在 0.25° 块整块从训练排除。
* 门槛：林型对要求 同 0.25° 块、坡度>5°、一阳一阴、高程差 ≤200 m、距离 ≤4 km；
        角色按块哈希确定，块内 role 唯一（R 块与 V 块不交）；块内取点有上限与最小间距。
* 输出：private/candidates_private.parquet（含先验/role/配对——判读代理不得读）
        private/role_blocks.json、private/scout_summary.json、logs/r1_scout.log
* 用法：python r1_scout_candidates.py [--target-pairs 50] [--dry]
"""
import argparse
import collections
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

V_MIN_DIST_KM = 1.0        # V 点与母库/v2 池点的最小距离
PAIR_MIN_SEP_KM = 0.5      # 同块内两对之间的最小间距
DISPUTE_MIN_SEP_KM = 0.3   # 同块内争议点最小间距
MAX_PAIRS_PER_BLOCK = 2
MAX_DISPUTE_PER_BLOCK = 4
V_RANDOM_PER_BLOCK = 400   # 每个 V 块内随机试采点数
ROLE_V_FRACTION = 0.40     # 块哈希 < 该值 → V 块（先验比例，最终以配额为准）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


# ---------------------------------------------------------------- 瓦片 / 地形
def sichuan_tiles():
    df = pd.read_csv(TILE_PLAN2)
    df = df[df['prov'] == '川'].copy()
    df['tile'] = df['tile'].astype(str)
    return df[['tile', 'x0', 'y0', 'x1', 'y1']].reset_index(drop=True)


def tile_of(lon, lat, tiles):
    """点 → 瓦片（四川计划内）；不在任何瓦片返回 None。"""
    for r in tiles.itertuples(index=False):
        if r.x0 <= lon < r.x1 and r.y0 <= lat < r.y1:
            return r.tile
    return None


def tile_of_vec(lon, lat, tiles):
    """向量化 tile_of（池级用，避免百万级 Python 循环）。"""
    out = np.full(len(lon), None, dtype=object)
    for r in tiles.itertuples(index=False):
        m = (lon >= r.x0) & (lon < r.x1) & (lat >= r.y0) & (lat < r.y1)
        out[m] = r.tile
    return out


def box_of_vec(lon, lat):
    out = np.full(len(lon), None, dtype=object)
    for k, v in BOXES.items():
        x0, y0, x1, y1 = v['bbox']
        m = (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
        out[m] = k
    return out


def in_seen_vec(lon, lat):
    out = np.zeros(len(lon), dtype=bool)
    for x0, y0, x1, y1 in SEEN_BOXES_2026_10_15:
        out |= (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
    return out


class DemCache:
    """SRTM 60 m：按瓦片惰性计算 slope/aspect（Horn），最多缓存 --dem-cache 个瓦片。"""

    def __init__(self, max_tiles=4):
        import rasterio
        self.rasterio = rasterio
        self.max_tiles = max_tiles
        self.cache = collections.OrderedDict()

    def _load(self, tile):
        if tile in self.cache:
            self.cache.move_to_end(tile)
            return self.cache[tile]
        fp = os.path.join(DEM, 'dem_%s.tif' % tile)
        if not os.path.exists(fp):
            self.cache[tile] = None
            return None
        with self.rasterio.open(fp) as s:
            a = s.read(1).astype('float32')
            tr = s.transform
            nod = s.nodata
            b = s.bounds
        a[a == nod] = np.nan
        # Horn 3×3（度/像元 → 米：纬度方向固定，经度方向乘 cos(lat)）
        px = abs(tr.a)                                    # 度/像元（x）
        py = abs(tr.e)
        mid = (b.top + b.bottom) / 2.0
        mx = px * 111320.0 * math.cos(math.radians(mid))
        my = py * 111320.0
        p = np.pad(a, 1, mode='edge')
        dzdx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) -
                (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8.0 * mx)
        dzdy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) -
                (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / (8.0 * my)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        # ESRI/Horn 坡向：0=北 90=东 180=南 270=西
        asp0 = np.degrees(np.arctan2(dzdy, -dzdx))
        aspect = np.where(asp0 < 0.0, 90.0 - asp0,
                          np.where(asp0 > 90.0, 450.0 - asp0, 90.0 - asp0)) % 360.0
        rec = dict(arr=a, slope=slope, aspect=aspect, transform=tr, bounds=b, w=a.shape[1])
        self.cache[tile] = rec
        while len(self.cache) > self.max_tiles:
            self.cache.popitem(last=False)
        return rec

    def at(self, tile, lon, lat):
        rec = self._load(tile)
        if rec is None:
            return np.nan, np.nan, np.nan
        tr = rec['transform']
        col = int((lon - tr.c) / tr.a)
        row = int((lat - tr.f) / tr.e)
        if not (0 <= row < rec['slope'].shape[0] and 0 <= col < rec['w']):
            return np.nan, np.nan, np.nan
        return (float(rec['slope'][row, col]), float(rec['aspect'][row, col]), float(rec['arr'][row, col]))


class RastCache:
    """两版 10 m 类别栅格：按瓦片惰性读取，多点向量化取样。"""

    def __init__(self):
        import rasterio
        self.rasterio = rasterio
        self.cache = {}

    def _open(self, root, tile):
        key = (root, tile)
        if key not in self.cache:
            fp = os.path.join(root, '%s_10m.tif' % tile)
            self.cache[key] = self.rasterio.open(fp) if os.path.exists(fp) else None
        return self.cache[key]

    def sample(self, root, tile, lons, lats):
        ds = self._open(root, tile)
        if ds is None:
            return np.zeros(len(lons), dtype=np.int16)
        out = np.zeros(len(lons), dtype=np.int16)
        for i, (lo, la) in enumerate(zip(lons, lats)):
            try:
                v = next(ds.sample([(lo, la)]))[0]
            except Exception:
                v = 0
            out[i] = int(v)
        return out


def aspect_class(a):
    if not (a == a):
        return None
    if ASPECT_SUN[0] <= a <= ASPECT_SUN[1]:
        return 'sun'
    for lo, hi in ASPECT_SHADE:
        if lo <= a <= hi:
            return 'shade'
    return None


# ---------------------------------------------------------------- 角色（块）
def role_of_block(bid, salt=0):
    h = int(stable_uid('blk%d' % salt, bid).split('_')[1][:8], 16) / float(0xFFFFFFFF)
    return ROLE_V if h < ROLE_V_FRACTION else ROLE_R


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='候选点踏查（林型配对 + 争议点 + R/V 角色）')
    ap.add_argument('--target-pairs', type=int, default=PAIRS_TOTAL)
    ap.add_argument('--dry', action='store_true', help='只打印计划，不写产物')
    ap.add_argument('--dem-cache', type=int, default=4)
    a = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    tiles = sichuan_tiles()
    box_tiles = {k: v['tile'] for k, v in BOXES.items()}
    tiles = tiles[tiles['tile'].isin(list(box_tiles.values()))].reset_index(drop=True)
    emit('局部训练框 %d 个：%s' % (len(BOXES), ' / '.join('%s=%s %s' % (k, v['tile'], v['bbox']) for k, v in BOXES.items())))
    dem = DemCache(a.dem_cache)
    rast = RastCache()

    # ---- 1) v2 池（R 来源）与母库（V 隔离守卫） ----
    pool = pd.read_parquet(POOL_V2, columns=['point_id', 'lon', 'lat', 'cl', 'src'])
    # V 的距离守卫用**全池**（不止本框）：训练窗为 tile±2°，相邻瓦片池点同样会进训练
    full_xy = np.c_[pool['lon'].values * 111.32 * np.cos(np.radians(pool['lat'].values)),
                    pool['lat'].values * 110.57]
    plo, pla = pool['lon'].values, pool['lat'].values
    pool['tile'] = tile_of_vec(plo, pla, tiles)
    pool = pool[pool['tile'].notna()].reset_index(drop=True)
    pool['box'] = box_of_vec(pool['lon'].values, pool['lat'].values)
    pool['seen'] = in_seen_vec(pool['lon'].values, pool['lat'].values)
    emit('v2 池两框内成员 %d（F=%d / O=%d）' % (
        len(pool), (pool['box'] == 'F').sum(), (pool['box'] == 'O').sum()))

    from scipy.spatial import cKDTree
    pool_tree = cKDTree(full_xy)

    mother_xy = None
    if os.path.exists(MOTHER):
        m = pd.read_parquet(MOTHER, columns=['lon', 'lat'])
        mother_xy = np.c_[m['lon'].values * 111.32 * np.cos(np.radians(m['lat'].values)),
                          m['lat'].values * 110.57]
        mother_tree = cKDTree(mother_xy)
        emit('母库 %d 点（V 距离守卫）' % len(m))
    else:
        mother_tree = None
        emit('警告：母库缺失，V 仅对 v2 池做距离守卫')

    def far_enough(lons, lats, km=V_MIN_DIST_KM):
        xy = np.c_[np.asarray(lons) * 111.32 * np.cos(np.radians(np.asarray(lats))),
                   np.asarray(lats) * 110.57]
        d1 = pool_tree.query(xy, k=1)[0]
        ok = d1 >= km
        if mother_tree is not None:
            d2 = mother_tree.query(xy, k=1)[0]
            ok &= (d2 >= km)
        return ok

    # ---- 2) 池内林型点（F 框＝川东北山地）：地形 + 两版类别 ----
    fcl = pool[(pool['cl'].isin(FOREST_CL)) & (pool['box'] == 'F')].copy()
    emit('池内林型点（F 框 %s）%d' % (BOXES['F']['tile'], len(fcl)))
    fcl = fcl.sample(n=min(len(fcl), 60000), random_state=SEED).reset_index(drop=True)
    terr = [dem.at(t, lo, la) for t, lo, la in zip(fcl['tile'], fcl['lon'], fcl['lat'])]
    fcl['slope'], fcl['aspect'], fcl['elev'] = [x[0] for x in terr], [x[1] for x in terr], [x[2] for x in terr]
    fcl['asp'] = [aspect_class(x) for x in fcl['aspect']]
    fcl['block'] = [block_id(lo, la, PAIR_BLOCK_DEG) for lo, la in zip(fcl['lon'], fcl['lat'])]

    c1 = np.zeros(len(fcl), dtype=np.int16)
    c2 = np.zeros(len(fcl), dtype=np.int16)
    for tl, idx in fcl.groupby('tile').groups.items():
        sub = fcl.loc[idx]
        c1[idx] = rast.sample(RAST1, tl, sub['lon'].values, sub['lat'].values)
        c2[idx] = rast.sample(RAST2, tl, sub['lon'].values, sub['lat'].values)
    fcl['c_v1'], fcl['c_v2'] = c1, c2
    emit('池内林型点地形/类别就绪（%.0fs）' % (time.time() - t0))

    # ---- 3) R 林型对：同块内一阳一阴，高程/距离匹配 ----
    cand_pairs = []
    for bid, g in fcl.groupby('block'):
        g = g[(g['slope'] > MIN_SLOPE_DEG) & (g['asp'].notna()) & (g['c_v2'] > 0)]
        sun = g[g['asp'] == 'sun']
        shade = g[g['asp'] == 'shade']
        if len(sun) < 1 or len(shade) < 1:
            continue
        sxy = np.c_[sun['lon'].values * 111.32 * np.cos(np.radians(sun['lat'].values)), sun['lat'].values * 110.57]
        hxy = np.c_[shade['lon'].values * 111.32 * np.cos(np.radians(shade['lat'].values)), shade['lat'].values * 110.57]
        st = cKDTree(sxy)
        d, j = st.query(hxy, k=min(6, len(sun)))
        d = np.atleast_2d(d)
        j = np.atleast_2d(j)
        taken = []
        for k in range(len(shade)):
            for kk in range(d.shape[1]):
                if d[k, kk] > PAIR_MAX_KM:
                    break
                si = j[k, kk]
                srow = sun.iloc[si]
                hrow = shade.iloc[k]
                if abs(srow['elev'] - hrow['elev']) > PAIR_ELEV_TOL_M:
                    continue
                if any(d[k, kk] == td for td in []):
                    continue
                if any(srow['point_id'] == p['sun_id'] or hrow['point_id'] == p['shade_id'] for p in taken):
                    continue
                cand_pairs.append(dict(
                    block=bid, group='forest_pair',
                    sun_id=str(srow['point_id']), shade_id=str(hrow['point_id']),
                    sun_lon=float(srow['lon']), sun_lat=float(srow['lat']),
                    shade_lon=float(hrow['lon']), shade_lat=float(hrow['lat']),
                    sun_cl=int(srow['cl']), shade_cl=int(hrow['cl']),
                    sun_src=str(srow['src']), shade_src=str(hrow['src']),
                    sun_aspect=float(srow['aspect']), shade_aspect=float(hrow['aspect']),
                    elev_sun=float(srow['elev']), elev_shade=float(hrow['elev']),
                    slope_sun=float(srow['slope']), slope_shade=float(hrow['slope']),
                    dist_km=float(d[k, kk]),
                ))
                taken.append(dict(sun_id=str(srow['point_id']), shade_id=str(hrow['point_id'])))
                break
            if len(taken) >= MAX_PAIRS_PER_BLOCK:
                break
    emit('候选林型对 %d（池内，同块一阳一阴）' % len(cand_pairs))

    # ---- 4) R 争议点（O 框＝川东）：池内 园地/森林/耕地，分歧区 + 稳定区 ----
    dcl = pool[(pool['cl'].isin(DISPUTE_CL)) & (pool['box'] == 'O')].copy()
    emit('池内争议类点（O 框 %s）%d' % (BOXES['O']['tile'], len(dcl)))
    dcl = dcl.sample(n=min(len(dcl), 60000), random_state=SEED + 1).reset_index(drop=True)
    d1 = np.zeros(len(dcl), dtype=np.int16)
    d2 = np.zeros(len(dcl), dtype=np.int16)
    for tl, idx in dcl.groupby('tile').groups.items():
        sub = dcl.loc[idx]
        d1[idx] = rast.sample(RAST1, tl, sub['lon'].values, sub['lat'].values)
        d2[idx] = rast.sample(RAST2, tl, sub['lon'].values, sub['lat'].values)
    dcl['c_v1'], dcl['c_v2'] = d1, d2
    dcl['block'] = [block_id(lo, la, BLOCK_DEG) for lo, la in zip(dcl['lon'], dcl['lat'])]
    dcl['change'] = np.where((dcl['c_v1'] > 0) & (dcl['c_v2'] > 0) & (dcl['c_v1'] != dcl['c_v2']), 'disagree',
                             np.where((dcl['c_v1'] > 0) & (dcl['c_v1'] == dcl['c_v2']), 'stable', 'nodata'))
    emit('池内争议类点 %d；分歧 %d / 稳定 %d' % (len(dcl), (dcl['change'] == 'disagree').sum(),
                                          (dcl['change'] == 'stable').sum()))

    # ---- 5) V 块随机点（林型 + 争议类）：不与池/母库近邻、避开既有已看块 ----
    all_blocks = sorted(set(fcl['block']) | set(dcl['block']))
    blk_box = {}
    for b in all_blocks:
        _, ix, iy = b.split('_')
        clon, clat = (int(ix) + 0.5) * BLOCK_DEG, (int(iy) + 0.5) * BLOCK_DEG
        blk_box[b] = box_of(clon, clat)
        if blk_box[b] is None:
            blk_box[b] = 'F' if role_of_block(b) == ROLE_R else 'O'
    v_blocks = [b for b in all_blocks if role_of_block(b) == ROLE_V and not in_seen(*((int(b.split('_')[1]) + 0.5) * BLOCK_DEG,
                                                                                    (int(b.split('_')[2]) + 0.5) * BLOCK_DEG))]
    emit('候选块 %d（F=%d / O=%d）；其中哈希先验 V 块 %d（已扣既有已看块）' % (
        len(all_blocks), sum(1 for v in blk_box.values() if v == 'F'),
        sum(1 for v in blk_box.values() if v == 'O'), len(v_blocks)))

    def block_bbox(bid, deg):
        _, ix, iy = bid.split('_')
        x0, y0 = int(ix) * deg, int(iy) * deg
        return x0, y0, x0 + deg, y0 + deg

    v_pts = []
    for bid in v_blocks:
        x0, y0, x1, y1 = block_bbox(bid, BLOCK_DEG)
        lo = rng.uniform(x0, x1, V_RANDOM_PER_BLOCK)
        la = rng.uniform(y0, y1, V_RANDOM_PER_BLOCK)
        tl = tile_of_vec(lo, la, tiles)
        ok = np.array([t is not None for t in tl])
        lo, la, tl = lo[ok], la[ok], tl[ok]
        if len(lo) == 0:
            continue
        ok = far_enough(lo, la)
        lo, la, tl = lo[ok], la[ok], tl[ok]
        if len(lo) == 0:
            continue
        cv2 = np.zeros(len(lo), dtype=np.int16)
        cv1 = np.zeros(len(lo), dtype=np.int16)
        for t in np.unique(tl):
            m = tl == t
            cv2[m] = rast.sample(RAST2, t, lo[m], la[m])
            cv1[m] = rast.sample(RAST1, t, lo[m], la[m])
        for i in range(len(lo)):
            c2v, c1v = int(cv2[i]), int(cv1[i])
            if c2v <= 0 and c1v <= 0:
                continue
            grp = 'forest_v' if (c2v in FOREST_CL or c1v in FOREST_CL) else (
                'dispute_v' if (c2v in DISPUTE_CL or c1v in DISPUTE_CL) else None)
            if grp is None:
                continue
            if grp == 'forest_v' and blk_box.get(bid) != 'F':
                continue
            if grp == 'dispute_v' and blk_box.get(bid) != 'O':
                continue
            sl, aspd, el = dem.at(tl[i], lo[i], la[i])
            v_pts.append(dict(block=bid, group=grp, lon=float(lo[i]), lat=float(la[i]),
                              c_v1=c1v, c_v2=c2v, slope=sl, aspect=aspd, elev=el,
                              asp=aspect_class(aspd),
                              change='disagree' if (c1v > 0 and c2v > 0 and c1v != c2v) else (
                                  'stable' if (c1v > 0 and c1v == c2v) else 'nodata'),
                              tile=tl[i]))
    vdf = pd.DataFrame(v_pts)
    emit('V 块独立随机候选点 %d（林型 %d / 争议 %d）' % (
        len(vdf), (vdf['group'] == 'forest_v').sum() if len(vdf) else 0,
        (vdf['group'] == 'dispute_v').sum() if len(vdf) else 0))

    # V 林型对：同块一阳一阴（整对同 role＝V）
    v_pairs = []
    if len(vdf):
        for bid, g in vdf[vdf['group'] == 'forest_v'].groupby('block'):
            g = g[(g['slope'] > MIN_SLOPE_DEG) & (g['asp'].notna())]
            sun = g[g['asp'] == 'sun']
            shade = g[g['asp'] == 'shade']
            if len(sun) < 1 or len(shade) < 1:
                continue
            used = []
            for _, hrow in shade.iterrows():
                cand = sun[(sun['elev'].sub(hrow['elev']).abs() <= PAIR_ELEV_TOL_M)]
                if len(cand) == 0:
                    continue
                dkm = np.hypot((cand['lon'] - hrow['lon']) * 111.32 * math.cos(math.radians(hrow['lat'])),
                               (cand['lat'] - hrow['lat']) * 110.57)
                cand = cand[dkm <= PAIR_MAX_KM]
                if len(cand) == 0:
                    continue
                srow = cand.iloc[int(np.argmin(dkm[dkm <= PAIR_MAX_KM].values))]
                if any(abs(srow['lon'] - u['lon']) < 1e-6 and abs(srow['lat'] - u['lat']) < 1e-6 for u in used):
                    continue
                v_pairs.append(dict(block=bid, group='forest_pair',
                                    sun_lon=float(srow['lon']), sun_lat=float(srow['lat']),
                                    shade_lon=float(hrow['lon']), shade_lat=float(hrow['lat']),
                                    sun_cl=int(srow['c_v2'] or srow['c_v1']), shade_cl=int(hrow['c_v2'] or hrow['c_v1']),
                                    sun_aspect=float(srow['aspect']), shade_aspect=float(hrow['aspect']),
                                    elev_sun=float(srow['elev']), elev_shade=float(hrow['elev']),
                                    slope_sun=float(srow['slope']), slope_shade=float(hrow['slope']),
                                    sun_src='v_block_random', shade_src='v_block_random',
                                    sun_c_v1=int(srow['c_v1']), sun_c_v2=int(srow['c_v2']),
                                    shade_c_v1=int(hrow['c_v1']), shade_c_v2=int(hrow['c_v2']),
                                    dist_km=float(math.hypot((srow['lon'] - hrow['lon']) * 111.32 * math.cos(math.radians(hrow['lat'])),
                                                             (srow['lat'] - hrow['lat']) * 110.57))))
                used.append(dict(lon=float(srow['lon']), lat=float(srow['lat'])))
                if len(v_pairs) >= MAX_PAIRS_PER_BLOCK * 6:
                    break
            if sum(1 for p in v_pairs if p['block'] == bid) >= MAX_PAIRS_PER_BLOCK:
                continue
    emit('V 块林型候选对 %d' % len(v_pairs))

    # ---- 6) 角色与配额装配 ----
    role_blocks = {}
    for b in all_blocks:
        role_blocks[b] = role_of_block(b)

    def order(blocks):
        return sorted(blocks, key=lambda b: stable_uid('ord', b))

    # 林型对：R 30 / V 20
    r_pairs = [p for p in cand_pairs if role_blocks[p['block']] == ROLE_R]
    v_pairs_all = [p for p in v_pairs if role_blocks[p['block']] == ROLE_V]
    sel_pairs_R = r_pairs[:PAIRS_R]
    sel_pairs_V = v_pairs_all[:PAIRS_V]
    # 若某侧不足，用另一侧块改派（整块改派，保 role 唯一）
    if len(sel_pairs_R) < PAIRS_R:
        need = PAIRS_R - len(sel_pairs_R)
        extra = [p for p in v_pairs_all if p not in sel_pairs_V][:need]
        for p in extra:
            role_blocks[p['block']] = ROLE_R
        sel_pairs_R += extra
        sel_pairs_V = [p for p in sel_pairs_V if role_blocks[p['block']] == ROLE_V]
    if len(sel_pairs_V) < PAIRS_V:
        need = PAIRS_V - len(sel_pairs_V)
        extra = [p for p in r_pairs if p not in sel_pairs_R][:need]
        for p in extra:
            role_blocks[p['block']] = ROLE_V
        sel_pairs_V += extra
        sel_pairs_R = [p for p in sel_pairs_R if role_blocks[p['block']] == ROLE_R]
    sel_pairs_R = sel_pairs_R[:PAIRS_R]
    sel_pairs_V = sel_pairs_V[:PAIRS_V]

    # 争议点：R 60（池内，O 框 R 块） / V 40（独立随机，O 框 V 块）；分歧/稳定各半
    def pick_with_sep(df, need, per_block=MAX_DISPUTE_PER_BLOCK, sep_km=DISPUTE_MIN_SEP_KM, order_col='point_id'):
        out, used = [], {}
        if len(df) == 0:
            return out
        for bid, g in df.groupby('block'):
            g = g.sort_values(order_col) if order_col in g else g
            k = 0
            for _, r in g.iterrows():
                if k >= per_block:
                    break
                pts = used.setdefault(bid, [])
                km = [math.hypot((r['lon'] - p[0]) * 111.32 * math.cos(math.radians(r['lat'])),
                                 (r['lat'] - p[1]) * 110.57) for p in pts]
                if km and min(km) < sep_km:
                    continue
                pts.append((r['lon'], r['lat']))
                out.append(r)
                k += 1
                if len(out) >= need:
                    return out
        return out

    dcl_r = dcl[[role_blocks[b] == ROLE_R for b in dcl['block']]].copy()
    vdisp = vdf[vdf['group'] == 'dispute_v'].copy() if len(vdf) else pd.DataFrame(columns=['block', 'lon', 'lat', 'c_v1', 'c_v2', 'change'])
    if len(vdisp):
        vdisp = vdisp[[role_blocks.get(b) == ROLE_V for b in vdisp['block']]].copy()
    sel_disp = {ROLE_R: [], ROLE_V: []}
    for ch in ('disagree', 'stable'):
        got_r = pick_with_sep(dcl_r[dcl_r['change'] == ch], DISPUTE_R // 2)
        sel_disp[ROLE_R] += got_r
        got_v = pick_with_sep(vdisp[vdisp['change'] == ch] if len(vdisp) else vdisp, DISPUTE_V // 2, order_col='lon')
        sel_disp[ROLE_V] += got_v
    for role, need in ((ROLE_R, DISPUTE_R), (ROLE_V, DISPUTE_V)):
        got = len(sel_disp[role])
        if got < need:
            pool_df = dcl_r if role == ROLE_R else vdisp
            seen_ids = {str(getattr(r, 'point_id', '%s_%s' % (r['lon'], r['lat']))) for r in sel_disp[role]}
            rest = pool_df[[str(getattr(r, 'point_id', '%s_%s' % (r['lon'], r['lat']))) not in seen_ids
                            for r in pool_df.to_dict('records')]]
            sel_disp[role] += pick_with_sep(rest, need - got,
                                            order_col='point_id' if 'point_id' in rest.columns else 'lon')

    # ---- 7) 输出 private 侧表 ----
    rows = []
    for k, p in enumerate(sel_pairs_R):
        rows.append(dict(group='forest_pair', role=ROLE_R, pair_no=k, axis_pair='aspect', box='F',
                         a_lon=p['sun_lon'], a_lat=p['sun_lat'], b_lon=p['shade_lon'], b_lat=p['shade_lat'],
                         a_src=p.get('sun_src', ''), b_src=p.get('shade_src', ''),
                         a_cl_pool=p.get('sun_cl'), b_cl_pool=p.get('shade_cl'),
                         a_aspect=p['sun_aspect'], b_aspect=p['shade_aspect'],
                         a_elev=p['elev_sun'], b_elev=p['elev_shade'],
                         a_slope=p['slope_sun'], b_slope=p['slope_shade'],
                         a_c_v1=p.get('sun_c_v1'), a_c_v2=p.get('sun_c_v2'),
                         b_c_v1=p.get('shade_c_v1'), b_c_v2=p.get('shade_c_v2'),
                         dist_km=p['dist_km'], block=p['block'], a_slope_cls='sun', b_slope_cls='shade'))
    for k, p in enumerate(sel_pairs_V):
        rows.append(dict(group='forest_pair', role=ROLE_V, pair_no=k, axis_pair='aspect', box='F',
                         a_lon=p['sun_lon'], a_lat=p['sun_lat'], b_lon=p['shade_lon'], b_lat=p['shade_lat'],
                         a_src=p.get('sun_src', ''), b_src=p.get('shade_src', ''),
                         a_cl_pool=None, b_cl_pool=None,
                         a_aspect=p['sun_aspect'], b_aspect=p['shade_aspect'],
                         a_elev=p['elev_sun'], b_elev=p['elev_shade'],
                         a_slope=p['slope_sun'], b_slope=p['slope_shade'],
                         a_c_v1=p.get('sun_c_v1'), a_c_v2=p.get('sun_c_v2'),
                         b_c_v1=p.get('shade_c_v1'), b_c_v2=p.get('shade_c_v2'),
                         dist_km=p['dist_km'], block=p['block'], a_slope_cls='sun', b_slope_cls='shade'))
    for role in (ROLE_R, ROLE_V):
        for r in sel_disp[role]:
            cl = r['cl'] if 'cl' in r.index else np.nan
            src = r['src'] if 'src' in r.index else 'v_block_random'
            rows.append(dict(group='dispute', role=role, pair_no=None, axis_pair='', box='O',
                             a_lon=float(r['lon']), a_lat=float(r['lat']), b_lon=None, b_lat=None,
                             a_src=str(src), b_src='',
                             a_cl_pool=(int(cl) if cl == cl else None), b_cl_pool=None,
                             a_aspect=float(r['aspect']) if 'aspect' in r.index else np.nan,
                             b_aspect=np.nan,
                             a_elev=float(r['elev']) if 'elev' in r.index else np.nan, b_elev=np.nan,
                             a_slope=float(r['slope']) if 'slope' in r.index else np.nan, b_slope=np.nan,
                             a_c_v1=int(r['c_v1']), a_c_v2=int(r['c_v2']), b_c_v1=None, b_c_v2=None,
                             dist_km=np.nan, block=str(r['block']), a_slope_cls='',
                             change=str(r['change'])))
    priv = pd.DataFrame(rows)
    if 'box' in priv.columns:
        priv['box'] = priv['box'].fillna('F')
    emit('装配：R 对 %d / V 对 %d；争议 R %d / V %d' % (
        len(sel_pairs_R), len(sel_pairs_V), len(sel_disp[ROLE_R]), len(sel_disp[ROLE_V])))

    if a.dry:
        emit('--dry：不写产物')
        return
    priv.to_parquet(os.path.join(PRIV, 'candidates_private.parquet'), index=False)
    jdump(os.path.join(PRIV, 'role_blocks.json'), role_blocks)
    jdump(os.path.join(PRIV, 'experiment_context.json'), {
        'boxes': {k: dict(tile=v['tile'], region_bbox=v['bbox'], purpose=v['purpose'],
                          training_bbox=[v['bbox'][0] - BOX_TRAIN_EXPAND_DEG, v['bbox'][1] - BOX_TRAIN_EXPAND_DEG,
                                         v['bbox'][2] + BOX_TRAIN_EXPAND_DEG, v['bbox'][3] + BOX_TRAIN_EXPAND_DEG],
                          n_train_limit=20000, seeds=BOX_SEEDS, aef_year=S2_YEAR)
                  for k, v in BOXES.items()},
        'seen_blocks_excluded_from_V': SEEN_BOXES_2026_10_15,
        'train_exclusion': 'V 所在 0.25° 块整块排除；R 点保留在训练池内（否则 candidate 与 baseline 训练集相同）',
        'evaluation': 'V 标签由 audit 封存，experiment 只用 V 点做预测（同点同类别表）',
    })
    jdump(os.path.join(PRIV, 'scout_summary.json'), dict(
        time=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1),
        boxes={k: v['tile'] for k, v in BOXES.items()},
        pool_in_boxes=int(len(pool)), pool_forest=int(len(fcl)), pool_dispute=int(len(dcl)),
        cand_pairs_pool=int(len(cand_pairs)), cand_pairs_vblock=int(len(v_pairs)),
        v_random_candidates=int(len(vdf)),
        selected=dict(pairs_R=len(sel_pairs_R), pairs_V=len(sel_pairs_V),
                      dispute_R=len(sel_disp[ROLE_R]), dispute_V=len(sel_disp[ROLE_V]),
                      points_total=2 * (len(sel_pairs_R) + len(sel_pairs_V)) +
                      len(sel_disp[ROLE_R]) + len(sel_disp[ROLE_V])),
        v_min_dist_km=V_MIN_DIST_KM, mother_used=bool(mother_tree is not None),
        blocks=len(role_blocks), blocks_R=sum(1 for v in role_blocks.values() if v == ROLE_R),
        blocks_V=sum(1 for v in role_blocks.values() if v == ROLE_V),
        changemix=dict(disagree=int((dcl['change'] == 'disagree').sum()),
                       stable=int((dcl['change'] == 'stable').sum())),
    ))
    emit('完成 %.0fs → private/candidates_private.parquet' % (time.time() - t0))


if __name__ == '__main__':
    main()
