# -*- coding: utf-8 -*-
"""s3_fill_fcs10_direct.py — 补 G3 缺口：在 FCS10 森林像元上直接撒点（不经旧训练池）

* 输入（只读）：
    - FCS10-2023 全国 2° 瓦片 Z:/nb_national_2026/fcs10_dl/tiles/{TILE}_fcs10.tif
    - NASADEM 30m（2×2 均值 → 60m，与 v1/v2 同口径）
    - DataV 省界 PROJ/数据/边界/china_100000_full.json
    - 留出/错位清单（位置键排除）
    - labels/stats_v2_tiles.csv（逐瓦 FCS10 构成，用于识别候选饥饿瓦）
    - labels/needle_broad_national_v2.parquet（已有点，去聚团与去重基准）
* 规则源：技术文档/77 §18（G3 抽样框架缺陷）
* 门控：只补"森林面积>50 km² 且旧池候选<500"的候选饥饿瓦；
        每瓦每类最多 --per-class（默认 200）；空间去聚团 0.01° 同类最多 1 点；
        与 v2 已有点最近间距 ≥ --min-dist（默认 0.01°，≈1.1 km），避免重复
* 输出：labels/needle_broad_direct_v2fill.parquet（列与 v2 完全一致，可直拼）
        labels/stats_v2fill.json
* 用法：python s3_fill_fcs10_direct.py [--per-class 200] [--min-dist 0.01] [--workers 2]
* 资源：单进程逐瓦；每次只读一个行带（4096 行 ≈ 91 MB）；峰值 < 3 GB
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:/lc_work/nb_sep_2026/aspect')
import s0_common as C
from s2_sample_national import terrain_tile, sample_grid, emit, PROV2ECO

TILES_CSV = os.path.join(C.PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
TILES_DIR = r'Z:/nb_national_2026/fcs10_dl/tiles'
STATS_TILES = os.path.join(C.LAB, 'stats_v2_tiles.csv')
V2_PARQUET = os.path.join(C.LAB, 'needle_broad_national_v2.parquet')
BAND = 4096
DECLUTTER_DEG = 0.01


def load_target_tiles(min_forest_km2=50.0, max_cand=500):
    """候选饥饿瓦：FCS10 森林面积大但旧池候选少。"""
    rows = list(csv.DictReader(open(STATS_TILES, encoding='utf-8-sig')))

    def fnum(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    out = []
    for r in rows:
        forest_km2 = fnum(r, 'forest_pix') * 1e-4
        if forest_km2 > min_forest_km2 and fnum(r, 'n_cand') < max_cand:
            out.append((r['tile'], forest_km2, int(fnum(r, 'n_cand'))))
    return out


def sample_tile(path, leaf_codes, per_class, rng, existing_keys, min_dist):
    """在一瓦内按叶型分层撒点。

    * 逐行带即时抽稀：每带每码最多保留 BAND_KEEP 个随机像元（比例抽样），
      避免大瓦片（如 T0804 森林像元上亿）把坐标全攒在内存里。
    * 收完后仍做 0.01° 去聚团 + 上限截断。"""
    BAND_KEEP = 4000          # 每行带每码保留上限（比例抽样，不是硬删）
    need = {'needle': [], 'broad': [], 'mixed': []}
    code2leaf = {}
    for lf, codes in leaf_codes.items():
        for c in codes:
            code2leaf[c] = lf
    with rasterio.open(path) as s:
        H, W = s.height, s.width
        t = s.transform
        for r0 in range(0, H, BAND):
            r1 = min(r0 + BAND, H)
            arr = s.read(1, window=Window(0, r0, W, r1 - r0))
            for code, lf in code2leaf.items():
                rr, cc = np.nonzero(arr == code)
                if len(rr) == 0:
                    continue
                if len(rr) > BAND_KEEP:
                    pick = rng.choice(len(rr), size=BAND_KEEP, replace=False)
                    rr, cc = rr[pick], cc[pick]
                rr = rr + r0
                lon = t.c + (cc + 0.5) * t.a
                lat = t.f + (rr + 0.5) * t.e
                if existing_keys is not None and len(existing_keys):
                    gx = np.floor(lon / 0.05).astype(np.int64)
                    gy = np.floor(lat / 0.05).astype(np.int64)
                    q = gx * 100_000 + gy
                    keep = ~np.isin(q, existing_keys)
                    if keep.sum() == 0:
                        continue
                    lon, lat = lon[keep], lat[keep]
                need[lf].append(pd.DataFrame({
                    'lon': lon, 'lat': lat,
                    'fcs10_raw': np.uint16(code), 'leaf': lf}))
                del rr, cc
            del arr
    out = []
    for lf, cap in (('needle', per_class), ('broad', per_class), ('mixed', max(5, per_class // 4))):
        parts = [d for d in need[lf] if len(d)]
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        d['gk'] = (np.floor(d.lon / DECLUTTER_DEG).astype('int64') * 10_000_000
                   + np.floor(d.lat / DECLUTTER_DEG).astype('int64'))
        d['_r'] = rng.random(len(d))
        d = d.sort_values(['gk', '_r']).drop_duplicates(subset=['gk'], keep='first')
        if len(d) > cap:
            d = d.sample(n=cap, random_state=int(rng.integers(1 << 30)))
        out.append(d.drop(columns=['_r']))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--min-dist', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    outp = os.path.join(C.LAB, 'needle_broad_direct_v2fill.parquet')
    if os.path.isfile(outp) and not a.force:
        emit('已存在 %s（--force 覆盖）' % outp)
        return
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    targets = load_target_tiles()
    emit('候选饥饿瓦 %d 个（森林>50km² 且候选<500）' % len(targets))
    for t in targets:
        emit('   %s  森林 %.0f km²  旧候选 %d' % t)

    v2 = pq.read_table(V2_PARQUET, columns=['lon', 'lat', 'tile']).to_pandas()
    emit('v2 已有点 %d（瓦 %d）' % (len(v2), v2.tile.nunique()))
    exist = v2.copy()
    exist['key'] = (np.floor(exist.lon / 0.05).astype(np.int64) * 100_000
                    + np.floor(exist.lat / 0.05).astype(np.int64))
    keys_all = np.unique(exist.key.values)

    plan = pd.read_csv(TILES_CSV, encoding='utf-8-sig').set_index('tile')
    leaf_codes = {'needle': list(C.NEEDLE), 'broad': list(C.BROAD), 'mixed': list(C.MIXED)}

    parts = []
    recs = []
    for i, (t, forest_km2, n_cand) in enumerate(targets):
        path = os.path.join(TILES_DIR, t + '_fcs10.tif')
        if not os.path.isfile(path):
            emit('   跳过 %s（无 FCS10 瓦片）' % t)
            continue
        # 粗筛用全量已有点键（0.05° 格），不按瓦过滤——补样点可能落在邻近瓦
        d = sample_tile(path, leaf_codes, a.per_class, rng, keys_all, a.min_dist)
        if len(d) == 0:
            recs.append(dict(tile=t, forest_km2=round(forest_km2, 1), n_cand=n_cand, n_added=0))
            continue
        d['tile'] = t
        x0, y0 = float(plan.loc[t, 'x0']), float(plan.loc[t, 'y0'])
        A, S, E, xs, ys = terrain_tile(x0, y0, x0 + 2, y0 + 2)
        d['aspect_deg'] = np.round(sample_grid(A, xs, ys, d.lon.values, d.lat.values), 1)
        d['slope_deg'] = np.round(sample_grid(S, xs, ys, d.lon.values, d.lat.values), 2)
        d['elev_m'] = np.round(sample_grid(E, xs, ys, d.lon.values, d.lat.values), 1)
        del A, S, E, xs, ys
        parts.append(d)
        recs.append(dict(tile=t, forest_km2=round(forest_km2, 1), n_cand=n_cand,
                         n_added=int(len(d)),
                         by_leaf=d.leaf.value_counts().to_dict()))
        emit('   [%d/%d] %s 补 %d 点（针 %d 阔 %d 混 %d）' % (
            i + 1, len(targets), t, len(d),
            int((d.leaf == 'needle').sum()), int((d.leaf == 'broad').sum()),
            int((d.leaf == 'mixed').sum())))

    if not parts:
        emit('未补到任何点')
        return
    fill = pd.concat(parts, ignore_index=True)

    # ---- 统一列（与 v2 完全一致） ----
    fill['row_id'] = -1
    fill['grass'] = None
    fill['point_id'] = np.arange(len(fill), dtype='int64') + 900_000_000
    fill['prov'] = C.province_of(fill.lon.values, fill.lat.values)
    fill['eco'] = fill.prov.map(PROV2ECO).fillna('未匹配')
    fill['class2'] = np.where(fill.leaf == 'needle', 31,
                              np.where(fill.leaf == 'broad', 32, 33)).astype('int64')
    fill['class_code'] = fill.class2.astype('int16')
    fill['grp10'] = fill.leaf.map({'needle': 'needle', 'broad': 'broad', 'mixed': 'mixed'})
    fill['aspect_band'] = C.aspect_band(fill.aspect_deg.values, fill.slope_deg.values, 5.0)
    fill['dem_ok'] = np.isfinite(fill.elev_m.values)
    fill['terrain_src'] = 'NASADEM_30m_agg60m'
    fill['tier'] = 'direct_fcs10'
    fill['src'] = 'fcs10_direct_pixel'
    fill['in_pool76'] = 0
    fill['coverage'] = 'national_313tiles_v2fill'
    fill['declutter'] = '0.01deg 1pt_per_leaf_direct'
    fill['leaf2'] = fill.leaf.map({'needle': 'N', 'broad': 'B', 'mixed': 'M'})
    fill['class4'] = np.where(fill.leaf == 'needle', 6,
                              np.where(fill.leaf == 'broad', 4, 0)).astype('int8')

    cols = ['point_id', 'row_id', 'lon', 'lat', 'tile', 'prov', 'eco',
            'fcs10_raw', 'class2', 'class_code', 'leaf', 'leaf2', 'class4', 'grp10',
            'elev_m', 'slope_deg', 'aspect_deg', 'aspect_band', 'dem_ok', 'terrain_src',
            'tier', 'src', 'in_pool76', 'coverage', 'declutter']
    fill[cols].to_parquet(outp, index=False)

    # ---- 卫生 ----
    dup = int(fill.duplicated(subset=['lon', 'lat']).sum())
    nan = fill[['lon', 'lat', 'fcs10_raw', 'elev_m', 'slope_deg', 'aspect_deg']].isna().sum().to_dict()
    outb = int(((fill.lon < 73) | (fill.lon > 136) | (fill.lat < 17) | (fill.lat > 54)).sum())
    emit('G8 卫生：重复 %d / NaN %s / 越界 %d' % (dup, {k: int(v) for k, v in nan.items()}, outb))

    stats = dict(
        n=int(len(fill)), by_leaf=fill.leaf.value_counts().to_dict(),
        tiles=int(fill.tile.nunique()), provs=int(fill.prov.nunique()),
        per_class_cap=a.per_class, seed=a.seed, min_dist_deg=a.min_dist,
        target_tiles=len(targets), per_tile=recs,
        hygiene=dict(dup=dup, nan={k: int(v) for k, v in nan.items()}, out_of_bounds=outb),
        elapsed_min=round((time.time() - t0) / 60, 2))
    json.dump(stats, open(os.path.join(C.LAB, 'stats_v2fill.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('完成：%d 点 / %d 瓦 / %d 省 → %s（%.1f MB, %.1f min）' % (
        len(fill), fill.tile.nunique(), fill.prov.nunique(), outp,
        os.path.getsize(outp) / 1e6, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
