# -*- coding: utf-8 -*-
"""s1_sample_2023.py — 从 FCS10 2023 十米瓦片抽取全国针阔两类样本（v1：76 瓦五省先行版）
* 输入：
    - FCS10 已对齐瓦片 F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/{TILE}_fcs10.tif
    - 2023 年度训练子集 F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet（提供位置与 AEF64）
    - FCS30D 已对齐瓦片（仅用于"与旧标签源不一致率"诊断，不作标签）
    - 1 km SRTM DEM（坡向/坡度/高程）
    - DataV 省界（省归属）
    - 留出/错位清单（位置键排除）
* 规则源：技术文档/76（G1–G10）、技术文档/75（坡向伪影与 FCS10 纠偏）、判据_针阔两类样本v1.json
* 门槛：G7 防泄漏=0；G8 卫生；G4 两类共存
* 输出：labels/_stage1_points.parquet（抽样点 + FCS10 原始码 + 地形 + 省 + 排除标记）
        labels/_stage1_tile_stats.csv（逐瓦 FCS10 像元构成与可得点数）
* 用法：python s1_sample_2023.py [--target 500] [--mixed-cap 100] [--force]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def read_codes_at_points(tif, lon, lat):
    """逐点取栅格码（原生 10 m 索引，最近邻，不插值）。返回 uint16 数组（越界=0）。"""
    out = np.zeros(len(lon), dtype='uint16')
    with rasterio.open(tif) as s:
        rr, cc = rasterio.transform.rowcol(s.transform, lon, lat)
        rr = np.asarray(rr); cc = np.asarray(cc)
        ok = (rr >= 0) & (rr < s.height) & (cc >= 0) & (cc < s.width)
        if not ok.any():
            return out
        r0, r1 = int(rr[ok].min()), int(rr[ok].max()) + 1
        c0, c1 = int(cc[ok].min()), int(cc[ok].max()) + 1
        arr = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        out[ok] = arr[rr[ok] - r0, cc[ok] - c0].astype('uint16')
    return out


def read_full_codes(tif):
    with rasterio.open(tif) as s:
        return s.read(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=500, help='每瓦每类目标点数')
    ap.add_argument('--mixed-cap', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    outp = os.path.join(C.LAB, '_stage1_points.parquet')
    statp = os.path.join(C.LAB, '_stage1_tile_stats.csv')
    if os.path.isfile(outp) and not a.force:
        emit('已存在 %s（--force 覆盖）' % outp)
        return

    tiles = sorted(f[:-len('_fcs10.tif')] for f in os.listdir(C.FCS10_TILES)
                   if f.endswith('_fcs10.tif'))
    emit('FCS10 瓦片 %d 个' % len(tiles))

    # ---- 2023 子集位置（提供 AEF64） ----
    sub = pq.read_table(os.path.join(C.YEARLY, 'r7_train_2023.parquet'),
                        columns=['row_id', 'lon', 'lat', 'class_new', 'tier', 'src']).to_pandas()
    sub['tile'] = C.tile_of(sub.lon.values, sub.lat.values)
    sub = sub[sub.tile.isin(set(tiles))].copy()
    emit('候选点（76 瓦内）%d' % len(sub))

    # ---- 位置排除 ----
    excl, ecounts = C.build_exclusion_positions()
    emit('排除索引：%s' % ecounts)
    keys = C.pos_keys(sub.lon.values, sub.lat.values)
    sub['excl_leak'] = [k in excl for k in keys]
    emit('命中留出/错位：%d（剔除后 %d）' % (int(sub.excl_leak.sum()), int((~sub.excl_leak).sum())))
    sub = sub[~sub.excl_leak].copy()

    # ---- 逐瓦读取 FCS10 / FCS30D ----
    codes = np.zeros(len(sub), dtype='uint16')
    old_raw = np.zeros(len(sub), dtype='uint16')
    tile_stats = []
    for i, t in enumerate(tiles):
        m = (sub.tile == t).values
        if not m.any():
            continue
        g = sub[m]
        f10 = os.path.join(C.FCS10_TILES, t + '_fcs10.tif')
        codes[m] = read_codes_at_points(f10, g.lon.values, g.lat.values)
        f30 = os.path.join(C.FCS30D_TILES, t + '_fcs30d.tif')
        if os.path.isfile(f30):
            old_raw[m] = read_codes_at_points(f30, g.lon.values, g.lat.values)
        # 瓦片像元级构成（用于报告与面积口径）
        arr = read_full_codes(f10)
        bc = np.bincount(arr.ravel(), minlength=256)
        tile_stats.append(dict(tile=t, n_pix=int(arr.size),
                              needle_pix=int(bc[list(C.NEEDLE)].sum()),
                              broad_pix=int(bc[list(C.BROAD)].sum()),
                              mixed_pix=int(bc[list(C.MIXED)].sum()),
                              forest_pix=int(bc[list(C.FOREST10)].sum()),
                              cand_points=int(m.sum())))
        del arr, bc
        if (i + 1) % 10 == 0:
            emit('  读取进度 %d/%d' % (i + 1, len(tiles)))
    sub['fcs10_raw'] = codes
    sub['fcs30d_raw'] = old_raw

    # ---- 分组与排除 ----
    sub['leaf'] = [C.leaf_of(int(c)) for c in sub.fcs10_raw]
    sub['class2'] = [C.class2_of(int(c)) for c in sub.fcs10_raw]
    sub['grp10'] = [C.FCS10_GROUPS.get(int(c), 'unknown') for c in sub.fcs10_raw]
    emit('FCS10 组构成：%s' % sub.grp10.value_counts().to_dict())
    cand = sub[sub.leaf.notna()].copy()
    emit('FCS10 森林类点：%d（needle %d / broad %d / mixed %d）' % (
        len(cand), (cand.leaf == 'needle').sum(), (cand.leaf == 'broad').sum(), (cand.leaf == 'mixed').sum()))

    # ---- 0.01° 去聚团：同格同类最多 1 点 ----
    cand['gk'] = (np.floor(cand.lon.values / C.DECLUTTER_DEG).astype('int64') * 10_000_000
                  + np.floor(cand.lat.values / C.DECLUTTER_DEG).astype('int64'))
    rng = np.random.default_rng(a.seed)
    cand['_r'] = rng.random(len(cand))
    cand = cand.sort_values(['leaf', 'gk', '_r'])
    before = len(cand)
    cand = cand.drop_duplicates(subset=['leaf', 'gk'], keep='first')
    emit('去聚团：%d → %d（同 1km 格同类仅留 1 点）' % (before, len(cand)))
    # 双类同格核查（needle 与 broad 同格最多各 1）
    gk_leaf = cand.groupby('gk')['leaf'].nunique()
    emit('同格双类格数 %d（上限即总格数的 1 倍，允许）' % int((gk_leaf > 1).sum()))

    # ---- 每瓦每类抽样 ----
    parts = []
    for t, g in cand.groupby('tile'):
        for leaf, cap in (('needle', a.target), ('broad', a.target), ('mixed', a.mixed_cap)):
            d = g[g.leaf == leaf]
            if len(d) == 0:
                continue
            if len(d) > cap:
                d = d.sample(n=cap, random_state=a.seed)
            parts.append(d)
    samp = pd.concat(parts, ignore_index=True)
    emit('抽样结果 %d（%s）' % (len(samp), samp.leaf.value_counts().to_dict()))

    # ---- 地形（1 km DEM） ----
    slope, asp, elev, tfm, crs, w, h = C.dem_arrays('1k')
    elev_v, ok1 = C.sample_raster(elev, tfm, samp.lon.values, samp.lat.values, w, h)
    slope_v, ok2 = C.sample_raster(slope, tfm, samp.lon.values, samp.lat.values, w, h)
    asp_v, ok3 = C.sample_raster(asp, tfm, samp.lon.values, samp.lat.values, w, h)
    samp['elev_m'] = np.round(elev_v, 1)
    samp['slope_deg'] = np.round(slope_v, 2)
    samp['aspect_deg'] = np.round(asp_v, 1)
    samp['aspect_band'] = C.aspect_band(asp_v, slope_v, 5.0)
    samp['dem_ok'] = ok1 & ok2

    # ---- 省归属 ----
    samp['prov'] = C.province_of(samp.lon.values, samp.lat.values)
    emit('省数 %d，未匹配 %d' % (samp.prov.nunique(), int((samp.prov == '').sum())))

    # ---- 卫生（G8） ----
    samp['point_id'] = samp.row_id.astype('int64')
    dup = samp.duplicated(subset=['lon', 'lat']).sum()
    nan = samp[['lon', 'lat', 'fcs10_raw', 'elev_m', 'slope_deg', 'aspect_deg']].isna().sum().to_dict()
    outb = ((samp.lon < 73) | (samp.lon > 136) | (samp.lat < 17) | (samp.lat > 54)).sum()
    emit('G8 卫生：重复位置 %d / NaN %s / 越界 %d' % (dup, nan, outb))

    cols = ['point_id', 'row_id', 'lon', 'lat', 'tile', 'prov',
            'fcs10_raw', 'fcs30d_raw', 'class2', 'leaf', 'grp10',
            'elev_m', 'slope_deg', 'aspect_deg', 'aspect_band', 'dem_ok',
            'class_new', 'tier', 'src']
    samp[cols].to_parquet(outp, index=False)
    emit('写 %s（%d 行）' % (outp, len(samp)))
    pd.DataFrame(tile_stats).to_csv(statp, index=False, encoding='utf-8-sig')
    emit('写 %s' % statp)


if __name__ == '__main__':
    main()
