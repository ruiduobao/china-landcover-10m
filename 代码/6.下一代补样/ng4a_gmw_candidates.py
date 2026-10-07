# -*- coding: utf-8 -*-
"""
ng4a_gmw_candidates.py — 用本地 GMW v3 中国瓦片生成红树林候选点（本地，零 EECU）

背景：GEE 公开目录没有红树林专题产品（`24` §三.6），而 **GMW v3 的中国瓦片已在本项目**
`数据/外部样本源/rare_class/GlobalMangroveWatch_v3/china_tiles_2020/`（54 个 25 m GeoTIFF，
EPSG:4326，值 1 = 红树林）。所以候选点改由**本地 GMW 掩膜**生成，再上 GEE 取证据波段
（WorldCover95 / GSW 潮间带 / GLO-30 海拔）做交叉验证 —— 这就是"GMW ∩ WorldCover 双证"，
不需要把任何数据上传到 GEE。

输出
  Z:\\...\\下一代补样\\ng_raw\\mangrove184\\gmw_candidates.parquet
    列：lon / lat / gmw(=1) / gmw_tile
用法: python ng4a_gmw_candidates.py [--step 6] [--max 20000]
      step=6 → 每 6 个像元取 1 个（约 150 m 网格）
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

TILES = os.path.join(P.PROJ, '数据/外部样本源/rare_class/GlobalMangroveWatch_v3/china_tiles_2020')
COAST = S.REGIONS['HN_COAST']  # 仅用于说明，不做空间限制（GMW 本身只在海岸）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=6, help='像元抽稀步长（6 ≈ 150 m 网格）')
    ap.add_argument('--max', type=int, default=20000, help='候选点上限')
    a = ap.parse_args()
    P.ensure_all(['mangrove184'])
    files = sorted(glob.glob(os.path.join(TILES, '*.tif')))
    if not files:
        raise SystemExit(f'未找到 GMW 瓦片: {TILES}')
    print(f'GMW 瓦片 {len(files)} 个，抽稀步长 {a.step}')
    t0 = time.time()
    rows = []
    n_mangrove_px = 0
    for f in files:
        tile = os.path.basename(f).replace('GMW_', '').replace('_2020_v3.tif', '')
        with rasterio.open(f) as ds:
            arr = ds.read(1)
            m = arr == 1
            if not m.any():
                continue
            n_mangrove_px += int(m.sum())
            rr, cc = np.nonzero(m)
            rr = rr[::a.step]
            cc = cc[::a.step]
            xs, ys = rasterio.transform.xy(ds.transform, rr, cc)
            rows.append(pd.DataFrame({'lon': np.asarray(xs, float),
                                      'lat': np.asarray(ys, float),
                                      'gmw': 1, 'gmw_tile': tile}))
        del arr, m
    if not rows:
        raise SystemExit('GMW 里没有任何红树林像元')
    df = pd.concat(rows, ignore_index=True)
    # 与母库国界/坐标合法性检查留给 ng9_qa；这里只做格网去重
    dlat = a.step * 0.000222
    df['_c'] = (np.floor(df.lat / dlat).astype(np.int64) * 100000
                + np.floor(df.lon / dlat).astype(np.int64))
    n0 = len(df)
    df = df.drop_duplicates('_c').drop(columns='_c')
    if len(df) > a.max:
        df = df.sample(a.max, random_state=20260913).reset_index(drop=True)
    out = os.path.join(P.RAW, 'mangrove184', 'gmw_candidates.parquet')
    df.to_parquet(out, index=False)
    print(f'红树林像元 {n_mangrove_px:,}（25 m）→ 候选 {n0:,} → 格网去重 {len(df) - 0:,} '
          f'→ 上限截取 {len(df):,}')
    print(f'经纬度范围 lon {df.lon.min():.2f}–{df.lon.max():.2f} / '
          f'lat {df.lat.min():.2f}–{df.lat.max():.2f}')
    print('涉及瓦片:', df.gmw_tile.nunique(), '个')
    print('输出:', out, f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
