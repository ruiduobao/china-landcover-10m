# -*- coding: utf-8 -*-
"""
n9_esri_national.py — ESRI 全国 10m 年度（2017–2021，G 盘 41 瓦/年）采样 → 年份专属样本
* 路径: G:\data\Land_Cover\high_resolution\ESRI_worldcover\2017-2021年全国土地利用分类数据（精度为10m）\{年}\{带}\*.tif (UTM)
* 类码（ESRI 9 类，已实证 7=built 5=crops）:
  1水→202  3草→130  4洪泛→181  5作物→10  6灌丛→121  8裸→201  9雪→220
  2树→跳过（30类无法细分）  7建成→class_new=7 哨兵，合并阶段按 GUB 拆 190/200
* 每 1° 格每类 ≤12 点；输出: 数据/外部样本源/esri_national_yearly.parquet
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from pyproj import Transformer
from multiprocessing import Pool

BASE = r'G:\data\Land_Cover\high_resolution\ESRI_worldcover\2017-2021年全国土地利用分类数据（精度为10m）'
OUT = '数据/外部样本源/esri_national_yearly.parquet'
BBOX = (73, 17, 136, 54)
LUT = np.full(16, -1, dtype=np.int16)
for a, b in {1: 202, 3: 130, 4: 181, 5: 10, 6: 121, 8: 201, 9: 220, 7: 7, 2: -1, 0: -1, 10: -1, 11: -1}.items():
    LUT[a] = b

def year_tiles(year):
    out = []
    for fp in glob.glob(os.path.join(BASE, str(year), '**', '*.tif'), recursive=True):
        try:
            with rasterio.open(fp) as s:
                b = s.bounds
                tr = Transformer.from_crs(s.crs, 4326, always_xy=True)
                lons, lats = tr.transform([b.left, b.right], [b.bottom, b.top])
                if max(lons) > 72 and min(lons) < 136 and max(lats) > 17 and min(lats) < 54:
                    out.append(fp)
        except Exception:
            pass
    return out

def task(args):
    fp, year = args
    out = []
    try:
        with rasterio.open(fp) as s:
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            b = s.bounds
            c0f, r0f = ~s.transform * (b.left, b.top)
            w, h = s.width, s.height
            # 1°格遍历（tile 覆盖的经纬范围）
            tr2 = Transformer.from_crs(s.crs, 4326, always_xy=True)
            lons, lats = tr2.transform([b.left, b.right], [b.bottom, b.top])
            for cx in range(int(np.floor(min(lons))), int(np.ceil(max(lons)))):
                for cy in range(int(np.floor(min(lats))), int(np.ceil(max(lats)))):
                    if not (BBOX[0] <= cx < BBOX[2] and BBOX[1] <= cy < BBOX[3]):
                        continue
                    xs, ys = [], []
                    for X in (cx, cx + 1):
                        for Y in (cy, cy + 1):
                            x, y = tr.transform(X, Y); xs.append(x); ys.append(y)
                    cc0 = max(0, int(min(xs)) - 1); rr0 = max(0, int(max(ys)) - 1)
                    cc1 = min(w, int(max(xs)) + 2); rr1 = min(h, int(min(ys)) + 2)
                    if cc1 <= cc0 or rr1 <= rr0 or (cc1-cc0)*(rr1-rr0) > 20000**2:
                        continue
                    a = s.read(1, window=Window(cc0, rr0, cc1-cc0, rr1-rr0))
                    rows, cols = np.where(a > 0)
                    if len(rows) == 0:
                        continue
                    vals = a[rows, cols]
                    rng = np.random.default_rng(abs(hash((year, cx, cy))) % (2**32))
                    for cls in np.unique(vals):
                        new = LUT[int(cls)]
                        if new <= 0:
                            continue
                        idx = np.flatnonzero(vals == cls)
                        take = min(12, len(idx))
                        pick = rng.choice(idx, take, replace=False)
                        for k in pick:
                            x, y = s.transform * (cols[k] + 0.5, rows[k] + 0.5)
                            lo, la = tr2.transform(x, y)
                            out.append({'lon': lo, 'lat': la, 'class_new': int(new),
                                        'class_raw': int(cls), 'src': f'esri_nat_{year}',
                                        'src_conf': 0.7, 'year': year})
        if out:
            return pd.DataFrame(out)
        return None
    except Exception as e:
        print('ERR', os.path.basename(fp), str(e)[:100], flush=True)
        return None

if __name__ == '__main__':
    tasks = []
    for y in range(2017, 2022):
        for fp in year_tiles(y):
            tasks.append((fp, y))
    print('年度瓦片任务:', len(tasks), flush=True)
    frames = []
    t0 = time.time()
    with Pool(6) as p:
        for i, r in enumerate(p.imap_unordered(task, tasks, chunksize=1)):
            if r is not None:
                frames.append(r)
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print('ESRI 年度样本:', len(df))
    print(dict(df.class_new.value_counts()))
