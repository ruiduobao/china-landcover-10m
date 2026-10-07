# -*- coding: utf-8 -*-
"""
n8_glc_fcs10_sample.py — GLC_FCS10 2023（10m 30类，本地 217 瓦）采样 → 第二教师样本
* 瓦片: E:\地理所\论文\全球土地覆盖数据\数据\GLC_FCS10\GLC_FCS10maps_2023_E{L}-E{L+5}\GLC_FCS10_2023_E{L}N{U}.tif
  命名=左上角（E115N40 覆盖 115-120E, 35-40N）
* 码表（UserGuide 无表，实证+FCS30 同源推定）:
  191=城镇不透水 192=乡村不透水 210=水 200=裸地 11=旱地 20=乔灌/灌溉
  森林/湿地沿用 FCS30 约定（CODE_MAP 已含交换逻辑）
* 采样: 每 1° 格窗口，每类 ≤15 点
* 输出: 数据/外部样本源/glc_fcs10_samples.parquet
"""
import os, sys, glob, re, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'Z:/Mywork/论文/中国土地覆盖数据/代码/0.本地流水线')
from lc_conf import CODE_MAP

D = r'E:\地理所\论文\全球土地覆盖数据\数据\GLC_FCS10'
OUT = '数据/外部样本源/glc_fcs10_samples.parquet'
BBOX = (73, 17, 136, 54)

FCS10_EXTRA = {191: 190, 192: 200}   # 城乡不透水（实证）
LUT = np.full(256, -1, dtype=np.int16)
for a, b in CODE_MAP.items():
    if 0 <= a < 256:
        LUT[a] = b
LUT[191] = 190; LUT[192] = 200; LUT[210] = 202
# FCS30 的 210→202 已在 CODE_MAP；防覆盖
LUT[200] = 201; LUT[201] = 201; LUT[202] = 201

def china_tiles():
    out = []
    for fp in glob.glob(os.path.join(D, '**', '*.tif'), recursive=True):
        m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not m:
            continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        # 瓦片覆盖 [lon0, lon0+5] x [lat0-5, lat0]
        if lon0 + 5 > BBOX[0] and lon0 < BBOX[2] and lat0 > BBOX[1] and lat0 - 5 < BBOX[3]:
            out.append(fp)
    return out

def sample_cell(s, cx, cy, lut, cap=15):
    inv = ~s.transform
    c0, r0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
    c1, r1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
    c0 = max(0, c0); r0 = max(0, r0)
    c1 = min(s.width, c1); r1 = min(s.height, r1)
    if c1 <= c0 or r1 <= r0:
        return None
    a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    rows, cols = np.where(a > 0)
    if len(rows) == 0:
        return None
    vals = a[rows, cols]
    out = []
    rng = np.random.default_rng(abs(hash((cx, cy))) % (2**32))
    for cls in np.unique(vals):
        new = lut[int(cls)] if int(cls) < 256 else -1
        if new <= 0:
            continue
        idx = np.flatnonzero(vals == cls)
        take = min(cap, len(idx))
        pick = rng.choice(idx, take, replace=False)
        for k in pick:
            x, y = s.transform * (cols[k] + 0.5, rows[k] + 0.5)
            out.append({'lon': x, 'lat': y, 'class_new': int(new),
                        'class_raw': int(cls), 'src': 'glc_fcs10_2023',
                        'src_conf': 0.85, 'year': 2023})
    return out

def tile_task(fp):
    try:
        frames = []
        with rasterio.open(fp) as s:
            b = s.bounds
            for cx in range(int(np.floor(b.left)), int(np.ceil(b.right))):
                for cy in range(int(np.floor(b.bottom)), int(np.ceil(b.top))):
                    if not (BBOX[0] <= cx < BBOX[2] and BBOX[1] <= cy < BBOX[3]):
                        continue
                    res = sample_cell(s, cx, cy, LUT, cap=15)
                    if res:
                        frames.extend(res)
        if frames:
            return pd.DataFrame(frames)
        return None
    except Exception as e:
        print('ERR', os.path.basename(fp), str(e)[:100], flush=True)
        return None

if __name__ == '__main__':
    tiles = china_tiles()
    print('中国区瓦片:', len(tiles), flush=True)
    frames = []
    t0 = time.time()
    with Pool(6) as p:
        for i, r in enumerate(p.imap_unordered(tile_task, tiles, chunksize=1)):
            if r is not None:
                frames.append(r)
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(tiles)} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print('GLC_FCS10 样本:', len(df))
    print(dict(df.class_new.value_counts().head(15)))

