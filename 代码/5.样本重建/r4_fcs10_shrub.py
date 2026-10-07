# -*- coding: utf-8 -*-
"""
r4_fcs10_shrub.py — 从 GLC_FCS10 2023 全国重采灌丛样本（修复 120/121 谱系分歧与截断）
* 背景：FCS30D(2020) 的灌丛标注与 FCS10(2023) 系统性不一致（120 池点 90% 被 FCS10 判常绿阔叶林；
  121 池点被 FCS10 判草地 35%/裸地 25%），且盲区保底规则叠加产品覆盖不均 → 空间硬截断。
  对标产品即 GLC_FCS10，故灌丛标签以其为权威。
* 采样：全瓦片扫描（不限过渡格），3×3 纯净像元，映射后仅保留目标灌丛类：
    raw 121 → 产品 120（常绿灌丛）；raw 120/122 → 产品 121（落叶灌丛）
  每 0.05° 细格每类 ≤2（固定种子），国界 + 生态过滤。
* 输出: 数据/本地处理/样本重建/r4_fcs10_shrub.parquet
"""
import os, re, sys, glob, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

OUT_DIR = os.path.join(C.WORK, 'r4_shrub_shards')
# 目标：raw → 产品码（仅灌丛三码）
TARGET = {121: 120, 120: 121, 122: 121}
SHURB_LUT = np.zeros(256, dtype=bool)
for _k in TARGET:
    SHURB_LUT[_k] = True

def pure_mask(a):
    c = a[1:-1, 1:-1]
    p = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            p &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return p

_polys = None

def _winit():
    global _polys
    _polys = G.china_polys()

def _w_contains(lon, lat):
    from shapely import contains_xy
    global _polys
    if _polys is None:
        _polys = G.china_polys()
    keep = np.zeros(len(lon), dtype=bool)
    for p in _polys:
        bx = (lon >= p.bounds[0]) & (lon <= p.bounds[2]) & \
             (lat >= p.bounds[1]) & (lat <= p.bounds[3])
        if bx.any():
            keep[bx] |= contains_xy(p, lon[bx], lat[bx])
    return keep

def china_tiles():
    out = []
    for fp in glob.glob(os.path.join(C.FCS10_TILE_ROOT, 'GLC_FCS10maps_*', '*.tif')):
        m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not m:
            continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        b = C.BBOX
        if lon0 + 5 > b[0] and lon0 < b[2] and lat0 > b[1] and lat0 - 5 < b[3]:
            out.append(fp)
    return out

def tile_task(fp):
    m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    b = C.BBOX
    if lon0 + 5 <= b[0] or lon0 >= b[2] or lat0 - 5 >= b[3] or lat0 <= b[1]:
        return None
    out = []
    try:
        with rasterio.open(fp) as s:
            inv = ~s.transform
            for cx in range(max(int(np.floor(b[0])), lon0), min(int(np.ceil(b[2])), lon0 + 5)):
                for cy in range(max(int(np.floor(b[1])), lat0 - 5), min(int(np.ceil(b[3])), lat0)):
                    cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                    cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                    cc0 = max(0, cc0 - 1); rr0 = max(0, rr0 - 1)
                    cc1 = min(s.width, cc1 + 2); rr1 = min(s.height, rr1 + 2)
                    if cc1 <= cc0 or rr1 <= rr0:
                        continue
                    a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                    if a.ndim == 3:
                        a = a[0]
                    a = a.astype(np.uint8)
                    center = a[1:-1, 1:-1]
                    rows, cols = np.where(pure_mask(a) & SHURB_LUT[center])
                    if len(rows) == 0:
                        continue
                    raw_c = center[rows, cols]
                    cls_c = np.array([TARGET[int(v)] for v in raw_c], dtype=np.int64)
                    X, Y = riotrans.xy(s.transform, rows + rr0 + 0.5, cols + cc0 + 0.5)
                    X = np.asarray(X); Y = np.asarray(Y)
                    rng = np.random.default_rng(cx * 10000 + cy + 4242)
                    for cc in (120, 121):
                        m2 = cls_c == cc
                        if not m2.any():
                            continue
                        fine = (np.floor(X[m2] / 0.05).astype(np.int64) * 100000 +
                                np.floor(Y[m2] / 0.05).astype(np.int64))
                        df = pd.DataFrame({'lon': X[m2], 'lat': Y[m2], 'class_new': cc,
                                           'class_raw': raw_c[m2], '_f': fine})
                        df = df.sample(frac=1.0, random_state=int(rng.integers(1e9)))
                        df = df.groupby('_f', group_keys=False).head(2)
                        out.append(df[['lon', 'lat', 'class_new', 'class_raw']])
        if out:
            df = pd.concat(out, ignore_index=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            df.to_parquet(os.path.join(OUT_DIR, os.path.basename(fp)[:-4] + '.parquet'), index=False)
        return None
    except Exception as e:
        print('ERR', os.path.basename(fp), str(e)[:90], flush=True)
        return None

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    tiles = china_tiles()
    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(OUT_DIR, '*.parquet'))}
    todo = [f for f in tiles if os.path.basename(f)[:-4] not in done]
    print(f'瓦片 {len(tiles)}，待跑 {len(todo)}', flush=True)
    with Pool(3, initializer=_winit) as p:
        for i, _ in enumerate(p.imap_unordered(tile_task, todo, chunksize=1)):
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)', flush=True)
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT_DIR, '*.parquet')))]
    df = pd.concat(parts, ignore_index=True)
    print(f'候选 {len(df):,}', flush=True)
    # 国界 + 生态
    ok = _w_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[ok].reset_index(drop=True)
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    df['src'] = 'glc_fcs10_2023_shrub'
    df['src_conf'] = 0.85
    df['year'] = 2023
    df['tier'] = 'external'
    df['agree_n'] = -1
    df.to_parquet(os.path.join(C.WORK, 'r4_fcs10_shrub.parquet'), index=False)
    print('输出:', len(df), dict(df.class_new.value_counts()))
    print(f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

