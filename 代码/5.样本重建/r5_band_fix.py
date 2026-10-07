# -*- coding: utf-8 -*-
"""
r5_band_fix.py — 补采 FCS10 E130-E135 缺失带（中国最东部）所有类别
* 背景：E 盘 FCS10 解压副本缺 E130-E135 整带（文件夹从 E120-E125 跳到 E140-E145），
  导致所有 FCS10 派生样本在中国最东部（130-135°E，黑龙江抚远/双鸭山/牡丹江东部、吉林延边）
  形成 130°E 直线截断。Z 盘 zip 完好，用 /vsizip 直读补采。
* 采样：3×3 纯净像元；灌丛用专用映射（raw 121→120、raw 120/122→121），
  其余用 CODE_MAP+FCS10_EXTRA；每 0.05° 格每类 ≤2；国界+生态过滤。
* 输出: r5_band.parquet
"""
import os, re, sys, zipfile, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

ZIP = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/外部样本源/glc_fcs10_2023/GLC_FCS10maps_2023_E130-E135.zip'
OUT = os.path.join(C.WORK, 'r5_band.parquet')

FMAP = np.full(256, -1, dtype=np.int16)
for a, b in C.CODE_MAP.items():
    if 0 <= a < 256:
        FMAP[a] = b
for a, b in C.FCS10_EXTRA.items():
    FMAP[a] = b
# 灌丛专用（与 r4_fcs10_shrub 一致）
FMAP[121] = 120   # raw 121 常绿灌丛 → 产品 120
FMAP[120] = 121   # raw 120 一般灌丛 → 产品 121
FMAP[122] = 121   # raw 122 落叶灌丛 → 产品 121

def pure_mask(a):
    c = a[1:-1, 1:-1]
    p = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            p &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return p

def main():
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    tiles = sorted(n for n in z.namelist() if n.endswith('.tif'))
    # 只取中国纬度带（40-50N）与 E130 列
    use = [n for n in tiles if re.search(r'_E130N(40|45|50)\.tif$', n)]
    print('补采瓦片:', use, flush=True)
    out = []
    for name in use:
        p = f'/vsizip/{ZIP}/{name}'
        with rasterio.open(p) as s:
            inv = ~s.transform
            b = C.BBOX
            for cx in range(max(int(b[0]), 130), min(int(b[2]), 135)):
                for cy in range(max(int(b[1]), 35), min(int(b[3]), 50)):
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
                    rows, cols = np.where(pure_mask(a) & (FMAP[center] > 0))
                    if len(rows) == 0:
                        continue
                    cls_c = FMAP[center[rows, cols]].astype(np.int64)
                    raw_c = center[rows, cols]
                    X, Y = riotrans.xy(s.transform, rows + rr0 + 1.5, cols + cc0 + 1.5)
                    X = np.asarray(X); Y = np.asarray(Y)
                    rng = np.random.default_rng(cx * 10000 + cy + 555)
                    for cc in np.unique(cls_c):
                        m2 = cls_c == cc
                        fine = (np.floor(X[m2] / 0.05).astype(np.int64) * 100000 +
                                np.floor(Y[m2] / 0.05).astype(np.int64))
                        df = pd.DataFrame({'lon': X[m2], 'lat': Y[m2], 'class_new': cc,
                                           'class_raw': raw_c[m2], '_f': fine})
                        df = df.sample(frac=1.0, random_state=int(rng.integers(1e9)))
                        df = df.groupby('_f', group_keys=False).head(2)
                        out.append(df[['lon', 'lat', 'class_new', 'class_raw']])
        print(f'  {name} 完成 ({time.time()-t0:.0f}s)', flush=True)
    df = pd.concat(out, ignore_index=True)
    print('候选:', len(df), flush=True)
    ok = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[ok].reset_index(drop=True)
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    df['src'] = 'glc_fcs10_2023_band130'
    df['src_conf'] = 0.85
    df['year'] = 2023
    df['tier'] = 'external'
    df['agree_n'] = -1
    df.to_parquet(OUT, index=False)
    print('境内+生态内:', len(df))
    print('类别分布:', dict(df.class_new.value_counts().sort_index()))
    print(f'({time.time()-t0:.0f}s) → {OUT}')

if __name__ == '__main__':
    main()
