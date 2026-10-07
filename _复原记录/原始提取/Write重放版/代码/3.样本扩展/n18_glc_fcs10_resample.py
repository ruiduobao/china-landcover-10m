# -*- coding: utf-8 -*-
"""
n18_glc_fcs10_resample.py — GLC_FCS10 重新采样（修复矩形簇根因）
* 根因: n8 采样器对每 (1°格 × 出现类) 无脑采 15 点，在多类混合格导致每类 ~450 点
         （稀有类仅占 1% 面积也获得与主导类等量样本），叠加多次运行更放大。
* 修复策略:
  1. 只采**占比 ≥5%** 的类（丢弃过渡带混合像元）
  2. **面积加权配额**：类占比 × 总配额（每 1° 格类样本数 ∝ 面积占比）
  3. 每 1° 格总配额 = 60 点（主导类多、稀有类少甚至没有）
  4. 国界裁剪（DataV）
  5. 生态规则（ECO_RULES，剔除生态不合理点）
  6. 单次运行、固定种子（可复现）
* 输出: 覆盖 glc_fcs10_samples.parquet
"""
import os, sys, glob, re, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool
import shapely
from shapely import contains_xy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'Z:\Mywork\论文\中国土地覆盖数据\代码\0.本地流水线')
from lc_conf import CODE_MAP

D = r'E:\地理所\论文\全球土地覆盖数据\数据\GLC_FCS10'
OUT = '数据/外部样本源/glc_fcs10_samples.parquet'
BOUND = r'数据/边界/china_100000_full.json'
BBOX = (73, 17, 136, 54)
MIN_FRAC = 0.05           # 类占比低于 5% 则不采
TOTAL_CAP = 60            # 每 1° 格总样本数（全类共享）

LUT = np.full(256, -1, dtype=np.int16)
for a, b in CODE_MAP.items():
    if 0 <= a < 256:
        LUT[a] = b
LUT[191] = 190; LUT[192] = 200; LUT[210] = 202
LUT[200] = 201; LUT[201] = 201; LUT[202] = 201

# 生态规则（与 n16 相同）
ECO_HARD = {
    120: [lambda x, y: y > 42, lambda x, y: (x>80)&(x<95)&(y>32), lambda x, y: (x>80)&(x<90)&(y>36)],
    121: [lambda x, y: (x>80)&(x<92)&(y>37)],
    51:  [lambda x, y: y < 18],
    52:  [lambda x, y: y < 18],
    220: [lambda x, y: y < 30],
}

def china_polys():
    import json
    d = json.load(open(BOUND, encoding='utf-8'))
    ps = []
    for f in d['features']:
        sg = shapely.geometry.shape(f['geometry'])
        ps += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return ps

_POLYS = None
def _init(bound_path):
    global _POLYS
    _POLYS = china_polys()

def china_contains(lon, lat):
    from shapely import contains_xy
    return contains_xy(_POLYS, lon, lat)

def tile_task(fp):
    """单瓦片：逐 1° 格主导类面积加权采样"""
    m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
    if not m:
        return None
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    if lon0 + 5 <= BBOX[0] or lon0 >= BBOX[2] or lat0 <= BBOX[1] or lat0 - 5 >= BBOX[3]:
        return None
    out = []
    try:
        with rasterio.open(fp) as s:
            b = s.bounds
            for cx in range(int(np.floor(max(b.left, BBOX[0]))), int(np.ceil(min(b.right, BBOX[2])))):
                for cy in range(int(np.floor(max(b.bottom, BBOX[1]))), int(np.ceil(min(b.top, BBOX[3])))):
                    inv = ~s.transform
                    c0, r0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                    c1, r1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                    c0 = max(0, c0); r0 = max(0, r0)
                    c1 = min(s.width, c1); r1 = min(s.height, r1)
                    if c1 <= c0 or r1 <= r0:
                        continue
                    a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                    rows, cols = np.where(a > 0)
                    if len(rows) == 0:
                        continue
                    vals = a[rows, cols]
                    new_vals = LUT[vals]
                    valid = new_vals > 0
                    if valid.sum() == 0:
                        continue
                    vals_v = vals[valid]; new_v = new_vals[valid]; rows_v = rows[valid]; cols_v = cols[valid]
                    # 每类计数
                    uniq, counts = np.unique(new_v, return_counts=True)
                    total_valid = len(new_v)
                    # 占比 ≥5% 才采
                    keep_classes = uniq[counts / total_valid >= MIN_FRAC]
                    if len(keep_classes) == 0:
                        continue
                    # 面积加权配额
                    rng = np.random.default_rng(cx * 10000 + cy)
                    for kc in keep_classes:
                        quota = max(1, int(TOTAL_CAP * counts[uniq == kc][0] / total_valid))
                        idx_k = np.flatnonzero(new_v == kc)
                        take = min(quota, len(idx_k))
                        pick = rng.choice(idx_k, take, replace=False)
                        for k in pick:
                            x, y = s.transform * (cols_v[k] + 0.5, rows_v[k] + 0.5)
                            new_cls = int(new_v[k])
                            # 生态规则
                            bad = False
                            for fn in ECO_HARD.get(new_cls, []):
                                if fn(x, y):
                                    bad = True; break
                            if bad:
                                continue
                            if not china_contains(x, y):
                                continue
                            out.append({'lon': x, 'lat': y, 'class_new': new_cls,
                                        'class_raw': int(vals_v[k]),
                                        'src': 'glc_fcs10_2023', 'src_conf': 0.85,
                                        'year': 2023})
        if out:
            return pd.DataFrame(out)
        return None
    except Exception as e:
        print('ERR', os.path.basename(fp), str(e)[:100], flush=True)
        return None

def china_tiles():
    out = []
    for fp in glob.glob(os.path.join(D, '**', '*.tif'), recursive=True):
        mm = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not mm:
            continue
        lon0, lat0 = int(mm.group(1)), int(mm.group(2))
        if lon0 + 5 > BBOX[0] and lon0 < BBOX[2] and lat0 > BBOX[1] and lat0 - 5 < BBOX[3]:
            out.append(fp)
    return out

if __name__ == '__main__':
    tiles = china_tiles()
    print('中国区瓦片:', len(tiles), flush=True)
    frames = []
    t0 = time.time()
    with Pool(6, initializer=_init, initargs=(BOUND,)) as p:
        for i, r in enumerate(p.imap_unordered(tile_task, tiles, chunksize=1)):
            if r is not None:
                frames.append(r)
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(tiles)} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    # 最终国界裁剪
    _init(BOUND)
    keep = china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[keep].reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print('GLC_FCS10 重采样本:', len(df))
    print(dict(df.class_new.value_counts().head(15)))
    print('输出:', OUT)
