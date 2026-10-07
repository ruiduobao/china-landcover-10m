# -*- coding: utf-8 -*-
"""
s4_fcs10.py — r1 步骤3：GLC_FCS10 2023 全覆盖重采 → r1_topup.parquet
* 修复旧链两处缺陷：
    1) n8 旧采样（多次运行叠加、无国界、等额采样）——v4/v5_train 里的 FCS10 层主要是它；
    2) n18_resample 的窗口上限 8000×8000 在 10m 分辨率下把几乎所有 1° 格判为
       "窗口过大"跳过 → 17,299 点仅覆盖 73/2331 个 1° 格，空间严重偏倚。
* 本脚本：每 1° 格整窗读取（uint8 约 123MB，安全）：
    - 3×3 纯净像元约束（中心+8邻域 raw 同码才可采，用户指定）
    - 面积加权配额：非稀缺类 frac≥2% 才采，quota=80×frac；稀缺类(RARE_CLASSES)每格最多15点
    - 固定种子（cx*10000+cy）、国界裁剪、生态硬规则
* 输出: r1_topup.parquet（lon/lat/class_new/class_raw/src/src_conf/year）
"""
import os, re, sys, glob, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

OUT_DIR = os.path.join(C.WORK, 'topup_shards')

FMAP = np.full(256, -1, dtype=np.int16)
for a, b in C.CODE_MAP.items():
    if 0 <= a < 256:
        FMAP[a] = b
for a, b in C.FCS10_EXTRA.items():
    FMAP[a] = b

# 3×3 纯净掩膜的 8 个位移（中心 vs 邻域）
_SHIFTS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

def pure_mask(a):
    """a: (h+2, w+2) 带边窗 raw；返回中心 (h, w) 的 3×3 同码布尔"""
    c = a[1:-1, 1:-1]
    pure = np.ones(c.shape, dtype=bool)
    for dr, dc in _SHIFTS:
        pure &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return pure

_polys = None

def _winit():
    global _polys
    _polys = G.china_polys()

def _w_contains(lon, lat):
    from shapely import contains_xy
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
        # 左上角命名: 瓦片 lat 范围 [lat0-5, lat0)
        if lon0 + 5 > b[0] and lon0 < b[2] and lat0 > b[1] and lat0 - 5 < b[3]:
            out.append(fp)
    return out

def tile_task(fp):
    m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    b = C.BBOX
    # 瓦片按左上角命名：E075N35 覆盖 lon 75-80、lat 30-35
    if lon0 + 5 <= b[0] or lon0 >= b[2] or lat0 - 5 >= b[3] or lat0 <= b[1]:
        return None
    out = []
    try:
        with rasterio.open(fp) as s:
            inv = ~s.transform
            for cx in range(max(int(np.floor(b[0])), lon0), min(int(np.ceil(b[2])), lon0 + 5)):
                for cy in range(max(int(np.floor(b[1])), lat0 - 5), min(int(np.ceil(b[3])), lat0)):
                    # 1° 窗口 + 1px 边（3×3 邻域需要）
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
                    pm = pure_mask(a)
                    center = a[1:-1, 1:-1]
                    mapped = FMAP[center]
                    rows, cols = np.where(pm & (mapped > 0))
                    if len(rows) == 0:
                        continue
                    vals_c = center[rows, cols]
                    mcls = mapped[rows, cols]
                    rng = np.random.default_rng(cx * C.FCS10_SEED_BASE + cy)
                    # 全类候选分组
                    picks = []
                    for cls in np.unique(mcls):
                        idx = np.flatnonzero(mcls == cls)
                        if cls in C.RARE_CLASSES:
                            quota = min(C.FCS10_TOPUP_PER_CELL, len(idx))
                        else:
                            frac = len(idx) / pm.size
                            if frac < C.FCS10_MIN_FRAC:
                                continue
                            quota = min(max(1, int(C.FCS10_TOTAL_CAP * len(idx) / pm.size)), len(idx))
                        if quota <= 0:
                            continue
                        for k in rng.choice(idx, quota, replace=False):
                            picks.append(k)
                    if not picks:
                        continue
                    picks = np.array(picks)
                    # 像元中心坐标
                    xs, ys = rasterio.transform.xy(s.transform, rows[picks] + rr0 + 0.5,
                                                   cols[picks] + cc0 + 0.5)
                    xs = np.asarray(xs); ys = np.asarray(ys)
                    cls_pick = mcls[picks].astype(int)
                    raw_pick = vals_c[picks].astype(int)
                    ok = _w_contains(xs, ys)
                    if not ok.any():
                        continue
                    xs, ys, cls_pick, raw_pick = xs[ok], ys[ok], cls_pick[ok], raw_pick[ok]
                    bad = C.eco_violation(xs, ys, cls_pick)
                    if bad.any():
                        xs, ys, cls_pick, raw_pick = xs[~bad], ys[~bad], cls_pick[~bad], raw_pick[~bad]
                    if len(xs) == 0:
                        continue
                    out.append(pd.DataFrame({
                        'lon': xs, 'lat': ys, 'class_new': cls_pick.astype(np.int64),
                        'class_raw': raw_pick, 'src': C.FCS10_SRC,
                        'src_conf': C.FCS10_CONF, 'year': C.FCS10_YEAR}))
        # 每 5° 瓦片落盘
        if out:
            df = pd.concat(out, ignore_index=True)
            df.to_parquet(os.path.join(OUT_DIR, os.path.basename(fp)[:-4] + '.parquet'), index=False)
        return None
    except Exception as e:
        print('ERR', os.path.basename(fp), str(e)[:120], flush=True)
        return None

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    tiles = china_tiles()
    print(f'FCS10 中国区 5°瓦片: {len(tiles)}', flush=True)
    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(OUT_DIR, '*.parquet'))}
    todo = [f for f in tiles if os.path.basename(f)[:-4] not in done]
    print(f'已完成 {len(tiles)-len(todo)}，待跑 {len(todo)}', flush=True)

    with Pool(4, initializer=_winit) as p:
        for i, _ in enumerate(p.imap_unordered(tile_task, todo, chunksize=1)):
            if (i + 1) % 5 == 0:
                print(f'  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)', flush=True)

    # 合并 + 与底座/外部源 100m 互斥
    import pandas as pd_  # noqa
    shards = sorted(glob.glob(os.path.join(OUT_DIR, '*.parquet')))
    parts = [pd.read_parquet(f) for f in shards]
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f'重采合计: {len(df):,} ({time.time()-t0:.0f}s)', flush=True)

    from scipy.spatial import cKDTree
    ref = []
    for f in [C.R1_BASE, C.R1_EXT]:
        if os.path.exists(f):
            ref.append(pd.read_parquet(f, columns=['lon', 'lat']))
    if ref:
        ref = pd.concat(ref, ignore_index=True)
        tree = cKDTree(np.column_stack([
            ref.lon.to_numpy() * 111.32 * np.cos(np.radians(ref.lat.to_numpy())),
            ref.lat.to_numpy() * 110.57]))
        km = np.column_stack([df.lon.to_numpy() * 111.32 * np.cos(np.radians(df.lat.to_numpy())),
                              df.lat.to_numpy() * 110.57])
        d, _ = tree.query(km, k=1)
        n0 = len(df)
        df = df[d > 0.1].reset_index(drop=True)
        print(f'距底座/外部 <100m 剔除: {n0-len(df):,} → {len(df):,}')

    df['tier'] = 'external'
    df.to_parquet(C.R1_TOPUP, index=False)
    print('\nr1_topup 类别分布:')
    for k, v in df['class_new'].value_counts().sort_index().items():
        print(f'  {k:>4}: {v:,}')
    cells = len(set(zip(np.floor(df.lon).astype(int), np.floor(df.lat).astype(int))))
    print(f'覆盖 1°格数: {cells} / ~2331')

    summary = {'total': int(len(df)), 'cells_covered': int(cells),
               'by_class': {str(k): int(v) for k, v in df['class_new'].value_counts().sort_index().items()},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(C.R1_TOPUP.replace('.parquet', '_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('输出:', C.R1_TOPUP, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

