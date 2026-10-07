# -*- coding: utf-8 -*-
"""
s2_ingest_base.py — r1 步骤1：从 64,800 个 GLC_FCS30D 稳定样本 CSV 重导中国底座
* 数据源即"GEE 7年稳定+形态学腐蚀"采样网格（CSV 内已是稳定核样本），本步骤：
  1) 瓦片粗筛（与中国 bbox 相交）
  2) 国界精确裁剪（s1_geom.china_contains，DataV）
  3) CODE_MAP 码表映射（lc_conf 权威）
  4) GLC 190 不透水点 GUB2020 城乡分割（in→190 城镇 / out→200 乡村）
* 输出: r1_base.parquet（lon/lat/class_glc/class_new/urban/year/tile_id）
* 自校验: 与旧 v1 逐点比对（行数 + 坐标集合全等），确认重导无误后再进后续步骤。
"""
import os, re, sys, glob, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

OUT_SHARD_DIR = os.path.join(C.WORK, 'base_shards')

# CODE_MAP → 256 查找表
CMAP = np.full(256, -1, dtype=np.int16)
for a, b in C.CODE_MAP.items():
    if 0 <= a < 256:
        CMAP[a] = b

_cpolys = None

def _winit():
    global _cpolys
    _cpolys = G.china_polys()

def _w_contains(lon, lat):
    from shapely import contains_xy
    keep = np.zeros(len(lon), dtype=bool)
    for p in _cpolys:
        bx = (lon >= p.bounds[0]) & (lon <= p.bounds[2]) & \
             (lat >= p.bounds[1]) & (lat <= p.bounds[3])
        if bx.any():
            keep[bx] |= contains_xy(p, lon[bx], lat[bx])
    return keep

def tile_task(fp):
    """单瓦片：读CSV → bbox → 国界 → 码表；不透明水点原样返回(待主进程GUB分割)"""
    m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv$', os.path.basename(fp))
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    b = C.BBOX
    if lon0 + 1 <= b[0] or lon0 >= b[2] or lat0 + 1 <= b[1] or lat0 >= b[3]:
        return None
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None
    if len(df) == 0:
        return None
    lon = df['lon'].to_numpy(float); lat = df['lat'].to_numpy(float)
    ok = (lon >= b[0] - 0.01) & (lon <= b[2] + 0.01) & \
         (lat >= b[1] - 0.01) & (lat <= b[3] + 0.01)
    if not ok.any():
        return None
    df = df[ok]
    lon = df['lon'].to_numpy(float); lat = df['lat'].to_numpy(float)
    inside = _w_contains(lon, lat)
    if not inside.any():
        return None
    df = df[inside]
    glc = df['class'].to_numpy(np.int64)
    new = CMAP[np.clip(glc, 0, 255)]
    keep = new > 0
    if not keep.any():
        return None
    df = df[keep]
    return pd.DataFrame({
        'lon': df['lon'].to_numpy(float),
        'lat': df['lat'].to_numpy(float),
        'class_glc': glc[keep],
        'class_new': new[keep].astype(np.int64),
        'year': df['year'].to_numpy(np.int64),
        'tile_id': df['tile_id'].astype(str).to_numpy(),
    })

def main():
    t0 = time.time()
    os.makedirs(OUT_SHARD_DIR, exist_ok=True)
    fps = glob.glob(os.path.join(C.GRID_DIR, 'Sample_Lon*_Lat*.csv'))
    # 只留与中国 bbox 相交的瓦片
    todo = []
    for fp in fps:
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv$', os.path.basename(fp))
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        b = C.BBOX
        if lon0 + 1 > b[0] and lon0 < b[2] and lat0 + 1 > b[1] and lat0 < b[3]:
            todo.append(fp)
    print(f'候选瓦片 {len(todo)} / {len(fps)}', flush=True)

    # 清空旧分片
    for f in glob.glob(os.path.join(OUT_SHARD_DIR, '*.parquet')):
        os.remove(f)

    n_total = 0
    buf, shard_i = [], 0
    with Pool(5, initializer=_winit) as p:
        for i, r in enumerate(p.imap_unordered(tile_task, todo, chunksize=16)):
            if r is not None and len(r):
                buf.append(r); n_total += len(r)
            if len(buf) >= 800:
                pd.concat(buf, ignore_index=True).to_parquet(
                    os.path.join(OUT_SHARD_DIR, f'shard_{shard_i:04d}.parquet'), index=False)
                shard_i += 1; buf = []
                print(f'  {i+1}/{len(todo)} 累计 {n_total:,} ({time.time()-t0:.0f}s)', flush=True)
    if buf:
        pd.concat(buf, ignore_index=True).to_parquet(
            os.path.join(OUT_SHARD_DIR, f'shard_{shard_i:04d}.parquet'), index=False)

    # 合并分片
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT_SHARD_DIR, '*.parquet')))]
    base = pd.concat(parts, ignore_index=True)
    del parts
    n_raw = len(base)
    print(f'国界内点: {n_raw:,}', flush=True)

    # ---- GUB 城乡分割（主进程一次性向量化） ----
    m_imp = base['class_new'].to_numpy() == 190
    n_imp = int(m_imp.sum())
    if n_imp:
        in_gub = G.gub_contains(base.loc[m_imp, 'lon'].to_numpy(),
                                base.loc[m_imp, 'lat'].to_numpy())
        cls = base['class_new'].to_numpy(np.int64).copy()
        cls[m_imp] = np.where(in_gub, 190, 200)
        base['class_new'] = cls
        base['urban'] = 0
        base.loc[m_imp, 'urban'] = in_gub.astype(np.int64)
        print(f'GUB 分割: 不透水 {n_imp:,} → 城镇 {int(in_gub.sum()):,} / 乡村 {n_imp-int(in_gub.sum()):,}', flush=True)
    else:
        base['urban'] = -1

    base.to_parquet(C.R1_BASE, index=False)
    vc = base['class_new'].value_counts().sort_index()
    print('r1_base 类别分布:')
    for k, v in vc.items():
        print(f'  {k:>4}: {v:,}')

    # ---- 与旧 v1 校验 ----
    print('\n=== 与旧 v1 逐点校验 ===', flush=True)
    v1 = pd.read_parquet(C.V1, columns=['lon', 'lat', 'class_new'])
    print(f'行数: r1={len(base):,}  v1={len(v1):,}  {"✓" if len(base)==len(v1) else "✗ 不一致!"}')
    a = base[['lon', 'lat', 'class_new']].sort_values(['lon', 'lat', 'class_new']).reset_index(drop=True)
    b = v1.sort_values(['lon', 'lat', 'class_new']).reset_index(drop=True)
    same = len(a) == len(b) and np.allclose(a.lon, b.lon, rtol=0, atol=1e-9) and \
           np.allclose(a.lat, b.lat, rtol=0, atol=1e-9) and (a.class_new == b.class_new).all()
    print(f'坐标+类别集合全等: {"✓ 完全一致" if same else "✗ 存在差异"}')
    if not same and len(a) == len(b):
        d = (a.class_new != b.class_new)
        print(f'  类别差异点数: {int(d.sum()):,}')
        dm = np.abs(a.lon - b.lon).max(), np.abs(a.lat - b.lat).max()
        print(f'  坐标最大差: {dm}')

    summary = {'total': int(len(base)), 'raw_before_gub': int(n_raw),
               'v1_total': int(len(v1)), 'match_v1': bool(same),
               'by_class': {str(k): int(v) for k, v in vc.items()},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    import json as _j
    _j.dump(summary, open(C.R1_BASE.replace('.parquet', '_summary.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('\n输出:', C.R1_BASE)

if __name__ == '__main__':
    main()
