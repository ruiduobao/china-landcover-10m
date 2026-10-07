# -*- coding: utf-8 -*-
"""
s3b_more_sources.py — r1 步骤2b：补充高价值源 → r1_ext2.parquet
* 新增源（用户指定扩容）：
    1 GALF_2020          刘良云团队全球不透水占比 2020（~930m, 0-1 float32）
                         frac≥0.6 → 不透水候选，GUB 拆 190/200，conf 0.70
    2 CN30 精细2020      中国30米精细地表覆盖（GLC 精细码系，与 FCS30D 独立）
                         只补底座薄类：91/92/140/11/184/185 每格≤8；湿地亚类每格≤5
                         3×3 纯净像元约束，conf 0.80
* 输出: r1_ext2.parquet（tier='external'，与 r1_ext 同 schema）
"""
import os, re, sys, glob, io, json, time, zipfile, tempfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

R_KM = 0.1

_polys = None

def _w_contains(lon, lat):
    global _polys
    if _polys is None:
        _polys = G.china_polys()
    from shapely import contains_xy
    keep = np.zeros(len(lon), dtype=bool)
    for p in _polys:
        bx = (lon >= p.bounds[0]) & (lon <= p.bounds[2]) & \
             (lat >= p.bounds[1]) & (lat <= p.bounds[3])
        if bx.any():
            keep[bx] |= contains_xy(p, lon[bx], lat[bx])
    return keep

def finalize(df, src, conf, year):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    lon = df['lon'].to_numpy(float); lat = df['lat'].to_numpy(float)
    cls = df['cls'].to_numpy(int)
    b = C.BBOX
    m = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
    lon, lat, cls = lon[m], lat[m], cls[m]
    ok = _w_contains(lon, lat)
    lon, lat, cls = lon[ok], lat[ok], cls[ok]
    bad = C.eco_violation(lon, lat, cls)
    lon, lat, cls = lon[~bad], lat[~bad], cls[~bad]
    return pd.DataFrame({'lon': lon, 'lat': lat, 'class_new': cls.astype(np.int64),
                         'src': src, 'src_conf': conf, 'year': year})

def quota_per_cell(df, per_cell, seed=C.SEED):
    if not len(df):
        return df
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    ck = np.floor(df.lon).astype(int) * 10000 + np.floor(df.lat).astype(int)
    df['_ck'] = ck
    df = df.groupby(['_ck', 'class_new']).head(per_cell).drop(columns=['_ck']).reset_index(drop=True)
    return df

def pure_mask(a):
    c = a[1:-1, 1:-1]
    pure = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            pure &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return pure

# ---------------- 1) GALF 2020 ----------------
def galf():
    fp = os.path.join(C.EXT, 'GALF_2020', 'GALF_2020.tif')
    if not os.path.exists(fp):
        z = os.path.join(C.EXT, 'GALF_2020.zip')
        if os.path.exists(z):
            import zipfile as zf
            zf.ZipFile(z).extractall(os.path.join(C.EXT, 'GALF_2020'))
        else:
            return pd.DataFrame()
    cand = []
    with rasterio.open(fp) as s:
        inv = ~s.transform
        for bx in range(73, 137, 4):
            for by in range(17, 55, 4):
                c0, r0 = [int(v) for v in inv * (float(bx), float(min(by + 4, 55)))]
                c1, r1 = [int(v) for v in inv * (float(min(bx + 4, 137)), float(by))]
                c0 = max(0, c0); r0 = max(0, r0)
                c1 = min(s.width, c1); r1 = min(s.height, r1)
                if c1 <= c0 or r1 <= r0:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                m = np.isfinite(a) & (a >= 0.6)
                if not m.any():
                    continue
                rows, cols = np.where(m)
                X, Y = riotrans.xy(s.transform, rows + r0 + 0.5, cols + c0 + 0.5)
                cand.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y), 'cls': 190}))
    df = finalize(pd.concat(cand, ignore_index=True) if cand else pd.DataFrame(),
                  'galf_2020', 0.70, 2020)
    # GUB 拆 190/200
    if len(df):
        in_gub = G.gub_contains(df['lon'].to_numpy(), df['lat'].to_numpy())
        df['class_new'] = np.where(in_gub, 190, 200).astype(np.int64)
    return quota_per_cell(df, 8)

# ---------------- 2) CN30 精细 2020（只补薄类） ----------------
CN30_TARGET = {91: 8, 92: 8, 140: 8, 11: 8, 184: 5, 185: 5,
               180: 5, 181: 5, 182: 5, 183: 5, 186: 5}
CMAP = np.full(256, -1, dtype=np.int16)
for a_, b_ in C.CODE_MAP.items():
    if 0 <= a_ < 256:
        CMAP[a_] = b_

def cn30():
    z = zipfile.ZipFile(C.CN30_ZIP)
    inners = [n for n in z.namelist() if n.endswith('.zip')]
    print(f'   CN30 嵌套瓦片: {len(inners)}', flush=True)
    cand = []
    for ii, inner_name in enumerate(inners):
        tmp = None
        try:
            zi = zipfile.ZipFile(C.CN30_ZIP)
            inner = zipfile.ZipFile(io.BytesIO(zi.read(inner_name)))
            tifn = [n for n in inner.namelist() if n.lower().endswith('.tif')][0]
            data = inner.read(tifn)
            tmp = os.path.join(tempfile.gettempdir(), f'cn30b_{os.getpid()}_{ii}.tif')
            with open(tmp, 'wb') as f:
                f.write(data)
            with rasterio.open(tmp) as s:
                tr = Transformer.from_crs(4326, s.crs, always_xy=True)
                b = s.bounds
                lons, lats = tr.transform([b.left, b.right], [b.bottom, b.top])
                if max(lons) < 73 or min(lons) > 136 or max(lats) < 17 or min(lats) > 54:
                    continue
                inv = ~s.transform
                for cx in range(70, 140):
                    for cy in range(14, 56):
                        cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                        cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                        cc0 = max(0, cc0 - 1); rr0 = max(0, rr0 - 1)
                        cc1 = min(s.width, cc1 + 2); rr1 = min(s.height, rr1 + 2)
                        if cc1 <= cc0 or rr1 <= rr0 or (cc1 - cc0) * (rr1 - rr0) > 4200 * 4200:
                            continue
                        a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                        if a.ndim == 3:
                            a = a[0]
                        a = a.astype(np.uint8)
                        pm = pure_mask(a)
                        center = a[1:-1, 1:-1]
                        rows, cols = np.where(pm & (CMAP[center] > 0))
                        if len(rows) == 0:
                            continue
                        mcls = CMAP[center[rows, cols]]
                        for cls_v, per in CN30_TARGET.items():
                            idx = np.flatnonzero(mcls == cls_v)
                            if len(idx) == 0:
                                continue
                            rng = np.random.default_rng(cx * 10000 + cy + cls_v)
                            take = min(per, len(idx))
                            pick = rng.choice(idx, take, replace=False)
                            X, Y = riotrans.xy(s.transform, rows[pick] + rr0 + 0.5,
                                               cols[pick] + cc0 + 0.5)
                            cand.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                                                      'cls': cls_v}))
        except Exception as e:
            print(f'CN30 {os.path.basename(inner_name)}: {str(e)[:80]}', flush=True)
        finally:
            if tmp and os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
        if (ii + 1) % 20 == 0:
            print(f'  CN30 瓦片 {ii+1}/{len(inners)} 累计候选 {sum(len(x) for x in cand)}', flush=True)
    df = finalize(pd.concat(cand, ignore_index=True) if cand else pd.DataFrame(),
                  'cn30_fine_2020', 0.80, 2020)
    return df

def main():
    t0 = time.time()
    base = pd.read_parquet(C.R1_BASE, columns=['lon', 'lat'])
    tree = None

    frames = []
    print('1) GALF 2020 不透水 …', flush=True)
    g = galf()
    print(f'   GALF 候选 {len(g):,}', flush=True)
    frames.append(g)

    print('2) CN30 精细 2020 薄类补采 …', flush=True)
    c = cn30()
    print(f'   CN30 候选 {len(c):,}', flush=True)
    frames.append(c)

    ext2 = pd.concat([f for f in frames if len(f)], ignore_index=True)
    print(f'合计候选 {len(ext2):,}，与底座 100m 互斥 …', flush=True)
    from scipy.spatial import cKDTree
    bt = cKDTree(np.column_stack([
        base.lon.to_numpy() * 111.32 * np.cos(np.radians(base.lat.to_numpy())),
        base.lat.to_numpy() * 110.57]))
    km = np.column_stack([ext2.lon.to_numpy() * 111.32 * np.cos(np.radians(ext2.lat.to_numpy())),
                          ext2.lat.to_numpy() * 110.57])
    d, _ = bt.query(km, k=1)
    ext2 = ext2[d > R_KM].reset_index(drop=True)
    ext2['tier'] = 'external'
    ext2.to_parquet(os.path.join(C.WORK, 'r1_ext2.parquet'), index=False)
    print('\nr1_ext2:', len(ext2))
    print(ext2.groupby(['class_new', 'src']).size().to_string())
    json.dump({'total': int(len(ext2)),
               'by_src': {k: int(v) for k, v in ext2['src'].value_counts().items()}},
              open(os.path.join(C.WORK, 'r1_ext2_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
