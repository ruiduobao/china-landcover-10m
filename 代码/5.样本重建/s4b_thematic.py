# -*- coding: utf-8 -*-
"""
s4b_thematic.py — r1 步骤3b：稀缺类专题产品采样 → r1_thematic.parquet
* 数据源: 数据/外部样本源/rare_class/（开放下载产品，图例见各 _说明.md，均已实测验证）
    11 园地   : AOMC苹果(30m,13省,Albers) / 柑橘2020(10m,4.7GB) / 茶叶2022 / 橡胶(儋州shp)
    184 红树林: GMW v3 2020 (25m) + 中国海岸湿地CW_2020值2
    185 盐沼  : 中国盐沼2022 + CW_2020值3
    186 潮滩  : CW_2020值1
    180-183   : GWL_FCS30 2020 栅格 (raw 181-187 → CODE_MAP 平移)
* 采样: 每 1°格每类 ≤10 点，固定种子，国界裁剪 + 生态硬规则
* 断点: 每产品结果即写 thematic_shards/<产品>.parquet，重跑自动跳过
"""
import os, re, sys, glob, json, time, zipfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

RARE = os.path.join(C.EXT, 'rare_class')
SHARD = os.path.join(C.WORK, 'thematic_shards')
PER_CELL = 10

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
    out = pd.DataFrame({'lon': lon, 'lat': lat, 'class_new': cls.astype(np.int64),
                        'src': src, 'src_conf': conf, 'year': year})
    if len(out):
        rng = np.random.default_rng(C.SEED)
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
        ck = np.floor(out.lon).astype(int) * 10000 + np.floor(out.lat).astype(int)
        out['_ck'] = ck
        out = out.groupby(['_ck', 'class_new']).head(PER_CELL).drop(columns=['_ck']).reset_index(drop=True)
        out['tier'] = 'external'
    return out

def scan_mask(path, value, cls, progress_name=''):
    """0.25° 块扫描 mask==value；返回候选点帧"""
    cand = []
    with rasterio.open(path) as s:
        tr4326 = (None if s.crs and s.crs.to_epsg() == 4326
                  else Transformer.from_crs(s.crs, 4326, always_xy=True))
        inv = ~s.transform
        b = s.bounds
        xs = np.arange(np.floor(b.left / 0.25) * 0.25, b.right, 0.25)
        ys = np.arange(np.floor(b.bottom / 0.25) * 0.25, b.top, 0.25)
        total = len(xs) * len(ys)
        done = 0
        for bx in xs:
            for by in ys:
                done += 1
                if done % 1500 == 0:
                    print(f'    [{progress_name}] 块 {done}/{total}', flush=True)
                c0, r0 = [int(v) for v in inv * (float(bx), float(by + 0.25))]
                c1, r1 = [int(v) for v in inv * (float(bx + 0.25), float(by))]
                c0 = max(0, c0); r0 = max(0, r0)
                c1 = min(s.width, c1); r1 = min(s.height, r1)
                if c1 <= c0 or r1 <= r0 or (c1 - c0) * (r1 - r0) > 4000 * 4000:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                if a.ndim == 3:
                    a = a[0]
                m = a == value
                if not m.any():
                    continue
                rows, cols = np.where(m)
                if len(rows) > 4000:
                    rng = np.random.default_rng(int(bx * 100) + int(by * 100) + 7)
                    k = rng.choice(len(rows), 4000, replace=False)
                    rows, cols = rows[k], cols[k]
                X, Y = riotrans.xy(s.transform, rows + r0 + 0.5, cols + c0 + 0.5)
                X = np.asarray(X); Y = np.asarray(Y)
                if tr4326 is not None:
                    X, Y = tr4326.transform(X, Y)
                cand.append(pd.DataFrame({'lon': np.asarray(X, float),
                                          'lat': np.asarray(Y, float), 'cls': cls}))
    return pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()

SHARDS = {}
def cache_or(name, fn):
    f = os.path.join(SHARD, name + '.parquet')
    if os.path.exists(f):
        print(f'[缓存] {name}: {len(pd.read_parquet(f))}', flush=True)
        return pd.read_parquet(f)
    t0 = time.time()
    d = fn()
    if len(d):
        d.to_parquet(f, index=False)
    print(f'[完成] {name}: {len(d)} ({time.time()-t0:.0f}s)', flush=True)
    return d

def run_all():
    os.makedirs(SHARD, exist_ok=True)
    t0 = time.time()
    frames = []

    # ---------- 11 园地族 ----------
    def aomc():
        cand = []
        for fp in glob.glob(os.path.join(RARE, 'AppleOrchard_China_AOMC', 'AOMC_30_*_2020.tif')):
            with rasterio.open(fp) as s:
                a = s.read(1)
                if a.ndim == 3:
                    a = a[0]
                m = a == 1
                if not m.any():
                    continue
                rows, cols = np.where(m)
                if len(rows) > 30000:
                    rng = np.random.default_rng(C.SEED + len(cand))
                    k = rng.choice(len(rows), 30000, replace=False)
                    rows, cols = rows[k], cols[k]
                X, Y = riotrans.xy(s.transform, rows + 0.5, cols + 0.5)
                tr = Transformer.from_crs(s.crs, 4326, always_xy=True)
                X, Y = tr.transform(np.asarray(X), np.asarray(Y))
                cand.append(pd.DataFrame({'lon': X, 'lat': Y, 'cls': 11}))
        return pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()
    frames.append(cache_or('aomc_apple', lambda: finalize(aomc(), 'aomc_apple_2020', 0.85, 2020)))

    cit = os.path.join(RARE, 'Citrus_China_2020', 'citrusmap202010m.tif')
    if os.path.exists(cit):
        frames.append(cache_or('citrus_2020', lambda: finalize(
            scan_mask(cit, 1, 11, 'citrus'), 'citrus_2020_10m', 0.85, 2020)))

    tea = None
    for nm in ['Teemap_2022_China.tif', 'TeaMap_2022_China.tif']:
        p = os.path.join(RARE, 'Teamap_China_2022', nm)
        if os.path.exists(p):
            tea = p; break
    if tea:
        frames.append(cache_or('teamap_2022', lambda: finalize(
            scan_mask(tea, 1, 11, 'tea'), 'teamap_2022', 0.80, 2022)))

    def rubber():
        rs = glob.glob(os.path.join(RARE, 'Rubber_Danzhou_Hainan', '*.shp'))
        if not rs:
            return pd.DataFrame()
        import geopandas as gpd
        g = gpd.read_file(rs[0])
        if g.crs and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        pts = g.geometry.representative_point()
        return pd.DataFrame({'lon': pts.x.to_numpy(), 'lat': pts.y.to_numpy(), 'cls': 11})
    frames.append(cache_or('rubber_danzhou', lambda: finalize(rubber(), 'rubber_danzhou', 0.75, 2020)))

    # ---------- 184 红树林 GMW ----------
    def gmw():
        zpath = os.path.join(RARE, 'GlobalMangroveWatch_v3', 'gmw_v3_2020_gtiff.zip')
        if not os.path.exists(zpath):
            return pd.DataFrame()
        z = zipfile.ZipFile(zpath)
        cand = []
        for n in z.namelist():
            if not n.endswith('.tif'):
                continue
            m = re.search(r'([NS])(\d+)([EW])(\d+)\.tif$', n)
            if not m:
                continue
            la = -int(m.group(2)) if m.group(1) == 'S' else int(m.group(2))
            lo = -int(m.group(4)) if m.group(3) == 'W' else int(m.group(4))
            if not (98 <= lo <= 128 and 16 <= la <= 27):
                continue
            try:
                with rasterio.open(f'/vsizip/{zpath}/{n}') as s:
                    a = s.read(1)
                    if a.ndim == 3:
                        a = a[0]
                    mm = a == 1
                    if not mm.any():
                        continue
                    rows, cols = np.where(mm)
                    if len(rows) > 2000:
                        rng = np.random.default_rng(abs(hash(n)) % (2**31))
                        k = rng.choice(len(rows), 2000, replace=False)
                        rows, cols = rows[k], cols[k]
                    X, Y = riotrans.xy(s.transform, rows + 0.5, cols + 0.5)
                    cand.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y), 'cls': 184}))
            except Exception as e:
                print(f'GMW {n}: {str(e)[:60]}', flush=True)
        return pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()
    frames.append(cache_or('gmw_2020', lambda: finalize(gmw(), 'gmw_v3_2020', 0.90, 2020)))

    # ---------- 中国海岸湿地 CW_2020 (1潮滩/2红树/3盐沼) ----------
    def cw2020_path():
        f = glob.glob(os.path.join(RARE, 'ChinaWetlands', 'extract', '**', 'CW_2020.tif'),
                      recursive=True)
        if f:
            return f[0]
        rars = glob.glob(os.path.join(RARE, 'ChinaWetlands', '*.rar'))
        if not rars:
            return None
        ex = os.path.join(RARE, 'ChinaWetlands', 'extract')
        os.makedirs(ex, exist_ok=True)
        import subprocess
        r = subprocess.run(['7z', 'x', rars[0], f'-o{ex}', 'CW_2020.tif', '-y'],
                           capture_output=True, text=True)
        f = glob.glob(os.path.join(ex, '**', 'CW_2020.tif'), recursive=True)
        return f[0] if f else None

    p = cw2020_path()
    if p:
        for v, cls in [(1, 186), (2, 184), (3, 185)]:
            frames.append(cache_or(f'cw2020_{cls}', lambda v=v, cls=cls: finalize(
                scan_mask(p, v, cls, f'CW{cls}'), 'china_coastal_wetland_2020', 0.85, 2020)))

    # ---------- 185 中国盐沼 2022 ----------
    sm = os.path.join(RARE, 'Saltmarsh_WCMC027', 'nationwide_2022_saltmarsh.tif')
    if os.path.exists(sm):
        def salt():
            d = scan_mask(sm, 1, 185, 'salt1')
            if len(d) == 0:
                d = scan_mask(sm, 2, 185, 'salt2')
            return d
        frames.append(cache_or('saltmarsh_2022', lambda: finalize(
            salt(), 'china_saltmarsh_2022', 0.85, 2022)))

    # ---------- GWL_FCS30 2020 栅格（湿地精细类） ----------
    def gwl():
        ex = os.path.join(RARE, 'GWL_FCS30_2020', 'extract')
        if not glob.glob(os.path.join(ex, '**', '*.tif'), recursive=True):
            os.makedirs(ex, exist_ok=True)
            import subprocess
            for r_ in glob.glob(os.path.join(RARE, 'GWL_FCS30_2020', '*.rar')):
                subprocess.run(['7z', 'x', r_, f'-o{ex}', '-y'],
                               capture_output=True, text=True)
        gmap = {181: 180, 182: 181, 183: 182, 184: 183, 185: 184, 186: 185, 187: 186}
        cand = {v: [] for v in gmap}
        for fp in glob.glob(os.path.join(ex, '**', '*.tif'), recursive=True):
            try:
                with rasterio.open(fp) as s:
                    bb = s.bounds
                    if bb.right < 70 or bb.left > 140 or bb.top < 15 or bb.bottom > 56:
                        continue
                    a = s.read(1)
                    if a.ndim == 3:
                        a = a[0]
                    for v in gmap:
                        m = a == v
                        if not m.any():
                            continue
                        rows, cols = np.where(m)
                        if len(rows) > 3000:
                            rng = np.random.default_rng(abs(hash(fp)) % (2**31) + v)
                            k = rng.choice(len(rows), 3000, replace=False)
                            rows, cols = rows[k], cols[k]
                        X, Y = riotrans.xy(s.transform, rows + 0.5, cols + 0.5)
                        cand[v].append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                                                     'cls': gmap[v]}))
            except Exception as e:
                print(f'GWL {os.path.basename(fp)}: {str(e)[:60]}', flush=True)
        parts = []
        for v, lst in cand.items():
            if lst:
                parts.append(finalize(pd.concat(lst, ignore_index=True),
                                      'gwl_fcs30_2020_raster', 0.90, 2020))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    frames.append(cache_or('gwl_raster', gwl))

    # ---------- 汇总 ----------
    out = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(os.path.join(C.WORK, 'r1_thematic.parquet'), index=False)
    print('\nr1_thematic 合计:', len(out))
    if len(out):
        print(out.groupby(['class_new', 'src']).size().to_string())
    print(f'({time.time()-t0:.0f}s) →', os.path.join(C.WORK, 'r1_thematic.parquet'))

if __name__ == '__main__':
    run_all()
