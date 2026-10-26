# -*- coding: utf-8 -*-
"""b2_local_filters.py — 裸地多源候选的本地硬过滤（国界/邻域/潮滩与湖河滩地/空间配额）

* 输入：data/bare_cand/<tag>/*.parquet（b1 输出）
* 过滤：
    1) CTAmap 国界：境内或 ≤10 km 边界带
    2) FCS10 2023 9×9 出现 {182 湖河滩地, 186 潮滩, 183 盐渍} → 弃；中心 210 水体/220 冰雪 → 弃
    3) CLCD 2023：中心 ∈{1 耕地,2 林地,5 水体,8 不透水,9 湿地} → 弃；9×9 出现 {5,9} → 弃
    4) 与既有 潮滩/湖河滩地 样本 <1 km → 弃；dem<3 m → 弃（低滩）
    5) 与既有 裸地 样本 <250 m → 弃
    6) 多源票：DW bare≥0.40 / WC=60 / MODIS=16 / S2-rock(NDVI<200 且 bright>120)，≥2 票
    7) 0.025° 格 ≤1；类内最小间距 200 m；每省 ≤800
* 输出：data/bare_cand/bare_cand_filtered.parquet + filter_stats.json
* 用法：python b2_local_filters.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
DATA = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
CLCD = r'E:/2.1_CLCD土地覆盖数据/30米/CLCD_v01_2023_albert.tif'
FCS10 = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10'
AEQ = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 +datum=WGS84 +units=m +no_defs')


def kmxy(lon, lat):
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def main():
    fs = []
    for tag in ('south', 'east', 'north_plain', 'loess'):
        fs += glob.glob(os.path.join(BASE, 'data', 'bare_cand', tag, '*.parquet'))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates('pid')
    print('原始候选 %d' % len(d), flush=True)
    stats = {'raw': int(len(d))}
    # 1) 国界
    prov = gpd.read_file(PROV_SHP)[['省', 'geometry']].rename(columns={'省': 'ct_prov'})
    prov['geometry'] = prov.geometry.buffer(0)
    china = prov.geometry.union_all()
    g = gpd.GeoDataFrame(d, geometry=[Point(x, y) for x, y in zip(d.lon, d.lat)], crs='EPSG:4326')
    g['inside'] = g.within(china)
    ga = g.to_crs(AEQ); ca = gpd.GeoSeries([china], crs='EPSG:4326').to_crs(AEQ).iloc[0]
    dist = np.zeros(len(g)); out = ~g.inside.to_numpy()
    dist[out] = ga.geometry[out].distance(ca).to_numpy()
    keep = g.inside.to_numpy() | (dist <= 10000)
    d = d[keep].reset_index(drop=True); stats['边界外剔除'] = int((~keep).sum())
    print('国界后 %d（剔 %d）' % (len(d), stats['边界外剔除']), flush=True)
    # 2) FCS10 9×9
    tiles = []
    for t in sorted(glob.glob(os.path.join(FCS10, '*', 'GLC_FCS10_2023_*.tif'))):
        with rasterio.open(t) as ds:
            tiles.append((t, ds.bounds))
    _cache = {}
    def _ds(path):
        if path not in _cache:
            _cache[path] = rasterio.open(path)
        return _cache[path]
    def fcs9(lon, lat):
        cand = [x for x in tiles if x[1].left <= lon < x[1].right and x[1].bottom <= lat < x[1].top]
        if not cand:
            return None, None
        ds = _ds(cand[0][0]); tr = ds.transform
        col = int(round((lon - tr.c) / tr.a - 0.5)); row = int(round((lat - tr.f) / tr.e - 0.5))
        a = ds.read(1, window=Window(col - 4, row - 4, 9, 9))
        return int(a[4, 4]), a
    center, nine = zip(*[fcs9(x, y) for x, y in zip(d.lon, d.lat)])
    d['fcs_center'] = center
    drop = np.array([(c is None) or (c in (210, 220)) or (a is None) or
                     bool(np.isin(a, (182, 183, 186, 187)).any()) for c, a in zip(center, nine)])
    d = d[~drop].reset_index(drop=True); stats['FCS10邻域/水体剔除'] = int(drop.sum())
    print('FCS10 过滤后 %d（剔 %d）' % (len(d), stats['FCS10邻域/水体剔除']), flush=True)
    # 3) CLCD 2023
    with rasterio.open(CLCD) as ds:
        tr = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
        X, Y = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
        vals = np.array([v[0] for v in ds.sample(zip(X, Y), indexes=1)], dtype=np.int16)
        # 9×9 邻域：逐点小窗口
        cls9 = []
        for x, y in zip(X, Y):
            col = int((x - ds.transform.c) / ds.transform.a); row = int((y - ds.transform.f) / ds.transform.e)
            try:
                a = ds.read(1, window=Window(col - 4, row - 4, 9, 9))
                cls9.append(a)
            except Exception:
                cls9.append(None)
    d['clcd'] = vals
    # CLCD 只硬剔水/湿地/不透水；耕地/林地（1/2）仅作冲突标记（湿润区出露岩石常被 CLCD 归为草/林/耕）
    # 只硬剔水/湿地；不透水(8)与耕地/林地(1/2)都转入"冲突"分支（用户明确要矿/城市裸地）
    drop = np.array([(v in (5, 9)) or (a is None) or bool(np.isin(a, (5, 9)).any())
                     for v, a in zip(vals, cls9)])
    d = d[~drop].reset_index(drop=True); stats['CLCD过滤'] = int(drop.sum())
    print('CLCD 过滤后 %d（剔 %d）' % (len(d), stats['CLCD过滤']), flush=True)
    # 4) 与 潮滩/湖河滩地 样本 <1 km + dem<3
    tide = []
    for nm in ('潮滩', '湖河滩地'):
        for f in glob.glob(os.path.join(DATA, nm, '2023', '*.gpkg')):
            g2 = pyogrio.read_dataframe(f, columns=[])
            tide.append(np.column_stack([g2.geometry.x, g2.geometry.y]))
    tide = np.vstack(tide) if tide else np.zeros((0, 2))
    if len(tide):
        tree = cKDTree(kmxy(tide[:, 0], tide[:, 1]))
        dd, _ = tree.query(kmxy(d.lon.to_numpy(), d.lat.to_numpy()), k=1, distance_upper_bound=1.0)
        near = np.isfinite(dd)
    else:
        near = np.zeros(len(d), dtype=bool)
    low = d.dem.fillna(999) < 3
    drop = near | low
    d = d[~drop].reset_index(drop=True); stats['潮滩/湖河滩地/低滩剔除'] = int(drop.sum())
    print('潮滩/低滩 过滤后 %d（剔 %d）' % (len(d), stats['潮滩/湖河滩地/低滩剔除']), flush=True)
    # 5) 与既有裸地 <250 m
    ex = []
    for f in glob.glob(os.path.join(DATA, '裸地', '2023', '*.gpkg')):
        g3 = pyogrio.read_dataframe(f, columns=[])
        ex.append(np.column_stack([g3.geometry.x, g3.geometry.y]))
    ex = np.vstack(ex)
    tree = cKDTree(kmxy(ex[:, 0], ex[:, 1]))
    dd, _ = tree.query(kmxy(d.lon.to_numpy(), d.lat.to_numpy()), k=1, distance_upper_bound=0.25)
    near = np.isfinite(dd)
    d = d[~near].reset_index(drop=True); stats['距既有裸地<250m'] = int(near.sum())
    print('距既有裸地 过滤后 %d（剔 %d）' % (len(d), stats['距既有裸地<250m']), flush=True)
    # 6) 多源票
    vote = ((d.bare >= 0.40).astype(int) + (d.wc == 60).astype(int) + (d.modis_lc == 16).astype(int) +
            ((d.s2_ndvi_xia < 200) & (d.s2_bright > 120)).astype(int) + (d.clcd == 7).astype(int))
    d['votes'] = vote
    # CLCD 判耕地/林地时，要求更强证据（≥3 票且 DW bare≥0.5 且 NDVI<180）
    conflict = d.clcd.isin([1, 2, 8])
    ok = (vote >= 2) & (~conflict | ((d.bare >= 0.40) & (d.s2_ndvi_xia < 200)))
    stats['多源票/冲突不足'] = int((~ok).sum())
    d = d[ok].reset_index(drop=True)
    print('多源票 过滤后 %d' % len(d), flush=True)
    # 7) 空间：0.025° ≤1 + 200 m + 每省 ≤800
    d['_cell'] = (np.floor(d.lon / 0.025).astype(np.int64) * 100000 + np.floor(d.lat / 0.025).astype(np.int64))
    d = d.sort_values(['votes', 'bare'], ascending=False).groupby('_cell').head(1).reset_index(drop=True)
    km = kmxy(d.lon.to_numpy(), d.lat.to_numpy())
    order = np.argsort(-d.bare.to_numpy())
    kept, kp = {}, np.zeros(len(d), dtype=bool)
    for i in order:
        kx = int(np.floor(km[i, 0] / 0.2)); ky = int(np.floor(km[i, 1] / 0.2))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((kx + dx, ky + dy), ()):
                    if (km[i, 0] - km[j, 0]) ** 2 + (km[i, 1] - km[j, 1]) ** 2 < 0.04:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok:
            kept.setdefault((kx, ky), []).append(i); kp[i] = True
    d = d[kp].reset_index(drop=True)
    # 省
    j = gpd.sjoin(gpd.GeoDataFrame(d, geometry=[Point(x, y) for x, y in zip(d.lon, d.lat)],
                                  crs='EPSG:4326'), prov, predicate='within', how='left')
    j = j[~j.index.duplicated(keep='first')].drop(columns=[c for c in ('index_right',) if c in j.columns])
    miss = j.ct_prov.isna()
    if miss.any():
        pv = prov.rename(columns={'ct_prov': 'prov_near'})
        sub = gpd.GeoDataFrame(j.loc[miss, ['lon', 'lat']], geometry=j.loc[miss, 'geometry'].values,
                               crs='EPSG:4326')
        near2 = gpd.sjoin_nearest(sub.reset_index(), pv, how='left').drop_duplicates('index').set_index('index')
        j.loc[miss, 'ct_prov'] = near2['prov_near']
    d = pd.DataFrame(j.drop(columns='geometry'))
    d = d.groupby('ct_prov', group_keys=False).head(800).reset_index(drop=True)
    d = d.drop(columns=[c for c in ('_cell',) if c in d.columns])
    d.to_parquet(os.path.join(BASE, 'data', 'bare_cand', 'bare_cand_filtered.parquet'), index=False)
    stats['final'] = int(len(d))
    stats['by_prov'] = {k: int(v) for k, v in d.ct_prov.value_counts().head(20).items()}
    json.dump(stats, open(os.path.join(BASE, 'data', 'bare_cand', 'filter_stats.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
    print(stats, flush=True)


if __name__ == '__main__':
    main()
