# -*- coding: utf-8 -*-
"""m3_mangrove_assemble.py — 红树林扩充 第3步：候选汇总、去重、避让原样本（本地）

* 输入：data/mangrove/cand/parts/p*.parquet（m2 产物）
* 规则：
  1) 按 100 m 格去重（round(4) 级，取每格 1 点）；
  2) 与"原 652 点"最近距离 < 150 m 的候选剔除（避免与既有样本近重复）；
  3) 与"原 652 点"重合的（<30 m）优先剔除。
* 输出：data/mangrove/cand/candidates.parquet（lon/lat/cid）+ stats_candidates.json
* 用法：python m3_mangrove_assemble.py
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/红树林'


def emit(m):
    print('[m3] ' + str(m), flush=True)


def main():
    fs = sorted(glob.glob(os.path.join(WORK, 'cand', 'parts', 'p*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    emit('原始候选 %d 点（%d 片）' % (len(d), len(fs)))
    # 100m 格去重
    d['cx'] = (d.lon / 0.001).round().astype(int)
    d['cy'] = (d.lat / 0.001).round().astype(int)
    d = d.drop_duplicates(['cx', 'cy']).reset_index(drop=True)
    emit('100m 格去重后 %d' % len(d))
    # 中国范围过滤（CTAmap 省界 + 5km 缓冲——红树林多落在简化海岸线之外）
    import geopandas as gpd
    prov = gpd.read_file(r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp')
    chinageom = prov.geometry.union_all().buffer(0.05)
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs='EPSG:4326')
    inchn = g.within(chinageom)
    emit('中国范围过滤：%d → %d（剔除境外 %d）' % (len(d), int(inchn.sum()), int((~inchn).sum())))
    d = d[inchn.to_numpy()].reset_index(drop=True)
    # 原 652
    import shapefile
    r = shapefile.Reader(os.path.join(DEST, '2023', '红树林_2023_WGS84.shp'), encoding='utf-8')
    o = pd.DataFrame([p.points[0] for p in r.iterShapes()], columns=['lon', 'lat'])
    r.close()
    emit('原样本 %d 点' % len(o))
    # 距离过滤（近似平面：1°lon≈105km（20°N）、1°lat≈111km）
    from scipy.spatial import cKDTree
    K = np.column_stack([o.lon * 105.0, o.lat * 111.0])
    tree = cKDTree(K)
    C = np.column_stack([d.lon * 105.0, d.lat * 111.0])
    dist, _ = tree.query(C, k=1)
    keep = dist >= 0.15
    emit('避让原样本 150m 后 %d（剔除 %d）' % (int(keep.sum()), int((~keep).sum())))
    d = d[keep].reset_index(drop=True)
    d['cid'] = ['MG%06d' % i for i in range(len(d))]
    out = d[['cid', 'lon', 'lat']]
    out.to_parquet(os.path.join(WORK, 'cand', 'candidates.parquet'), index=False)
    json.dump(dict(n_raw=int(len(pd.concat([pd.read_parquet(f) for f in fs]))), n_after_dedup=len(out),
                   dropped_near_orig=int((~keep).sum()), n_orig=len(o)),
              open(os.path.join(WORK, 'cand', 'stats_candidates.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ candidates.parquet（%d 点）' % len(out))
    # 密度预览（与原样本对照）
    g = d.groupby([(d.lon * 100).astype(int), (d.lat * 100).astype(int)]).size()
    print('1km 格：候选 %d 格 / 点均 %.1f；原样本 635 格/点均 1.03' % (len(g), len(d) / len(g)))


if __name__ == '__main__':
    main()
