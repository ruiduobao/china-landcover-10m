# -*- coding: utf-8 -*-
"""h6b_china_filter.py — 国界过滤（保留境内 + 距国界 ≤10 km 的边界带；剔境外）

* 背景：FCS10 瓦片为矩形，覆盖蒙古/俄罗斯同类地表；交付点中 350 点（4.8%）落在 CTAmap 省界外，
  最远 242 km。项目样本口径为中国境内；边界带容差取 10 km（沿用项目"离岸点最远 9.0 km 仍保留"的先例）。
* 输入：data/final/cand_qc_fixed.parquet（status=='keep'）
* 输出：data/final/cand_qc_china.parquet（被剔点 status='china_drop'，其余 status='keep'）+ china_filter.json
* 用法：python h6b_china_filter.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.stdout.reconfigure(encoding='utf-8')
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
TOL_M = 10000.0
AEQ = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 +datum=WGS84 +units=m +no_defs')


def main():
    fp = r'F:/lc_work/bare_sparse_v3/data/final/cand_qc_fixed.parquet'
    d = pd.read_parquet(fp)
    if 'status' in d.columns:
        d = d[d.status == 'keep'].reset_index(drop=True)
    print('输入 %d' % len(d), flush=True)
    g = gpd.GeoDataFrame(d, geometry=[Point(x, y) for x, y in zip(d.lon, d.lat)], crs='EPSG:4326')
    prov = gpd.read_file(PROV_SHP)[['geometry']]
    china = prov.geometry.union_all()
    g['inside'] = g.within(china)
    ga = g.to_crs(AEQ)
    china_a = gpd.GeoSeries([china], crs='EPSG:4326').to_crs(AEQ).iloc[0]
    dist = np.zeros(len(g))
    out = ~g.inside.to_numpy()
    if out.any():
        dist[out] = ga.geometry[out].distance(china_a).to_numpy()
    keep = g.inside.to_numpy() | (dist <= TOL_M)
    st = {'n_in': int(len(g)), 'outside_boundary': int(out.sum()),
          'dropped_over_10km': int((~keep).sum()),
          'kept_border_band_10km': int((out & keep).sum()),
          'n_out': int(keep.sum()),
          'by_fam_kept': {k: int(v) for k, v in d.fam[keep].value_counts().items()},
          'max_outside_km': round(float(dist.max() / 1000.0), 1)}
    d['status'] = np.where(keep, 'keep', 'china_drop')
    d['dist_china_km'] = np.round(dist / 1000.0, 3)
    d.to_parquet(r'F:/lc_work/bare_sparse_v3/data/final/cand_qc_china.parquet', index=False)
    json.dump(st, open(r'F:/lc_work/bare_sparse_v3/data/final/china_filter.json', 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
    print(st, flush=True)


if __name__ == '__main__':
    main()
