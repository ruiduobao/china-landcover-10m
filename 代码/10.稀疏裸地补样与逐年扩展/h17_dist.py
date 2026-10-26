# -*- coding: utf-8 -*-
"""h17_dist.py — 逐年补样的省级分布表（支撑"分布"验收）

* 输入：data/yearly/topup_<year>.parquet；CTAmap 省界
* 输出：data/yearly/province_yearly.csv（年×省×类）+ reports/distribution.md
* 用法：python h17_dist.py
"""
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
YEARS = list(range(2017, 2025))


def main():
    prov = gpd.read_file(PROV_SHP)[['省', 'geometry']].rename(columns={'省': 'ct_prov'})
    prov['geometry'] = prov.geometry.buffer(0)
    rows = []
    for y in YEARS:
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        if not os.path.isfile(fp):
            continue
        d = pd.read_parquet(fp, columns=['lon', 'lat', 'cl'])
        g = gpd.GeoDataFrame(d, geometry=[Point(x, yy) for x, yy in zip(d.lon, d.lat)], crs='EPSG:4326')
        j = gpd.sjoin(g, prov, predicate='within', how='left')
        j = j[~j.index.duplicated(keep='first')]
        j = j.drop(columns=[c for c in ('index_right',) if c in j.columns])
        miss = j.ct_prov.isna()
        if miss.any():
            pv = prov.rename(columns={'ct_prov': 'prov_near'})
            near = gpd.sjoin_nearest(j[miss].reset_index(), pv, how='left').drop_duplicates('index').set_index('index')
            j.loc[miss, 'ct_prov'] = near['prov_near']
        for (p, cl), n in j.groupby(['ct_prov', 'cl']).size().items():
            rows.append({'year': y, 'ct_prov': p, 'class': 'sparse' if cl == 13 else 'bare', 'n': int(n)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BASE, 'data', 'yearly', 'province_yearly.csv'), index=False, encoding='utf-8-sig')
    piv = out.pivot_table(index='ct_prov', columns=['year', 'class'], values='n', aggfunc='sum').fillna(0).astype(int)
    # 只保留任一年有样本的省
    piv = piv[piv.sum(axis=1) > 0]
    print(piv.to_string())
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)
    md = ['# 逐年补样省级分布（点数）\n', piv.to_markdown()]
    open(os.path.join(BASE, 'reports', 'distribution.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('→ data/yearly/province_yearly.csv, reports/distribution.md')


if __name__ == '__main__':
    main()
