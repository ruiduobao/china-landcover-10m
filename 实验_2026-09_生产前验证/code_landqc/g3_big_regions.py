# -*- coding: utf-8 -*-
"""g3_big_regions.py — 由 2° 生产瓦片（T%02d%02d）拼 3×3 得 6°×6° 大区矢量

* 输入：代码/8.全国生产/plan/tiles.json（313 瓦：ix/iy/x0/y0/x1/y1/land_frac/n_points）
* 规则：ix∈[3gx,3gx+2]、iy∈[3gy,3gy+2] 归并为大区 B<gx><gy>（矩形 lon 73+6gx~+6、lat 18+6gy~+6）；
        仅保留至少含一个有效瓦片（land_frac≥0.02 或 n_points>0）的大区；附：瓦片数、母库点合计、
        相交省（CTAmap 省界，按相交面积给主省 prov_main + 全部 provs）。
* 输出：AEF整理与可分离度/大区矢量_3x3瓦片.gpkg（层 big_regions_3x3）+ 同名 .geojson + 大区清单.csv
* 用法：python g3_big_regions.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

sys.stdout.reconfigure(encoding='utf-8')
PLAN = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.json'
CTAMAP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
OUT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/AEF整理与可分离度'
X0, Y0, D = 73.0, 18.0, 2.0


def main():
    plan = json.load(open(PLAN, encoding='utf-8'))
    t = pd.DataFrame(plan['tiles'])
    t['gx'] = t.ix // 3
    t['gy'] = t.iy // 3
    rows = []
    for (gx, gy), g in t.groupby(['gx', 'gy']):
        rows.append(dict(region='B%02d%02d' % (gx, gy), gx=int(gx), gy=int(gy),
                         lon0=X0 + 6 * gx, lat0=Y0 + 6 * gy, lon1=X0 + 6 * gx + 6, lat1=Y0 + 6 * gy + 6,
                         n_tiles=int(len(g)), tiles='|'.join(sorted(g.tile)),
                         land_frac_mean=round(float(g.land_frac.mean()), 3),
                         mother_points=int(g.n_points.sum())))
    d = pd.DataFrame(rows).sort_values(['gy', 'gx']).reset_index(drop=True)
    geom = [box(r.lon0, r.lat0, r.lon1, r.lat1) for r in d.itertuples()]
    gdf = gpd.GeoDataFrame(d, geometry=geom, crs='EPSG:4326')
    # 相交省（CTAmap）
    prov = gpd.read_file(CTAMAP)[['省', 'geometry']]
    j = gpd.sjoin(gdf[['region', 'geometry']], prov, predicate='intersects', how='left')
    j['inter'] = [gdf.loc[i, 'geometry'].intersection(prov.loc[k, 'geometry']).area
                  if isinstance(k, (int, np.integer)) else 0.0
                  for i, k in zip(j.index, j.index_right.fillna(-1).astype(int))]
    provs, pmain = {}, {}
    for reg, g in j.groupby('region'):
        gg = g.sort_values('inter', ascending=False)
        provs[reg] = '|'.join(gg['省'].dropna().tolist())
        pmain[reg] = gg['省'].dropna().iloc[0] if len(gg['省'].dropna()) else ''
    gdf['provs'] = gdf.region.map(provs)
    gdf['prov_main'] = gdf.region.map(pmain)
    os.makedirs(OUT, exist_ok=True)
    gpkg = os.path.join(OUT, '大区矢量_3x3瓦片.gpkg')
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gdf.to_file(gpkg, driver='GPKG', layer='big_regions_3x3', engine='pyogrio', SPATIAL_INDEX='NO')
    gdf.to_file(os.path.join(OUT, '大区矢量_3x3瓦片.geojson'), driver='GeoJSON', engine='pyogrio')
    gdf.drop(columns='geometry').to_csv(os.path.join(OUT, '大区清单.csv'), index=False, encoding='utf-8-sig')
    print('大区 %d 个（含有效瓦片）｜ 覆盖 lon %.0f~%.0f lat %.0f~%.0f' % (
        len(gdf), gdf.lon0.min(), gdf.lon1.max(), gdf.lat0.min(), gdf.lat1.max()))
    print(gdf[['region', 'n_tiles', 'mother_points', 'prov_main']].head(8).to_string(index=False))
    print('→', gpkg)


if __name__ == '__main__':
    main()
