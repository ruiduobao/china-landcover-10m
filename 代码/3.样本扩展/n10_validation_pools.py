# -*- coding: utf-8 -*-
"""
n10_validation_pools.py — 三个独立验证池入库（不进训练）
* mountains: 全球高山 LC 验证 55,758 点（CCI 码 + Conf2020 质量列，筛 Conf>=2）
* yrd:       长三角不透水验证 3,845 点（name 1=不透水 → GUB 拆 190/200）
* hdlv:      新疆 19,503 点（已映射）
* 输出: 数据/本地处理/样本底座/validation_pools.parquet
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EXT = '数据/外部样本源'
GUB = r'E:\data\非洲城市发展和驱动力分析\数据\GUB_全球城市边界数据\GUB_Global_2020\GUB_Global_2020.shp'
OUT = '数据/本地处理/样本底座/validation_pools.parquet'

# CCI 码 → 我们 30 类（亚类歧义处 conf 降 0.5）
CCI_MAP = {10: 10, 20: 10, 50: 51, 60: 61, 70: 71, 80: 81, 90: 91,
           120: 121, 130: 130, 140: 140, 150: 150, 180: 181,
           190: 190, 200: 201, 210: 202, 220: 220}
CCI_CONF = {50: .5, 60: .5, 70: .5, 80: .5, 90: .5, 180: .6}

def gub_split(df):
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    gub = gpd.read_file(GUB)
    geoms = list(gub.geometry)
    tree = STRtree(geoms)
    urban = np.zeros(len(df), dtype=bool)
    pts = [Point(x, y) for x, y in zip(df.lon, df.lat)]
    for i, pt in enumerate(pts):
        for j in tree.query(pt):
            if geoms[j].contains(pt):
                urban[i] = True
                break
    return urban

def main():
    pools = []
    # 1) mountains
    mf = glob.glob(os.path.join(EXT, 'mountains_lc', '**', '*.shp'), recursive=True)
    if mf:
        import geopandas as gpd
        g = gpd.read_file(mf[0]).to_crs(4326) if gpd.read_file(mf[0]).crs.to_epsg() != 4326 else gpd.read_file(mf[0])
        g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
        m = g[(g['Conf2020'] >= 2) & (g['LandCover2'] > 0)].copy()
        m['class_new'] = m['LandCover2'].astype(int).map(CCI_MAP).fillna(-1).astype(int)
        m = m[m.class_new > 0]
        m['src'] = 'mountains_lc_validation'
        m['src_conf'] = m['LandCover2'].astype(int).map(lambda c: CCI_CONF.get(c, 0.75))
        m['year'] = 2020
        m['tier'] = 'validation'
        pools.append(m[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'tier']])
        print('mountains:', len(m))
    # 2) YRD
    yf = glob.glob(os.path.join(EXT, 'yrd_impervious', '*.shp'))
    if yf:
        import geopandas as gpd
        g = gpd.read_file(yf[0])
        g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
        imp = g[g['name'] == 1].copy()
        if len(imp):
            urban = gub_split(imp)
            imp['class_new'] = np.where(urban, 190, 200).astype(int)
            non = g[g['name'] == 0].copy()
            non['class_new'] = -1   # 非不透水参考（无类标签）
            both = pd.concat([imp, non], ignore_index=True)
            both['src'] = 'yrd_imperv_1985_2020'
            both['src_conf'] = 0.85
            both['year'] = 2020
            both['tier'] = 'validation'
            pools.append(both[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'tier']])
            print('YRD:', len(both), '(不透水', len(imp), ')')
    # 3) HDLV-XJ
    hf = os.path.join(EXT, 'hdlv_xj_validation.parquet')
    if os.path.exists(hf):
        h = pd.read_parquet(hf)
        if 'src_conf' not in h.columns:
            h['src_conf'] = 0.8
        if 'year' not in h.columns:
            h['year'] = 2020
        h['tier'] = 'validation'
        pools.append(h[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'tier']])
        print('HDLV-XJ:', len(h))
    if pools:
        df = pd.concat(pools, ignore_index=True)
        df.to_parquet(OUT, index=False)
        print('验证池合计:', len(df), '->', OUT)
        print(df.src.value_counts())

if __name__ == '__main__':
    main()

