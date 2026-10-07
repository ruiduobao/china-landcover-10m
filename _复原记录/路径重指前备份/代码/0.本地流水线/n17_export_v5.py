# -*- coding: utf-8 -*-
"""
n17_export_v5.py — v5 清洗样本 → by_class PNG+GPKG / by_province GPKG / 全量 GPKG
* 输入: cn_samples_v5.parquet (2,738,187 点)
* 输出:
    数据/样本交付/cn_samples_v5.gpkg
    数据/样本交付/by_class/class_XX_中文名.{png,gpkg}
    数据/样本交付/by_province/{省名}.gpkg
"""
import os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_conf import CLASSES

V5 = '数据/本地处理/样本底座/cn_samples_v5.parquet'
PROV_SHP = r'Z:\Mywork\论文\中国人口密度2000-2026\1.数据\1.1_CTAMAP\2026\省级\T2026年初省级.shp'
ROOT = '数据/样本交付'
BY_CLASS = os.path.join(ROOT, 'by_class')
BY_PROV = os.path.join(ROOT, 'by_province')
os.makedirs(BY_CLASS, exist_ok=True)
os.makedirs(BY_PROV, exist_ok=True)

TIER_LABEL = {'gold': '金标', 'silver': '银标', 'bronze': '铜标',
              'uncovered': '覆盖外', 'external_pending': '待清洗',
              'external_c': '外部已清洗', 'validation': '验证'}

def main():
    t0 = time.time()
    df = pd.read_parquet(V5)
    print(f'v5 点数: {len(df):,}', flush=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    prov = gpd.read_file(PROV_SHP)[['省', 'geometry']]
    prov = prov.to_crs(4326) if prov.crs.to_epsg() != 4326 else prov

    # ---- 省归属 ----
    print('省份归属…', flush=True)
    import shapely
    polys = []
    for _, r in prov.iterrows():
        sg = r.geometry
        polys += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    from shapely import contains_xy
    xs = df.lon.to_numpy(); ys = df.lat.to_numpy()
    prov_arr = np.full(len(df), '', dtype=object)
    for _, r in prov.iterrows():
        poly = r.geometry
        m = (xs >= poly.bounds[0]) & (xs <= poly.bounds[2]) & \
            (ys >= poly.bounds[1]) & (ys <= poly.bounds[3])
        if m.any():
            inside = contains_xy(poly, xs[m], ys[m])
            prov_arr[np.flatnonzero(m)[inside]] = r['省']
    gdf['省'] = prov_arr
    gdf.to_file(os.path.join(ROOT, 'cn_samples_v5.gpkg'), layer='cn_samples_v5', driver='GPKG')
    print(f'全量 GPKG: {os.path.join(ROOT, "cn_samples_v5.gpkg")}', flush=True)

    # ---- by_class ----
    print('\n=== by_class ===', flush=True)
    prov_boundary = prov.boundary
    classes = sorted(df.class_new.unique())
    for i, cls in enumerate(classes):
        sub = df[df.class_new == cls]
        en, cn, _, _ = CLASSES.get(int(cls), ('?', f'类{cls}', '?', '?'))
        # gpkg
        gf = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub.lon, sub.lat), crs=4326)
        gpf = os.path.join(BY_CLASS, f'class_{cls}_{cn}.gpkg')
        gf.to_file(gpf, layer='samples', driver='GPKG')
        # png
        fig, ax = plt.subplots(figsize=(11, 9))
        prov.boundary.plot(ax=ax, color='#444444', linewidth=0.6)
        ax.scatter(sub.lon, sub.lat, s=0.4, c='#c0392b', alpha=0.35,
                   rasterized=True, edgecolors='none')
        ax.set_title(f'{cls} {cn}（{en}）清洗后样本点分布  N={len(sub):,}',
                     fontsize=14, pad=12)
        ax.set_xlabel('经度 (°E)'); ax.set_ylabel('纬度 (°N)')
        ax.set_xlim(73, 137); ax.set_ylim(16, 55)
        ax.set_aspect(1.25)
        fig.text(0.99, 0.01, f'CNLC10 v5 清洗后 · {time.strftime("%Y-%m-%d")}',
                 ha='right', fontsize=8, color='#666666')
        png = os.path.join(BY_CLASS, f'class_{cls}_{cn}.png')
        fig.savefig(png, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f'  [{i+1}/{len(classes)}] {cls} {cn}: {len(sub):,}', flush=True)

    # ---- by_province ----
    print('\n=== by_province ===', flush=True)
    for _, r in prov.iterrows():
        pname = r['省']
        poly = r.geometry
        m = (xs >= poly.bounds[0]) & (xs <= poly.bounds[2]) & \
            (ys >= poly.bounds[1]) & (ys <= poly.bounds[3])
        if not m.any():
            continue
        from shapely import contains_xy
        inside = contains_xy(poly, xs[m], ys[m])
        sub = df[m][inside]
        if not len(sub):
            continue
        fp = os.path.join(BY_PROV, f'{pname}.gpkg')
        g = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub.lon, sub.lat), crs=4326)
        g.to_file(fp, layer='samples', driver='GPKG')
        print(f'  {pname}: {len(sub):,}', flush=True)

    print(f'\n完成，用时 {(time.time()-t0)/60:.1f} 分钟', flush=True)

if __name__ == '__main__':
    main()
