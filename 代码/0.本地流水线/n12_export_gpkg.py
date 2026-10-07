# -*- coding: utf-8 -*-
"""
n12_export_gpkg.py — v4 样本底座 → GPKG 交付（单文件 + 34 省分文件）
* 输入: cn_samples_v4.parquet (10,223,338 点) + CTAMAP 2026 省级 shp
* 输出:
    数据/样本交付/cn_samples_v4.gpkg          （全量单文件）
    数据/样本交付/by_province/{省名}.gpkg     （34 个分文件，空省跳过）
* 空间归属: geopandas sjoin（within），未匹配点保留在单文件并标注 省=未匹配
"""
import os, sys, time, glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
PROV_SHP = r'E:\地理所\论文\中国人口密度2000-2026\1.数据\1.1_CTAMAP\2026\省级\T2026年初省级.shp'
OUT_DIR = '数据/样本交付'
os.makedirs(os.path.join(OUT_DIR, 'by_province'), exist_ok=True)

def main():
    t0 = time.time()
    print('读取 v4…', flush=True)
    df = pd.read_parquet(V4, columns=['lon', 'lat', 'class_new', 'tier', 'year',
                                      'src', 'src_conf'])
    print('点数:', len(df), flush=True)
    prov = gpd.read_file(PROV_SHP)[['省', 'geometry']]
    prov = prov.to_crs(4326) if prov.crs.to_epsg() != 4326 else prov

    # 分块 sjoin 防内存溢出（每块 200 万）
    labels = np.full(len(df), '未匹配', dtype=object)
    xs = df['lon'].to_numpy(); ys = df['lat'].to_numpy()
    CH = 2_000_000
    for s in range(0, len(df), CH):
        e = min(s + CH, len(df))
        pts = gpd.GeoDataFrame(
            {'tmp': np.arange(e - s)},
            geometry=gpd.points_from_xy(xs[s:e], ys[s:e]), crs=4326)
        j = gpd.sjoin(pts, prov[['省', 'geometry']], predicate='within', how='left')
        lab = j['省'].to_numpy()
        lab = np.where(pd.isna(lab), '未匹配', lab)
        labels[s:e] = lab
        print(f'  sjoin {e:,}/{len(df)}', flush=True)
    df['省'] = labels
    print('省份分布:', df['省'].value_counts().to_dict(), flush=True)

    # GeoDataFrame（按行生成点，约 2-4 分钟）
    print('构造 GeoDataFrame…', flush=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    gdf = gdf.drop(columns=['lon', 'lat'])

    # 1) 全量单文件
    single = os.path.join(OUT_DIR, 'cn_samples_v4.gpkg')
    print('写全量 gpkg（约 5-15 分钟）…', flush=True)
    gdf.to_file(single, layer='cn_samples_v4', driver='GPKG')
    print('  ->', single, round(os.path.getsize(single)/1e9, 2), 'GB', flush=True)

    # 2) 34 省分文件
    for pname, sub in gdf.groupby('省'):
        safe = str(pname).replace('/', '_')
        fp = os.path.join(OUT_DIR, 'by_province', f'{safe}.gpkg')
        sub.to_file(fp, layer='samples', driver='GPKG')
        print(f'  {pname}: {len(sub):,} -> {fp}', flush=True)

    print(f'\n完成，用时 {(time.time()-t0)/60:.1f} 分钟', flush=True)

if __name__ == '__main__':
    main()
