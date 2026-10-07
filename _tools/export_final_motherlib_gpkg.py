# -*- coding: utf-8 -*-
"""把重建的终版母库导出为 GPKG（复用 e8_export_gpkg.py 的 clean 规范化）。"""
import os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
MOTHER = os.path.join(ROOT, r'数据\本地处理\样本重建')
OUT = os.path.join(ROOT, r'数据\样本交付\_终版母库_20260910')
os.makedirs(OUT, exist_ok=True)


def clean(df):
    for c in df.columns:
        if str(df[c].dtype) == 'category':
            df[c] = df[c].astype(str)
        elif df[c].dtype == object:
            df[c] = df[c].astype(str)
    num = df.select_dtypes(include=[np.floating]).columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df


for name in ['r7_train', 'r7_train_validity']:
    src = os.path.join(MOTHER, f'{name}.parquet')
    out = os.path.join(OUT, f'{name}.gpkg')
    t0 = time.time()
    df = clean(pd.read_parquet(src))
    n, ncol = len(df), len(df.columns)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
    if os.path.isfile(out):
        os.remove(out)
    pyogrio.write_dataframe(gdf, out, layer=name, driver='GPKG')
    print(f'{name}: {n:,} 点 × {ncol} 列 → {out} '
          f'({os.path.getsize(out)/1e9:.2f} GB, {time.time()-t0:.0f}s)', flush=True)
    del df, gdf
