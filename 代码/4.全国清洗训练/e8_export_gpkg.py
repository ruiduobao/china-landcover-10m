# -*- coding: utf-8 -*-
"""
e8_export_gpkg.py — r7 版样本点 parquet → GeoPackage（保留全部属性）
* 输出目录: 数据/样本交付/gpkg/<原名>.gpkg，每层一个 parquet，几何=Point(lon,lat) EPSG:4326
* 已存在且行数一致则跳过（可断点续跑）；inf → NULL（GDAL 不接受 inf 属性值）
* 用法:
    python e8_export_gpkg.py --list            # 列出任务与体量
    python e8_export_gpkg.py --job <key>       # 只跑一个
    python e8_export_gpkg.py --all             # 全部（建议后台）
"""
import os, sys, glob, json, time, argparse
import numpy as np
import pandas as pd
import pyogrio

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
BASE = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练'
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
PEDE = os.path.join(PROJ, '数据/本地处理/样本底座')
OUT = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/样本交付/gpkg'
os.makedirs(OUT, exist_ok=True)

YEARS = list(range(2017, 2025))


def build_jobs():
    jobs = {}
    for y in YEARS:
        jobs[f'yearly_rare_{y}'] = (os.path.join(BASE, '年度子集_含稀有类', f'r7_train_{y}.parquet'),
                                    f'r7_train_yearly_with_rare_{y}')
        jobs[f'yearly_{y}'] = (os.path.join(BASE, '年度子集', f'r7_train_{y}.parquet'),
                               f'r7_train_yearly_{y}')
    jobs['sample_year_qc'] = (os.path.join(BASE, 'sample_year_qc.parquet'), 'sample_year_qc')
    jobs['rare_samples'] = (os.path.join(BASE, '稀有类补样/r7_rare_samples.parquet'), 'r7_rare_samples')
    jobs['train'] = (os.path.join(WORKR, 'r7_train.parquet'), 'r7_train')
    jobs['train_validity'] = (os.path.join(WORKR, 'r7_train_validity.parquet'), 'r7_train_validity')
    jobs['pool'] = (os.path.join(WORKR, 'r7_pool.parquet'), 'r7_pool')
    jobs['validation_v2'] = (os.path.join(PEDE, 'validation_pool_v2.parquet'), 'validation_pool_v2')
    return jobs


def clean(df):
    """统一 dtype：category→str、object→str、inf→NaN"""
    for c in df.columns:
        if str(df[c].dtype) == 'category':
            df[c] = df[c].astype(str)
        elif df[c].dtype == object:
            df[c] = df[c].astype(str)
    num = df.select_dtypes(include=[np.floating]).columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df


def run_one(key, src, layer):
    out = os.path.join(OUT, f'{layer}.gpkg')
    t0 = time.time()
    if not os.path.isfile(src):
        print(f'[{key}] 源缺失 {src}', flush=True); return
    df = pd.read_parquet(src)
    n = len(df)
    if not {'lon', 'lat'} <= set(df.columns):
        print(f'[{key}] 无 lon/lat，跳过', flush=True); return
    if os.path.isfile(out):
        try:
            info = pyogrio.read_info(out, layer=layer)
            if info.get('features') == n:
                print(f'[{key}] 已存在且行数一致（{n:,}），跳过', flush=True); return
        except Exception:
            pass
    df = clean(df)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs='EPSG:4326')
    if os.path.isfile(out):
        os.remove(out)
    pyogrio.write_dataframe(gdf, out, layer=layer, driver='GPKG')
    sz = os.path.getsize(out) / 1e9
    print(f'[{key}] {n:,} 点 × {len(df.columns)} 属性 → {out} '
          f'({sz:.2f} GB, {time.time()-t0:.0f}s)', flush=True)
    del df, gdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    jobs = build_jobs()
    if a.list:
        for k, (src, layer) in jobs.items():
            sz = os.path.getsize(src) / 1e6 if os.path.isfile(src) else -1
            print(f'{k:20s} {sz:9.1f} MB  ->  {layer}.gpkg   {src}')
        return
    todo = list(jobs) if a.all else ([a.job] if a.job else [])
    if not todo:
        print('用 --list / --job / --all'); return
    for k in todo:
        src, layer = jobs[k]
        try:
            run_one(k, src, layer)
        except Exception as e:
            print(f'[{k}] ERR {str(e)[:150]}', flush=True)


if __name__ == '__main__':
    main()
