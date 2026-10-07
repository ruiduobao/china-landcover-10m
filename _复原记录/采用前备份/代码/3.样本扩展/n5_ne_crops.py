# -*- coding: utf-8 -*-
"""
n5_ne_crops.py — 东北主要作物高置信样本（Zenodo 18446149）入库
* 只抽取 7z 内 Major-crop Samples 子树（跳过大体积 Maps）
* Rice→12, Maize→10, Soybean→10；字段含 year/prov/crop
* 输出: 数据/外部样本源/ne_crops_samples.parquet
"""
import os, subprocess, glob
import numpy as np
import pandas as pd
import geopandas as gpd

SZ = r'数据/外部样本源/ne_china_crops.7z'
SEVENZ = r'C:\Program Files\7-Zip\7z.exe'
EXT_DIR = r'数据/外部样本源/ne_crops_samples'
OUT = '数据/外部样本源/ne_crops_samples.parquet'

CROP_MAP = {'rice': 12, 'maize': 10, 'soybean': 10}

def extract_samples():
    os.makedirs(EXT_DIR, exist_ok=True)
    if glob.glob(os.path.join(EXT_DIR, '**', '*.shp'), recursive=True):
        print('已解压，跳过')
        return
    r = subprocess.run([SEVENZ, 'x', SZ, f'-o{EXT_DIR}',
                        r'2017-2025 Samples and Maps of major-crop\Major-crop Samples\*',
                        '-r', '-y'], capture_output=True, text=True)
    print('7z rc=', r.returncode)

def load_all():
    frames = []
    for fp in glob.glob(os.path.join(EXT_DIR, '**', '*.shp'), recursive=True):
        base = os.path.basename(fp)
        # High_Con_Maize_Samples_HLJ_2017.shp
        crop = next((c for c in CROP_MAP if c.lower() in base.lower()), None)
        if not crop:
            continue
        prov = base.split('_')[-2]
        year = int(base.split('_')[-1].replace('.shp', ''))
        try:
            g = gpd.read_file(fp)
        except Exception as e:
            print('ERR', base, str(e)[:80]); continue
        g = g.to_crs(4326) if g.crs and g.crs.to_epsg() != 4326 else g
        if not (g.geometry.geom_type == 'Point').all():
            g['geometry'] = g.geometry.representative_point()
        df = pd.DataFrame({'lon': g.geometry.x.to_numpy(),
                           'lat': g.geometry.y.to_numpy(),
                           'class_new': CROP_MAP[crop],
                           'crop': crop, 'year': year, 'prov': prov,
                           'src': f'ne_crops_{crop}_{year}_{prov}',
                           'src_conf': 0.9})
        frames.append(df)
    alld = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    alld.to_parquet(OUT, index=False)
    print('东北作物样本合计:', len(alld))
    print('按作物:', dict(alld.crop.value_counts()))
    print('按年:', dict(alld.year.value_counts().head(9)))

if __name__ == '__main__':
    extract_samples()
    load_all()
