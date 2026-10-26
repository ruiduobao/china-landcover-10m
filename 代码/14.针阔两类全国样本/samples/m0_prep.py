# -*- coding: utf-8 -*-
"""m0_prep.py — 多方法针阔样本探索：基础表拼装
* 输入：F:/lc_work/nb_sep_2026/labels/forest_4class_hiconf.parquet（466,135 林点，62 列）
        F:/lc_work/v31_exp/data/forest_qc/out/forest_samples_v1.parquet（补 modis_lc / dw_* / aef_p_ever）
        F:/lc_work/年度子集_含稀有类/r7_train_{2017..2024}.parquet（AEF64 + class_new）
* 规则源：无（只拼表，不改任何标签）
* 输出：labels/methods/work_base.parquet
* 用法：python m0_prep.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
HICONF = r'F:/lc_work/nb_sep_2026/labels/forest_4class_hiconf.parquet'
FSV1 = r'F:/lc_work/v31_exp/data/forest_qc/out/forest_samples_v1.parquet'

os.makedirs(WORK, exist_ok=True)

h = pd.read_parquet(HICONF)
h['stable_uid'] = h['stable_uid'].astype(str)
print('hiconf', h.shape)

f = pd.read_parquet(FSV1, columns=['point_id', 'modis_lc', 'dw_shrub', 'dw_crops',
                                   'dw_grass', 'aef_p_ever', 'n_support',
                                   'n_refute', 'n_decisive'])
f['stable_uid'] = f['point_id'].astype(str)
f = f.drop(columns=['point_id'])
print('fsv1', f.shape)

df = h.merge(f, on='stable_uid', how='left', validate='one_to_one')
assert len(df) == len(h), (len(df), len(h))

# --- 标准化叶型/物候轴为 -1/0/1 便于投票
LEAF = {'N': 1, 'B': 0, '': np.nan, '?': np.nan}
df['leaf_label_s'] = df['leaf_label'].map(LEAF)
df['leaf_modis_s'] = df['leaf_modis'].map(LEAF)
df['leaf_fcs10_s'] = df['leaf_fcs10'].map(LEAF)
PHENO = {'E': 1, 'D': 0, '': np.nan, '?': np.nan}
df['pheno_2019_s'] = df['v2019'].map(PHENO)
df['pheno_2023_s'] = df['v2023'].map(PHENO)
df['pheno_modis_s'] = df['vmodis'].map(PHENO)
df['pheno_label_s'] = np.where(df['cl_24'].isin([4, 6]), 1,
                               np.where(df['cl_24'].isin([5, 7]), 0, np.nan))

# --- 冠层结构本地可得量
df['tcd_hansen'] = np.nan          # 走后 m3 GEE 填
df['cgls_ft'] = np.nan
df['cgls_tcd'] = np.nan
df['cgls_year'] = np.nan
df['gedi_rh98'] = np.nan
df['gedi_rh50'] = np.nan
df['gedi_cover'] = np.nan

# --- 高程带（MH 分层用）
df['elev_band'] = pd.cut(df['dem_nasadem'], [-1, 300, 800, 1500, 2500, 4000, 9999],
                         labels=['e0', 'e1', 'e2', 'e3', 'e4', 'e5'])
df['stratum'] = df['tile'].astype(str) + '|' + df['elev_band'].astype(str)

# --- 地理气候区（D 方法用，粗粒度大区）
def region(r):
    lon, lat, prov = r['lon'], r['lat'], str(r['prov_cta'])
    if lat >= 42:
        return 'NE'          # 东北/内蒙东
    if 32 <= lat < 42 and lon < 105:
        return 'NW'          # 西北/华北北部
    if 32 <= lat < 42 and lon >= 105:
        return 'NC'          # 华北
    if 24 <= lat < 32:
        return 'SC'          # 华中/华东/西南
    return 'TR'              # 华南/热带
df['region'] = df.apply(region, axis=1)

out = os.path.join(WORK, 'work_base.parquet')
df.to_parquet(out, index=False)
print('WROTE', out, df.shape)
print(df.groupby('cl_24').size().to_string())
print('leaf source coverage:')
for c in ['leaf_label_s', 'leaf_modis_s', 'leaf_fcs10_s']:
    print(' ', c, int(df[c].notna().sum()), round(df[c].notna().mean() * 100, 1), '%')
print('region', df.region.value_counts().to_dict())
