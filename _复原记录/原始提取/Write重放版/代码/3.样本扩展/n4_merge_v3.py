# -*- coding: utf-8 -*-
"""
n4_merge_v3.py — 合并全部新样本源 → 样本底座 v3
* 输入:
    v2:   数据/本地处理/样本底座/cn_samples_v2.parquet (849万)
    新源: 数据/外部样本源/{worldcereal_china, grass_gpw_china}.parquet
          数据/外部样本源/gee_thematic/{wetland_gwl, imperv_gisa, orchard_glc12}.parquet
* 规则:
    - 去重: 距 v2 既有点 <100m 的新点丢弃；新点之间 <100m 只留一个
    - 新点 tier='external_pending'（待嵌入提取后做嵌入清洗再升级置信层）
    - 不透水新点: gisa_year<=2000 → 190 城镇, >2000 → 200 乡村
    - GWL 湿地码 -1 平移映射 (181沼泽→180木本 ... 187潮滩→186)，180永久水体丢弃
* 输出: 数据/本地处理/样本底座/cn_samples_v3.parquet + v3_summary.json
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = '数据/本地处理/样本底座/cn_samples_v3.parquet'
EXT = '数据/外部样本源'
DEG = np.array([111.0, 88.0])          # 度→近似公里归一化
R_KM = 0.1                              # 100m

v2 = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v2.parquet')
base_tree = cKDTree(v2[['lon', 'lat']].to_numpy() / DEG)

def dedupe(df):
    n0 = len(df)
    d, _ = base_tree.query(df[['lon', 'lat']].to_numpy() / DEG, k=1)
    df = df[d > R_KM].reset_index(drop=True)
    if len(df) > 1:
        t = cKDTree(df[['lon', 'lat']].to_numpy() / DEG)
        d2, _ = t.query(df[['lon', 'lat']].to_numpy() / DEG, k=2)
        df = df[d2[:, 1] > R_KM].reset_index(drop=True)
    return df, n0

def load(pattern):
    fs = glob.glob(os.path.join(EXT, '**', pattern), recursive=True)
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)

frames = []

# 1) WorldCereal
wc = load('worldcereal_china.parquet')
if wc is not None and len(wc):
    frames.append(pd.DataFrame({'lon': wc.lon, 'lat': wc.lat,
        'class_new': wc.our_class.astype(int),
        'src': 'worldcereal_' + wc.src, 'src_conf': 0.9, 'year': wc.get('year', 2018)}))

# 2) GPW 草地 (class 1 人工/2 天然 → 130)
gp = load('grass_gpw_china.parquet')
if gp is not None and len(gp):
    g = gp[gp['class'].isin([1, 2])]
    frames.append(pd.DataFrame({'lon': g.lon, 'lat': g.lat,
        'class_new': 130, 'src': 'gpw_grass_vhr', 'src_conf': 0.95, 'year': 2024}))

# 3) GEE 湿地
wl = load('wetland_gwl.parquet')
if wl is not None and len(wl):
    m = {181: 180, 182: 181, 183: 182, 184: 183, 185: 184, 186: 185, 187: 186}
    w = wl[wl.code.isin(m)].copy()
    frames.append(pd.DataFrame({'lon': w.lon, 'lat': w.lat,
        'class_new': w.code.map(m).astype(int),
        'src': 'gwl_fcs30_stable', 'src_conf': 0.9, 'year': 2022}))

# 4) GEE 不透水 (GISA 首城市化年)
im = load('imperv_gisa.parquet')
if im is not None and len(im):
    yr = pd.to_numeric(im.gisa_year, errors='coerce')
    frames.append(pd.DataFrame({'lon': im.lon, 'lat': im.lat,
        'class_new': np.where(yr <= 2000, 190, 200).astype(int),
        'src': 'gisa_gisd', 'src_conf': 0.75, 'year': 2021}))

# 5) GEE 园地 glc12
orc = load('orchard_glc12.parquet')
if orc is not None and len(orc):
    frames.append(pd.DataFrame({'lon': orc.lon, 'lat': orc.lat,
        'class_new': 11, 'src': 'glc12_2020_single', 'src_conf': 0.6, 'year': 2020}))

print('=== 各源原始量 ===')
for f in frames:
    print(f'  {f.src.iloc[0]}: {len(f):,}')

print('=== 去重(<100m) ===')
kept = []
for f in frames:
    f2, n0 = dedupe(f)
    kept.append(f2)
    print(f'  {f2.src.iloc[0] if len(f2) else f.src.iloc[0]}: {n0:,} -> {len(f2):,}')

new_all = pd.concat(kept, ignore_index=True)
tree = cKDTree(new_all[['lon', 'lat']].to_numpy() / DEG)
pairs = tree.query_pairs(R_KM)
drop = {b for _, b in pairs}
new_all = new_all.drop(index=list(drop)).reset_index(drop=True)
print('新点互去重后:', f'{len(new_all):,}')

new_all['tier'] = 'external_pending'
for c in ['src', 'src_conf']:
    if c not in v2.columns:
        v2[c] = None
v3 = pd.concat([v2, new_all.reindex(columns=v2.columns)], ignore_index=True)
v3.to_parquet(OUT, index=False)

summary = {'v2_total': len(v2), 'new_added': int(len(new_all)), 'v3_total': int(len(v3)),
           'by_src': {k: int(v) for k, v in new_all.src.value_counts().items()},
           'by_class_new': {str(k): int(v) for k, v in new_all.class_new.value_counts().items()}}
json.dump(summary, open(OUT.replace('.parquet', '_summary.json'), 'w'),
          ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('输出:', OUT)
