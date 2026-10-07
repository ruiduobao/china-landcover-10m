# -*- coding: utf-8 -*-
"""
n18_final_clean.py — 从 v4 一步到位生成最终训练样本集
* 输入: cn_samples_v4.parquet (9,514,648 点)
* 过滤:
  1. 只保留 gold/silver/bronze/external_pending（去 reject+uncovered）
  2. 生态合理性（每类合法纬度范围，范围外剔除）
  3. 同类500m最小间距贪心抽稀
  4. 0.25°格配额（每格每类≤200）
* 输出: cn_samples_v5_train.parquet + 完整统计
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
OUT = '数据/本地处理/样本底座/cn_samples_v5_train.parquet'

LAT_RANGE = {
    10: (18,54), 11: (18,42), 12: (18,52),
    51: (18,34), 52: (18,36), 61: (18,54), 62: (18,54),
    71: (18,48), 72: (18,50), 81: (30,54), 82: (30,54),
    91: (33,54), 92: (33,54),
    120: (18,38), 121: (25,54),
    130: (18,54), 140: (26,54), 150: (26,54),
    180: (38,54), 181: (18,54), 182: (18,54),
    183: (30,50), 184: (18,26), 185: (30,42), 186: (18,42),
    190: (18,54), 200: (18,54), 201: (18,54),
    202: (18,54), 220: (28,54),
}
CELL_QUOTA = 200

def to_km(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return np.column_stack([lon*111.32*np.cos(np.radians(lat)), lat*110.57])

def main():
    t0 = time.time()
    df = pd.read_parquet(V4, columns=['lon','lat','class_new','tier','year','src','src_conf'])
    print(f'输入: {len(df):,}', flush=True)

    # 1) 剔 reject + uncovered
    n0 = len(df)
    df = df[df.tier.isin(['gold','silver','bronze','external_pending'])].reset_index(drop=True)
    print(f'1) 剔reject+uncovered: {n0:,} -> {len(df):,} (剔 {n0-len(df):,})', flush=True)

    # 2) 生态纬度范围
    lon = df.lon.to_numpy(); lat = df.lat.to_numpy()
    cls = df.class_new.to_numpy(int)
    lat_ok = np.ones(len(df), dtype=bool)
    n_eco = 0
    for c, (lo, hi) in LAT_RANGE.items():
        m = cls == c
        if m.sum() == 0: continue
        bad = (lat[m] < lo) | (lat[m] > hi)
        lat_ok[np.flatnonzero(m)[bad]] = False
        if bad.sum(): print(f'   class{c}: 剔纬度违规 {bad.sum():,}')
    df = df[lat_ok].reset_index(drop=True)
    print(f'2) 生态纬度: {n0} -> {len(df):,} (剔 {n0-len(df):,})', flush=True)

    # 3) 同类500m贪心抽稀
    from scipy.spatial import cKDTree
    km_x = df.lon.to_numpy() * 111.32 * np.cos(np.radians(df.lat.to_numpy()))
    km_y = df.lat.to_numpy() * 110.57
    km = np.column_stack([km_x, km_y])
    tree = cKDTree(km)
    pairs = tree.query_pairs(0.5)  # 500m
    # 构建邻接表
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in pairs:
        if df.class_new.iloc[i] == df.class_new.iloc[j]:
            adj[i].add(j); adj[j].add(i)
    # 贪心独立集（高conf优先）
    order = np.argsort(-df.src_conf.fillna(0.5).to_numpy())
    removed = set()
    for i in order:
        if i in removed: continue
        for j in adj.get(i, set()):
            if j not in removed and j in adj.get(i, set()):
                removed.add(j)
    n0 = len(df)
    df = df.drop(index=removed).reset_index(drop=True)
    print(f'3) 同类500m抽稀: {n0:,} -> {len(df):,} (剔 {n0-len(df):,})', flush=True)

    # 4) 0.25°格配额
    df['cell_q'] = np.floor(df.lon*4).astype(int)*10000 + np.floor(df.lat*4).astype(int)
    grp = df.groupby(['cell_q','class_new']).size()
    over = grp[grp > CELL_QUOTA].index
    drop_set = set()
    for (cq, c_cls) in over:
        m = (df.cell_q == cq) & (df.class_new == c_cls)
        if m.sum() > CELL_QUOTA:
            drop = df[m].sample(m.sum() - CELL_QUOTA, random_state=42).index
            drop_set.update(drop)
    if drop_set:
        df = df.drop(drop_set).reset_index(drop=True)
    print(f'4) 0.25°格配额: 剔 {quota_removed if (quota_removed:=len(drop_set)) else 0:,}', flush=True)

    # 统计
    print(f'\n{"="*50}')
    print(f'最终训练集: {len(df):,} 点, {df.class_new.nunique()} 类')
    print(f'\ntier: {dict(df.tier.value_counts())}')
    print(f'\n类别分布:')
    NAMES = {10:'旱地',11:'园地',12:'灌溉耕地',51:'郁闭常绿阔',52:'疏闭常绿阔',
             61:'郁闭落叶阔',62:'疏闭落叶阔',71:'郁闭常绿针',72:'疏闭常绿针',
             81:'郁闭落叶针',82:'疏闭落叶针',91:'郁闭混交',92:'疏闭混交',
             120:'常绿灌丛',121:'落叶灌丛',130:'草地',140:'苔藓',150:'稀疏',
             180:'木本沼泽',181:'草本沼泽',182:'湖河滩',183:'盐渍',184:'红树',
             185:'盐沼',186:'潮滩',190:'城镇',200:'乡村',201:'裸地',202:'水体',220:'冰雪'}
    for c in sorted(df.class_new.unique()):
        n = (df.class_new==c).sum()
        print(f'  {c:>3} {NAMES.get(c,"?"):<10} {n:>8,}')
    
    df.to_parquet(OUT, index=False)
    print(f'\n输出: {OUT}')
    json.dump({'total': len(df), 'classes': int(df.class_new.nunique()),
               'by_class': {str(k): int(v) for k, v in df.class_new.value_counts().items()}},
              open(OUT.replace('.parquet','_summary.json'),'w'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
