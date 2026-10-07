# -*- coding: utf-8 -*-
"""
n7_merge_v31.py — v3 + 东北作物(585k) + EGLC 中国(28k) → 样本底座 v3.1
* 去重: 距 v3 <100m 剔除；源内 250m 贪心抽稀；跨源 <100m 互斥
* 输出: 数据/本地处理/样本底座/cn_samples_v3_1.parquet + summary
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_KM, THIN = 0.1, 0.25
DEG_KM = None

def to_km(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])

def dedupe_thin(base_tree, df, thin=THIN):
    n0 = len(df)
    d, _ = base_tree.query(to_km(df.lon.to_numpy(), df.lat.to_numpy()), k=1)
    df = df[d > R_KM].reset_index(drop=True)
    km = to_km(df.lon.to_numpy(), df.lat.to_numpy())
    order = np.random.default_rng(7).permutation(len(df))
    cell = 0.25; kept = {}; keep_idx = []
    for i in order:
        key = (int(km[i, 0] // cell), int(km[i, 1] // cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((key[0] + dx, key[1] + dy), []):
                    if np.hypot(km[i, 0] - km[j, 0], km[i, 1] - km[j, 1]) < thin / 1000:
                        ok = False; break
                if not ok: break
            if ok: break
        if ok:
            kept.setdefault(key, []).append(i); keep_idx.append(i)
    return df.iloc[sorted(keep_idx)].reset_index(drop=True), n0

def main():
    v3 = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v3.parquet')
    base_tree = cKDTree(to_km(v3.lon.to_numpy(), v3.lat.to_numpy()))

    ne = pd.read_parquet('数据/外部样本源/ne_crops_samples.parquet')
    eg = pd.read_parquet('数据/外部样本源/eglc_china.parquet')

    ne_f, n0 = dedupe_thin(base_tree, ne)
    print(f'东北作物: {n0:,} -> {len(ne_f):,}')
    eg_f, n0 = dedupe_thin(base_tree, eg)
    print(f'EGLC: {n0:,} -> {len(eg_f):,}')

    new = pd.concat([ne_f, eg_f], ignore_index=True)
    tree = cKDTree(to_km(new.lon.to_numpy(), new.lat.to_numpy()))
    pairs = tree.query_pairs(R_KM)
    drop = {b for _, b in pairs}
    new = new.drop(index=list(drop)).reset_index(drop=True)
    new['tier'] = 'external_pending'
    print('互去重后:', len(new))

    for c in ['src', 'src_conf']:
        if c not in v3.columns:
            v3[c] = None
    v31 = pd.concat([v3, new.reindex(columns=v3.columns)], ignore_index=True)
    OUT = '数据/本地处理/样本底座/cn_samples_v3_1.parquet'
    v31.to_parquet(OUT, index=False)
    summary = {'v3_total': int(len(v3)), 'new_added': int(len(new)),
               'v3_1_total': int(len(v31)),
               'by_src': {k: int(v) for k, v in new.src.str.split('_2').str[0].value_counts().items()},
               'by_class_new': {str(k): int(v) for k, v in new.class_new.value_counts().items()}}
    json.dump(summary, open(OUT.replace('.parquet', '_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('输出:', OUT)

if __name__ == '__main__':
    main()
