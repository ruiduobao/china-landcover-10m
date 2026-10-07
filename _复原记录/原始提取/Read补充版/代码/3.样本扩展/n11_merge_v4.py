# -*- coding: utf-8 -*-
"""
n11_merge_v4.py — v3.1 + GLC_FCS10(第二教师) + ESRI 全国年度 → 样本底座 v4
* 去重: 距 v3.1 <100m 剔除；源内/跨源 250m 贪心抽稀
* ESRI built(哨兵码7) → GUB 拆 190/200
* 输出: 数据/本地处理/样本底座/cn_samples_v4.parquet + summary
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_KM, THIN = 0.1, 0.25

def to_km(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])

def dedupe_thin(base_tree, df, thin=THIN):
    n0 = len(df)
    d, _ = base_tree.query(to_km(df.lon.to_numpy(), df.lat.to_numpy()), k=1)
    df = df[d > R_KM].reset_index(drop=True)
    km = to_km(df.lon.to_numpy(), df.lat.to_numpy())
    order = np.random.default_rng(11).permutation(len(df))
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

def gub_split_190_200(df):
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    gub = gpd.read_file(r'Z:\Mywork\论文\非洲城市发展和驱动力分析\数据\GUB_全球城市边界数据\GUB_Global_2020\GUB_Global_2020.shp')
    geoms = list(gub.geometry); tree = STRtree(geoms)
    urban = np.zeros(len(df), dtype=bool)
    for i, (x, y) in enumerate(zip(df.lon, df.lat)):
        pt = Point(x, y)
        for j in tree.query(pt):
            if geoms[j].contains(pt):
                urban[i] = True; break
    df['class_new'] = np.where(urban, 190, 200).astype(int)
    return df

def main():
    v31 = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v3_1.parquet')
    base_tree = cKDTree(to_km(v31.lon.to_numpy(), v31.lat.to_numpy()))

    fcs = pd.read_parquet('数据/外部样本源/glc_fcs10_samples.parquet')
    es = pd.read_parquet('数据/外部样本源/esri_national_yearly.parquet')

    # ESRI built 哨兵 → GUB 拆分
    built_mask = es['class_new'] == 7
    if built_mask.any():
        print('ESRI built 点 GUB 拆分:', int(built_mask.sum()))
        es.loc[built_mask] = gub_split_190_200(es[built_mask].copy())

    fcs_f, n0 = dedupe_thin(base_tree, fcs)
    print(f'GLC_FCS10: {n0:,} -> {len(fcs_f):,}')
    es_f, n0 = dedupe_thin(base_tree, es)
    print(f'ESRI 年度: {n0:,} -> {len(es_f):,}')

    new = pd.concat([fcs_f, es_f], ignore_index=True)
    tree = cKDTree(to_km(new.lon.to_numpy(), new.lat.to_numpy()))
    pairs = tree.query_pairs(R_KM)
    drop = {b for _, b in pairs}
    new = new.drop(index=list(drop)).reset_index(drop=True)
    new['tier'] = 'external_pending'
    print('互去重后:', f'{len(new):,}')

    for c in ['src', 'src_conf']:
        if c not in v31.columns:
            v31[c] = None
    v4 = pd.concat([v31, new.reindex(columns=v31.columns)], ignore_index=True)
    OUT = '数据/本地处理/样本底座/cn_samples_v4.parquet'
    v4.to_parquet(OUT, index=False)
    summary = {'v3_1_total': int(len(v31)), 'new_added': int(len(new)),
               'v4_total': int(len(v4)),
               'by_src_group': {k: int(v) for k, v in
                                new.src.str.split('_').str[:2].str.join('_').value_counts().items()},
               'by_class_new': {str(k): int(v) for k, v in new.class_new.value_counts().items()}}
    json.dump(summary, open(OUT.replace('.parquet', '_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('输出:', OUT)

if __name__ == '__main__':
    main()

