# -*- coding: utf-8 -*-
"""h20_publish_class_yearly.py — 将最终逐年样本写入项目类目录（GPKG），旧样本先备份

* 输入：
    旧：数据/单类样本的单独处理_2026.10.12/<类>/<year>/<类>_<year>_WGS84.gpkg
    新：F:/lc_work/bare_sparse_v3/data/yearly/topup_<year>.parquet（含当年 AEF64）
* 动作：
    1) 旧 GPKG 移到 <类>/旧样本_2026-10-25/<year>/（保留原文件名）
    2) 旧 + 新补样（省界由 CTAmap 计算）合并，50 m 内去重，写回原路径
    3) 更新 逐年池成员数.csv（year,n_total,n_base,n_topup）
* 用法：python h20_publish_class_yearly.py [--years 2017 ...] [--classes 稀疏植被 裸地]
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import Point

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
DATA = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
AEQ = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 +datum=WGS84 +units=m +no_defs')
FEATS = ['A%02d' % i for i in range(64)]
CLS = {'稀疏植被': (13, 'sparse'), '裸地': (22, 'bare')}
BACKUP = '旧样本_2026-10-25'


def build_prov_map():
    fp = os.path.join(BASE, 'data', 'final', 'topup_prov.parquet')
    if os.path.isfile(fp):
        return pd.read_parquet(fp)
    d = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'),
                        columns=['pid', 'lon', 'lat'])
    g = gpd.GeoDataFrame(d, geometry=[Point(x, y) for x, y in zip(d.lon, d.lat)], crs='EPSG:4326')
    prov = gpd.read_file(PROV_SHP)[['省', '省级码', 'geometry']].rename(
        columns={'省': 'ct_prov', '省级码': 'ct_code'})
    prov['geometry'] = prov.geometry.buffer(0)
    j = gpd.sjoin(g, prov, predicate='within', how='left')
    j = j[~j.index.duplicated(keep='first')].drop(columns=[c for c in ('index_right',) if c in j.columns])
    miss = j.ct_prov.isna()
    j['ct_km'] = 0.0
    if miss.any():
        pv = prov.rename(columns={'ct_prov': 'prov_near'})
        sub_a = gpd.GeoDataFrame(j.loc[miss, ['lon', 'lat']].copy(),
                                 geometry=j.loc[miss, 'geometry'].values, crs='EPSG:4326').to_crs(AEQ).reset_index()
        near = gpd.sjoin_nearest(sub_a, pv.to_crs(AEQ), how='left', distance_col='_d')
        near = near.drop_duplicates('index').set_index('index')
        j.loc[miss, 'ct_prov'] = near['prov_near']
        j.loc[miss, 'ct_code'] = near['ct_code']
        j.loc[miss, 'ct_km'] = near['_d'] / 1000.0
    out = j[['pid', 'ct_prov', 'ct_code', 'ct_km']]
    out.to_parquet(fp, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2017, 2025)))
    ap.add_argument('--classes', nargs='*', default=['稀疏植被', '裸地'])
    a = ap.parse_args()
    pm = build_prov_map()
    stats = {}
    for cname in a.classes:
        cl, fam = CLS[cname]
        cdir = os.path.join(DATA, cname)
        bdir = os.path.join(cdir, BACKUP)
        os.makedirs(bdir, exist_ok=True)
        rows = []
        for y in a.years:
            old_fp = os.path.join(cdir, str(y), '%s_%d_WGS84.gpkg' % (cname, y))
            bk_fp = os.path.join(bdir, str(y), '%s_%d_WGS84.gpkg' % (cname, y))
            os.makedirs(os.path.dirname(bk_fp), exist_ok=True)
            if os.path.isfile(old_fp) and not os.path.isfile(bk_fp):
                shutil.move(old_fp, bk_fp)
                print('  backup %s' % bk_fp, flush=True)
            src_old = bk_fp if os.path.isfile(bk_fp) else old_fp
            old = pyogrio.read_dataframe(src_old) if os.path.isfile(src_old) else None
            top = pd.read_parquet(os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y))
            top = top[top.cl == cl].merge(pm, on='pid', how='left')
            add = pd.DataFrame({
                'pid': top.pid.astype(str), 'lon': top.lon, 'lat': top.lat,
                'cls24': cl, 'cls_zh': cname, 'tier': 'strong',
                'src': 'fcs10_2023_ne_east_topup',
                'ct_prov': top.ct_prov, 'ct_code': top.ct_code, 'ct_km': top.ct_km})
            for c in FEATS:
                add[c] = top[c].astype('float32')
            add = gpd.GeoDataFrame(add, geometry=[Point(x, yy) for x, yy in zip(add.lon, add.lat)],
                                   crs='EPSG:4326')
            cols = ['pid', 'lon', 'lat', 'cls24', 'cls_zh', 'tier', 'src',
                    'ct_prov', 'ct_code', 'ct_km'] + FEATS + ['geometry']
            add = add[cols].reset_index(drop=True)
            if old is not None:
                old = old[cols].reset_index(drop=True)
                old['pid'] = old.pid.astype(str)
                # 50m 去重
                from scipy.spatial import cKDTree
                def kmxy(lon, lat):
                    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])
                tree = cKDTree(kmxy(old.lon.to_numpy(), old.lat.to_numpy()))
                dist, _ = tree.query(kmxy(add.lon.to_numpy(), add.lat.to_numpy()), k=1,
                                     distance_upper_bound=0.05)
                keep = ~np.isfinite(dist)
                n_dup = int((~keep).sum())
                add = add[keep].reset_index(drop=True)
                out = pd.concat([old, add], ignore_index=True)
            else:
                n_dup = 0
                out = add
            out_fp = old_fp
            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            if os.path.isfile(out_fp):
                os.remove(out_fp)
            out.to_file(out_fp, layer='samples', driver='GPKG', engine='pyogrio')
            n_old = 0 if old is None else len(old)
            rows.append({'year': y, 'n_total': int(len(out)), 'n_base': int(n_old),
                         'n_topup': int(len(add)), 'dup_removed': n_dup})
            print('%s %d: base %d + topup %d (dup %d) = %d → %s' % (
                cname, y, n_old, len(add), n_dup, len(out), out_fp), flush=True)
            del old, out, add, top
        pd.DataFrame(rows).to_csv(os.path.join(cdir, '逐年池成员数.csv'), index=False,
                                  encoding='utf-8-sig')
        stats[cname] = rows
    json.dump(stats, open(os.path.join(BASE, 'data', 'yearly', 'publish_stats.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→ publish_stats.json')


if __name__ == '__main__':
    main()
