# -*- coding: utf-8 -*-
"""b6_publish_bare_south.py — 南方多源裸地层并入裸地类目录（备份 → 合并 → 逐年 GPKG）

* 备份：裸地/<year>/*.gpkg → 裸地/旧样本_2026-10-26_补样前/<year>/
* 合并：旧 + topup_bare_south_<year>（src='multisrc_s2_bare_south'），50 m 去重
* 输出：裸地/<year>/裸地_<year>_WGS84.gpkg（同 schema 74 字段）+ 逐年池成员数.csv + 补样来源明细.csv
* 用法：python b6_publish_bare_south.py
"""
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
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
DATA = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
CDIR = os.path.join(DATA, '裸地')
BACKUP = '旧样本_2026-10-26_补样前'
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
FEATS = ['A%02d' % i for i in range(64)]
COLS = ['pid', 'lon', 'lat', 'cls24', 'cls_zh', 'tier', 'src', 'ct_prov', 'ct_code', 'ct_km'] + FEATS


def kmxy(lon, lat):
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def main():
    prov = gpd.read_file(PROV_SHP)[['省', '省级码', 'geometry']].rename(
        columns={'省': 'ct_prov', '省级码': 'ct_code'})
    prov['geometry'] = prov.geometry.buffer(0)
    rows = []
    for y in range(2017, 2025):
        fp = os.path.join(CDIR, str(y), '裸地_%d_WGS84.gpkg' % y)
        bk = os.path.join(CDIR, BACKUP, str(y), '裸地_%d_WGS84.gpkg' % y)
        os.makedirs(os.path.dirname(bk), exist_ok=True)
        if os.path.isfile(fp) and not os.path.isfile(bk):
            try:
                shutil.copy2(fp, bk)      # 复制备份（QGIS 只读锁也能复制）
                print('backup(copy)', bk, flush=True)
            except Exception as e:
                print('backup FAIL', bk, e, flush=True); continue
        src = bk if os.path.isfile(bk) else fp
        old = pyogrio.read_dataframe(src) if os.path.isfile(src) else None
        tf = os.path.join(BASE, 'data', 'yearly', 'topup_bare_south_%d.parquet' % y)
        if not os.path.isfile(tf):
            print('%d 无新增，跳过' % y); continue
        t = pd.read_parquet(tf)
        g = gpd.GeoDataFrame(t, geometry=[Point(x, yy) for x, yy in zip(t.lon, t.lat)], crs='EPSG:4326')
        j = gpd.sjoin(g, prov, predicate='within', how='left')
        j = j[~j.index.duplicated(keep='first')].drop(columns=[c for c in ('index_right',) if c in j.columns])
        miss = j.ct_code.isna()
        if miss.any():
            pv = prov.rename(columns={'ct_prov': 'prov_near', 'ct_code': 'code_near'})
            sub = gpd.GeoDataFrame(j.loc[miss, ['lon', 'lat']], geometry=j.loc[miss, 'geometry'].values,
                                   crs='EPSG:4326')
            near = gpd.sjoin_nearest(sub.reset_index(), pv, how='left',
                                     distance_col='_d').drop_duplicates('index').set_index('index')
            j.loc[miss, 'ct_prov'] = near['prov_near']; j.loc[miss, 'ct_code'] = near['code_near']
            j['ct_km'] = 0.0; j.loc[miss, 'ct_km'] = near['_d'] / 1000.0 * 111.32
        else:
            j['ct_km'] = 0.0
        add = pd.DataFrame({
            'pid': j.pid.astype(str), 'lon': j.lon, 'lat': j.lat, 'cls24': 22, 'cls_zh': '裸地',
            'tier': 'strong', 'src': 'multisrc_s2_bare_south',
            'ct_prov': j.ct_prov, 'ct_code': j.ct_code, 'ct_km': j.ct_km})
        for c in FEATS:
            add[c] = j[c].astype('float32')
        add = gpd.GeoDataFrame(add, geometry=[Point(x, yy) for x, yy in zip(add.lon, add.lat)],
                               crs='EPSG:4326')[COLS + ['geometry']].reset_index(drop=True)
        n_add0 = len(add)
        if old is not None:
            old = old[COLS + ['geometry']].reset_index(drop=True)
            old['pid'] = old.pid.astype(str)
            tree = cKDTree(kmxy(old.lon.to_numpy(), old.lat.to_numpy()))
            dd, _ = tree.query(kmxy(add.lon.to_numpy(), add.lat.to_numpy()), k=1,
                               distance_upper_bound=0.05)
            keep = ~np.isfinite(dd)
            add = add[keep].reset_index(drop=True)
            out = pd.concat([old, add], ignore_index=True)
        else:
            out = add
        stage_dir = os.path.join(CDIR, '_staging_2026-10-26')
        os.makedirs(stage_dir, exist_ok=True)
        stg = os.path.join(stage_dir, '裸地_%d_WGS84.gpkg' % y)
        if os.path.isfile(stg):
            os.remove(stg)
        out.to_file(stg, layer='samples', driver='GPKG', engine='pyogrio')
        applied = False
        try:
            os.replace(stg, fp)
            applied = True
        except PermissionError:
            applied = False
        if not applied:
            print('%d 目标被占用（QGIS？）→ 暂存 %s' % (y, stg), flush=True)
        n_old = 0 if old is None else len(old)
        rows.append({'year': y, 'n_total': int(len(out)), 'n_base': int(n_old),
                     'n_south': int(len(add)), 'dup_removed': int(n_add0 - len(add)),
                     'applied': bool(applied)})
        print('%d: base %d + south %d (dup %d) = %d' % (y, n_old, len(add), n_add0 - len(add), len(out)),
              flush=True)
        del old, out, add, t
    if rows:
        df = pd.DataFrame(rows)
        # 重建 逐年池成员数.csv（保持旧列名；n_topup = 全部补样）
        old_mem = pd.read_csv(os.path.join(CDIR, '逐年池成员数.csv'))
        old_mem = old_mem.rename(columns={'n_topup': 'n_topup_east'})
        m = old_mem.merge(df[['year', 'n_south', 'n_total']], on='year', how='outer')
        m['n_topup'] = m['n_topup_east'].fillna(0) + m['n_south'].fillna(0)
        m['n_base'] = m['n_base'].fillna(m['n_total'] - m['n_topup'])
        m[['year', 'n_total', 'n_base', 'n_topup', 'dup_removed']].to_csv(
            os.path.join(CDIR, '逐年池成员数.csv'), index=False, encoding='utf-8-sig')
        m[['year', 'n_topup_east', 'n_south', 'n_topup', 'n_total']].to_csv(
            os.path.join(CDIR, '补样来源明细.csv'), index=False, encoding='utf-8-sig')
    json.dump(rows, open(os.path.join(BASE, 'data', 'yearly', 'publish_bare_south_stats.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→ 完成')


if __name__ == '__main__':
    main()
