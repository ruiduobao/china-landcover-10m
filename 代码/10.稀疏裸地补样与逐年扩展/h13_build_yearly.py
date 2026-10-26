# -*- coding: utf-8 -*-
"""h13_build_yearly.py — 生成逐年补样层 + 与年度训练池合并

* 输入：
    data/final/final_2023_v2.parquet（7,098 点基准属性）
    data/final/yearly_panel.parquet（h12 逐年 keep/drop）
    data/aef_yearly/<year>/（h4 逐年 AEF64）
    data/evidence_yearly/<year>/（h11 逐年证据）
    F:/lc_work/年度子集_含稀有类/r7_train_<year>.parquet（项目既有年度训练视图）
* 输出：
    data/yearly/topup_<year>.parquet（逐年补样层：pid/lon/lat/cl/fam/A00..A63/ndvi/evidence）
    data/pools/train_<year>.parquet（既有年度视图 + 补样层；统一 schema point_id/cl/lon/lat/A*/src）
    data/yearly/yearly_stats.json/.md（逐年计数、区域分布、合并后类总量）
* 用法：python h13_build_yearly.py [--no-merge] [--years 2017 2018 ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
SUBSET = r'F:/lc_work/年度子集_含稀有类/r7_train_%d.parquet'
YEARS = list(range(2017, 2025))
FEATS = ['A%02d' % i for i in range(64)]
REGIONS = {'NE(113-135,38-54)': (113, 38, 135, 54),
           'East(107-125,18-38.5)': (107, 18, 125, 38.5),
           'ne130(130-135,41.5-54)': (130, 41.5, 135, 54)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=YEARS)
    ap.add_argument('--no-merge', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(BASE, 'data', 'yearly'), exist_ok=True)
    os.makedirs(os.path.join(BASE, 'data', 'pools'), exist_ok=True)
    fin = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'))
    fin = fin[['pid', 'lon', 'lat', 'fam', 'class']].copy()
    fin['cl'] = fin.fam.map({'sparse': 13, 'bare': 22}).astype(int)
    panel = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'yearly_panel.parquet'))
    stats = {}
    for y in a.years:
        evf = sorted(glob.glob(os.path.join(BASE, 'data', 'evidence_yearly', str(y), '*.parquet')))
        aef = sorted(glob.glob(os.path.join(BASE, 'data', 'aef_yearly', str(y), '*.parquet')))
        if not evf or not aef:
            print('%d 缺证据/AEF，跳过' % y, flush=True)
            continue
        ev = pd.concat([pd.read_parquet(f) for f in evf], ignore_index=True).drop_duplicates('pid')
        ae = pd.concat([pd.read_parquet(f) for f in aef], ignore_index=True).drop_duplicates('pid')
        p = panel[panel.year == y][['pid', 'status', 'evidence', 'used_band', 'ndvi']]
        k = p[p.status == 'keep'].merge(fin, on='pid', how='inner')
        k = k.merge(ae[['pid'] + FEATS], on='pid', how='left', validate='one_to_one')
        evc = ev[['pid', 'water', 'built', 'crops', 'snow_and_ice', 'shrub', 'bare']]
        k = k.merge(evc, on='pid', how='left')
        k['year'] = y
        k['src'] = 'fcs10_2023_ne_east_topup'
        miss = int(k[FEATS].isna().all(axis=1).sum())
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        k.to_parquet(fp, index=False)
        reg = {}
        for nm, bb in REGIONS.items():
            m = (k.lon >= bb[0]) & (k.lon < bb[2]) & (k.lat >= bb[1]) & (k.lat < bb[3])
            reg[nm] = {'sparse': int(((k.fam == 'sparse') & m).sum()),
                       'bare': int(((k.fam == 'bare') & m).sum())}
        merged = None
        if not a.no_merge and os.path.isfile(SUBSET % y):
            base = pd.read_parquet(SUBSET % y)
            add = k.copy()
            add['row_id'] = 20000000 + y * 100000 + np.arange(len(add))
            add['class_new'] = add.fam.map({'sparse': 150, 'bare': 201}).astype(int)
            add['year'] = y
            add['tier'] = 'strong'
            add['src_conf'] = 0.95
            add['train_weight'] = 1.0
            add['qc_status'] = 'keep'
            cols = ['row_id'] + FEATS + ['lon', 'lat', 'class_new', 'year', 'tier', 'src',
                                         'src_conf', 'train_weight', 'qc_status']
            add = add[cols]
            base = base[cols]
            merged = pd.concat([base, add], ignore_index=True)
            merged.to_parquet(os.path.join(BASE, 'data', 'pools', 'train_%d.parquet' % y), index=False)
        stats[y] = {'topup_keep': int(len(k)), 'sparse': int((k.fam == 'sparse').sum()),
                    'bare': int((k.fam == 'bare').sum()),
                    'aef_missing': miss, 'by_region': reg,
                    'merged_n': int(len(merged)) if merged is not None else None,
                    'merged_class150': int((merged.class_new == 150).sum()) if merged is not None else None,
                    'merged_class201': int((merged.class_new == 201).sum()) if merged is not None else None}
        print('%d: 补样 %d（稀疏 %d/裸地 %d，AEF缺失 %d）%s' % (
            y, len(k), stats[y]['sparse'], stats[y]['bare'], miss,
            (' ｜ 合并池 %d' % len(merged)) if merged is not None else ''), flush=True)
    json.dump(stats, open(os.path.join(BASE, 'data', 'yearly', 'yearly_stats.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', os.path.join(BASE, 'data', 'yearly', 'yearly_stats.json'))


if __name__ == '__main__':
    main()
