# -*- coding: utf-8 -*-
"""b3_prep_list.py — 把多源裸地候选整理成逐年抓取所需的点表

* 输入：data/bare_cand/bare_cand_filtered.parquet
* 输出：data/final/bare_south_cand.parquet（pid/lon/lat/fam/class）
"""
import os, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
d = pd.read_parquet(os.path.join(BASE, 'data', 'bare_cand', 'bare_cand_filtered.parquet'))
out = d[['pid', 'lon', 'lat']].copy()
out['fam'] = 'bare'
out['class'] = 201
out.to_parquet(os.path.join(BASE, 'data', 'final', 'bare_south_cand.parquet'), index=False)
print('候选', len(out), out[['lon', 'lat']].describe().loc[['min', 'max']].to_dict())
