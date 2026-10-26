# -*- coding: utf-8 -*-
"""h1c_holdout_filter.py — 候选点剔除评估留出（25 km 块 + 五窗评估点 100 m）

* 依据：`数据/本地处理/样本底座/validation_pool_v2_holdout_blocks.json`
    Albers(+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104) 25 km 块键：
    bk = floor(X/25000)*100000 + floor(Y/25000)；块内训练点必须剔除（评估口径）
* 另：五窗 eval_w1..w5 的具体评估点 100 m 内剔除（row_id → 母库坐标）
* 输入/输出：data/candidates/cand_north.parquet、cand_east.parquet（就地重写）
* 用法：python h1c_holdout_filter.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
HOLD = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本底座/validation_pool_v2_holdout_blocks.json'
R7 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本重建/r7_train.parquet'
EVALS = ([r'F:/lc_work/样本修正/geetest/eval_w%d.parquet' % i for i in range(1, 5)]
         + [r'F:/lc_work/v31_exp/data/eval_w5.parquet'])
TR = Transformer.from_crs('EPSG:4326',
                          '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 '
                          '+datum=WGS84 +units=m', always_xy=True)
BLOCK = 25000


def kmxy(lon, lat):
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def main():
    keys = set(json.load(open(HOLD, encoding='utf-8'))['block_keys'])
    print('留出块 %d 个' % len(keys), flush=True)
    ids = set()
    for f in EVALS:
        if os.path.isfile(f):
            ids |= set(pd.read_parquet(f, columns=['row_id']).row_id.astype(str))
    r7 = pd.read_parquet(R7, columns=['row_id', 'lon', 'lat'])
    r7['row_id'] = r7.row_id.astype(str)
    ev = r7[r7.row_id.isin(ids)]
    print('五窗评估点（可定位）%d 个' % len(ev), flush=True)
    evt = cKDTree(kmxy(ev.lon.to_numpy(), ev.lat.to_numpy()))

    for reg in ('north', 'east', 'ne130'):
        fp = os.path.join(BASE, 'data', 'candidates', 'cand_%s.parquet' % reg)
        if not os.path.isfile(fp):
            continue
        d = pd.read_parquet(fp)
        X, Y = TR.transform(d.lon.to_numpy(), d.lat.to_numpy())
        bk = (np.floor(X / BLOCK).astype(np.int64) * 100000 +
              np.floor(Y / BLOCK).astype(np.int64))
        in_block = np.array([int(b) in keys for b in bk])
        dist, _ = evt.query(kmxy(d.lon.to_numpy(), d.lat.to_numpy()), k=1,
                            distance_upper_bound=0.1)
        near_ev = np.isfinite(dist)
        keep = ~(in_block | near_ev)
        st = {'n_in': int(len(d)), 'removed_holdout_block': int(in_block.sum()),
              'removed_near_eval': int((near_ev & ~in_block).sum()),
              'n_out': int(keep.sum())}
        d[keep].to_parquet(fp, index=False)
        print('%s: %s' % (reg, st), flush=True)
        json.dump(st, open(fp.replace('.parquet', '_holdout.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
