# -*- coding: utf-8 -*-
"""h7_pool_v3.py — 合并"池 v2 + 东北/东部补样层" → train2023_v3.parquet

* 输入：
    --pool  F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（1,764,020；point_id/cl/lon/lat/A00..A63/src）
    --topup F:/lc_work/bare_sparse_v3/delivery/bare_sparse_v3_2023.parquet（h6 交付长表）
* 规则：
    - 补样层 cl：稀疏植被 13 / 裸地 22（24 类码；与 v2 的 cl 同一空间）
    - 与既有同类点 50 m 内去重（KDTree）
    - src='fcs10_2023_ne_east_topup'；AEF 零缺失硬断言
* 输出：--outdir/train2023_v3.parquet + pool_v3_stats.json/.md
* 用法：python h7_pool_v3.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
FEATS = ['A%02d' % i for i in range(64)]
POOL = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
TOPUP = r'F:/lc_work/bare_sparse_v3/delivery/bare_sparse_v3_2023.parquet'


def kmxy(lon, lat):
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=POOL)
    ap.add_argument('--topup', default=TOPUP)
    ap.add_argument('--outdir', default=r'F:/lc_work/bare_sparse_v3/data')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    v2 = pd.read_parquet(a.pool)
    v2['point_id'] = v2.point_id.astype(str)
    print('v2: %d 点, 类数 %d' % (len(v2), v2.cl.nunique()), flush=True)

    t = pd.read_parquet(a.topup)
    t = t.rename(columns={'pid': 'point_id', 'cls24': 'cl'})
    t['point_id'] = t.point_id.astype(str)
    t['src'] = 'fcs10_2023_ne_east_topup'
    t = t[['point_id', 'cl', 'lon', 'lat'] + FEATS + ['src']]
    print('补样: %d 点（cl: %s）' % (len(t), t.cl.value_counts().to_dict()), flush=True)

    # 同类 50 m 去重
    keep = np.ones(len(t), dtype=bool)
    for c, g in t.groupby('cl'):
        base = v2[v2.cl == c]
        if len(base) == 0:
            continue
        tree = cKDTree(kmxy(base.lon.to_numpy(), base.lat.to_numpy()))
        d, _ = tree.query(kmxy(g.lon.to_numpy(), g.lat.to_numpy()), k=1, distance_upper_bound=0.05)
        keep[g.index.to_numpy()] = ~np.isfinite(d)
    n_dup = int((~keep).sum())
    t = t[keep].reset_index(drop=True)
    print('与 v2 同类 50m 内重复剔除: %d，余 %d' % (n_dup, len(t)), flush=True)

    out = pd.concat([v2, t], ignore_index=True, sort=False)
    assert out[FEATS].isna().all(axis=1).sum() == 0, 'AEF 全缺失行存在'
    out['cl'] = out.cl.astype(int)
    fp = os.path.join(a.outdir, 'train2023_v3.parquet')
    out.to_parquet(fp, index=False)

    by_cls = out.cl.value_counts().sort_index()
    stats = {
        'n_v2': int(len(v2)), 'n_topup_added': int(len(t)), 'n_dup_removed': n_dup,
        'n_v3': int(len(out)),
        'by_class': {int(k): int(v) for k, v in by_cls.items()},
        'topup_by_class': {int(k): int(v) for k, v in t.cl.value_counts().items()},
        'file': fp}
    json.dump(stats, open(os.path.join(a.outdir, 'pool_v3_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('v3: %d 点；稀疏植被 %d（v2 %d）、裸地 %d（v2 %d）' % (
        len(out), int(by_cls.get(13, 0)), int(by_cls.get(13, 0)) - int(t.cl.value_counts().get(13, 0)),
        int(by_cls.get(22, 0)), int(by_cls.get(22, 0)) - int(t.cl.value_counts().get(22, 0))), flush=True)
    print('→', fp, flush=True)


if __name__ == '__main__':
    main()
