# -*- coding: utf-8 -*-
"""o39_citrus_judge_pack.py — 柑橘候选判读包（省级分层 ~160 点）

* 输入：data/orchard/citrus_2020_cn_points.parquet（含 prov）
* 规则源：按候选量分配省级配额（总 ~160）；≥2 km 间距
* 输出：data/orchard/cj_judge_points.csv + cj_plan.json
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
N_TOTAL = 160


def greedy(df, n, rng, fixed):
    if len(df) == 0 or n <= 0:
        return df.iloc[[]]
    R = 2.0 / 111.0
    idx = rng.permutation(len(df))
    sel, pts = [], list(fixed)
    arr = df[['lon', 'lat']].to_numpy()
    for i in idx:
        lo, la = arr[i]
        if pts:
            a = np.asarray(pts)
            if np.hypot((a[:, 0] - lo) * np.cos(np.radians(la)), a[:, 1] - la).min() < R:
                continue
        sel.append(i); pts.append((lo, la))
        if len(sel) >= n:
            break
    return df.iloc[sel]


def main():
    rng = np.random.default_rng(20261013)
    d = pd.read_parquet(os.path.join(ORD, 'citrus_2020_cn_points.parquet'))
    cnt = d.prov.value_counts()
    cnt = cnt[cnt > 0]
    quota = (cnt / cnt.sum() * N_TOTAL).round().astype(int)
    rows, plan, fixed = [], {}, []
    used = 0
    for prov in cnt.index:
        n = max(1, int(quota.get(prov, 1)))
        g = d[d.prov == prov].reset_index(drop=True)
        got = greedy(g, n, rng, fixed)
        plan[prov] = dict(avail=int(len(g)), got=int(len(got)))
        for r in got.itertuples():
            fixed.append((float(r.lon), float(r.lat)))
            rows.append({'point_id': 'CJ-%03d' % used, 'lon': float(r.lon), 'lat': float(r.lat), 'prov': prov})
            used += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ORD, 'cj_judge_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'n_total': int(len(df)), 'plan': plan}, os.path.join(ORD, 'cj_plan.json'))
    print('判读点 %d' % len(df))
    print(json.dumps({k: v['got'] for k, v in plan.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
