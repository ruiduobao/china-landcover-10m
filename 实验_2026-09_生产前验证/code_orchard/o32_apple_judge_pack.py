# -*- coding: utf-8 -*-
"""o32_apple_judge_pack.py — 苹果候选判读包（省级分层 ~160 点）

* 输入：data/orchard/aomc_apple_points.parquet（o31 产出，含 prov）
* 规则源（预注册）：按省配额（陕/甘/鲁/晋/辽/冀/豫/新/滇/川/蒙/宁/苏 按候选量分配，总 ~160），
        每省 ≥2 km 贪心间距
* 输出：data/orchard/aa_judge_points.csv + aa_plan.json
* 用法：python o32_apple_judge_pack.py
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
MIN_KM = 2.0
SEED = 20261013


def greedy(df, n, rng, fixed):
    if len(df) == 0 or n <= 0:
        return df.iloc[[]]
    R = MIN_KM / 111.0
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
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(os.path.join(ORD, 'aomc_apple_points.parquet'))
    cnt = d.prov.value_counts()
    cnt = cnt[cnt > 0]
    weights = cnt / cnt.sum()
    quota = (weights * N_TOTAL).round().astype(int)
    rows, plan, fixed = [], {}, []
    used = 0
    for prov in cnt.index:
        n = max(1, int(quota.get(prov, 1)))
        g = d[d.prov == prov].reset_index(drop=True)
        got = greedy(g, n, rng, fixed)
        plan[prov] = dict(avail=int(len(g)), want=n, got=int(len(got)))
        for r in got.itertuples():
            fixed.append((float(r.lon), float(r.lat)))
            rows.append({'point_id': 'AA-%03d' % used, 'lon': float(r.lon), 'lat': float(r.lat),
                         'prov': prov})
            used += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ORD, 'aa_judge_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'n_total': int(len(df)), 'plan': plan, 'seed': SEED}, os.path.join(ORD, 'aa_plan.json'))
    print('判读点 %d' % len(df))
    print(json.dumps(plan, ensure_ascii=False, indent=0)[:800])


if __name__ == '__main__':
    main()
