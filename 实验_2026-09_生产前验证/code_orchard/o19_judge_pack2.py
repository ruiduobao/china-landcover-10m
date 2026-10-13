# -*- coding: utf-8 -*-
"""o19_judge_pack2.py — 第二批判读包：茶园（TeaMap 2022）× 橡胶（Rubber_SEA 2020）各 ~160 点

* 输入：data/orchard/tea_2022_cn_points.parquet（茶园候选）
        data/orchard/rubber_sea_cn_points.parquet（橡胶候选，海南/云南）
* 规则源（预注册配额）：
    茶园：按省分层（云南40/贵州30/四川25/湖北20/湖南15/福建15/其他15），≥2km 间距
    橡胶：海南 80 + 云南 80，≥2km 间距
* 门槛：总 ~320 点；campaign 列区分
* 输出：data/orchard/judge_points_r2.csv + judge_plan_r2.json
* 用法：python o19_judge_pack2.py
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
TEA_QUOTA = {'云南省': 40, '贵州省': 30, '四川省': 25, '湖北省': 20, '湖南省': 15, '福建省': 15}
TEA_OTHER = 15
RUB_QUOTA = {'海南省': 80, '云南省': 80}
MIN_KM = 2.0
SEED = 20261013


def greedy(df, n, min_km, rng, fixed):
    if len(df) == 0 or n <= 0:
        return df.iloc[[]]
    R = min_km / 111.0
    idx = rng.permutation(len(df))
    sel, pts = [], list(fixed)
    arr = df[['lon', 'lat']].to_numpy()
    for i in idx:
        lo, la = arr[i]
        if pts:
            a = np.asarray(pts)
            d = np.hypot((a[:, 0] - lo) * np.cos(np.radians(la)), a[:, 1] - la)
            if d.min() < R:
                continue
        sel.append(i); pts.append((lo, la))
        if len(sel) >= n:
            break
    return df.iloc[sel]


def main():
    rng = np.random.default_rng(SEED)
    rows, plan = [], {}
    fixed = []
    # 茶园
    tea = pd.read_parquet(os.path.join(ORD, 'tea_2022_cn_points.parquet'))
    try:
        import k19_provattrib as PA
        tea['prov'] = PA.attribute(tea)
    except Exception:
        tea['prov'] = ''
    used = 0
    for prov, quota in TEA_QUOTA.items():
        g = tea[tea.prov == prov].reset_index(drop=True)
        got = greedy(g, quota, MIN_KM, rng, fixed)
        plan['tea_' + prov.replace('省','')] = dict(avail=int(len(g)), want=quota, got=int(len(got)))
        for r in got.itertuples():
            fixed.append((float(r.lon), float(r.lat)))
            rows.append({'point_id': 'R2T-%03d' % used, 'lon': float(r.lon), 'lat': float(r.lat),
                         'campaign': 'tea', 'prov': prov})
            used += 1
    others = tea[~tea.prov.isin(TEA_QUOTA)].reset_index(drop=True)
    got = greedy(others, TEA_OTHER, MIN_KM, rng, fixed)
    plan['tea_其他'] = dict(avail=int(len(others)), want=TEA_OTHER, got=int(len(got)))
    for r in got.itertuples():
        fixed.append((float(r.lon), float(r.lat)))
        rows.append({'point_id': 'R2T-%03d' % used, 'lon': float(r.lon), 'lat': float(r.lat),
                     'campaign': 'tea', 'prov': str(r.prov)})
        used += 1
    # 橡胶
    rub = pd.read_parquet(os.path.join(ORD, 'rubber_sea_cn_points.parquet'))
    try:
        rub['prov'] = PA.attribute(rub)
    except Exception:
        rub['prov'] = ''
    used = 0
    for prov, quota in RUB_QUOTA.items():
        g = rub[rub.prov == prov].reset_index(drop=True)
        got = greedy(g, quota, MIN_KM, rng, fixed)
        plan['rub_' + prov.replace('省','')] = dict(avail=int(len(g)), want=quota, got=int(len(got)))
        for r in got.itertuples():
            fixed.append((float(r.lon), float(r.lat)))
            rows.append({'point_id': 'R2R-%03d' % used, 'lon': float(r.lon), 'lat': float(r.lat),
                         'campaign': 'rubber', 'prov': prov})
            used += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ORD, 'judge_points_r2.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'seed': SEED, 'min_km': MIN_KM, 'plan': plan, 'n_total': int(len(df)),
              'by_campaign': df.campaign.value_counts().to_dict()}, os.path.join(ORD, 'judge_plan_r2.json'))
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    print('判读点 %d（tea %d / rubber %d）→ judge_points_r2.csv' % (
        len(df), (df.campaign == 'tea').sum(), (df.campaign == 'rubber').sum()))


if __name__ == '__main__':
    main()
