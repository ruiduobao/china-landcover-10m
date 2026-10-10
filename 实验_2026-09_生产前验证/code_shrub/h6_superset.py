# -*- coding: utf-8 -*-
"""h6_superset.py — 预选超集：不等裁决，先按省/盒配额抽"拟用于放量"的候选点（供 h2b 抢先抓特征）

* 输入：cand_nat.parquet
* 规则源（预注册）：省配额=省候选数等比例，省上限 --prov-cap；盒上限 --box-cap；点距 ≥1 km；
        正样本仅 P 带（A/B 层），负样本仅 N 带；不按审计裁决过滤（裁决后由 h6_select 二次筛选）
* 门槛：默认正 90000 + 负 25000
* 输出：superset_points.csv（同 h6 输出格式）
* 用法：python h6_superset.py [--target 90000] [--neg 25000]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
MIN_KM = 1.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def draw(df, target, box_cap, prov_cap, rs, used):
    pp = df.groupby('prov').size().sort_values(ascending=False)
    wgt = (pp / pp.sum()) * target
    quota = {p: int(min(round(w), min(prov_cap, pp[p]))) for p, w in wgt.items()}
    got = []
    for prov, q in quota.items():
        sub = df[df['prov'] == prov]
        if len(sub) == 0 or q <= 0:
            continue
        sub = sub.iloc[rs.permutation(len(sub))]
        take, bc = [], {}
        for r in sub.itertuples():
            if len(take) >= q:
                break
            if bc.get(r.box, 0) >= box_cap:
                continue
            if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (MIN_KM / 111.0) ** 2
                   for u in used[-1500:]):
                continue
            used.append((r.lon, r.lat))
            bc[r.box] = bc.get(r.box, 0) + 1
            take.append(r.Index)
        got.append(df.loc[take])
        emit('  %s %d 点' % (prov, len(take)))
    return pd.concat(got, ignore_index=True) if got else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=90000)
    ap.add_argument('--neg', type=int, default=25000)
    ap.add_argument('--prov-cap', type=int, default=6000)
    ap.add_argument('--box-cap', type=int, default=1200)
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(OUTD, 'cand_nat.parquet'))
    rs = np.random.RandomState(20261011)
    used = []
    pos = draw(df[df['band'] == 'P'], a.target, a.box_cap, a.prov_cap, rs, used)
    pos['sample_role'] = 'shrub_pos'
    neg = draw(df[df['band'] == 'N'], a.neg, a.box_cap, a.prov_cap // 2, rs, used) if a.neg else pd.DataFrame()
    neg['sample_role'] = 'hard_neg'
    allp = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=rs).reset_index(drop=True)
    allp.insert(0, 'point_id', ['SN-%s-%06d' % ('P' if r.sample_role == 'shrub_pos' else 'N', i)
                                for i, r in enumerate(allp.itertuples())])
    allp.to_csv(os.path.join(OUTD, 'superset_points.csv'), index=False, encoding='utf-8-sig')
    json.dump(dict(target=a.target, neg=a.neg, n=len(allp),
                   pos_by_prov={p: int(n) for p, n in pos.groupby('prov').size().items()},
                   neg_by_prov={p: int(n) for p, n in neg.groupby('prov').size().items()}),
              open(os.path.join(OUTD, 'superset_plan.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ 超集 %d 点（正 %d + 负 %d）' % (len(allp), len(pos), len(neg)))


if __name__ == '__main__':
    main()
