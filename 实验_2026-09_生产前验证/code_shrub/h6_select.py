# -*- coding: utf-8 -*-
"""h6_select.py — 放量抽样：按审计裁决的分层精度，从全国候选池抽正样本 + 硬负样本

* 输入：cand_nat.parquet（候选）、audit_verdict.json（h5 裁决：accept/shrink/reject + 比例）
* 规则源（预注册）：
        采用判定逐格按 h5 verdict；ratio=1.0/0.5/0
        正样本配额按省候选数等比例分配，省上限 --prov-cap；每盒上限 --box-cap
        点间距 ≥1 km；避开 doc57 已判读点 ≥2 km（不与其混批重复）
        物理分档（特征到齐后按 rh98_mx11 标 P1<8m / P2≥8m，本脚本只留占位列）
* 门槛：目标正样本 --target（默认 60000）；硬负样本 --neg（默认 20000）
* 输出：mass_points.csv（point_id/lon/lat/zone/prov/k/band/tier + 分配表）
        data/shrub_nat/mass_plan.json（目标/实取/每省每格配额）
* 用法：python h6_select.py [--target 60000] [--neg 20000]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PREV_JUDGED = os.path.join(WORK, 'data', 'shrub', 'judged_all.csv')
MIN_KM = 1.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=60000)
    ap.add_argument('--neg', type=int, default=20000)
    ap.add_argument('--prov-cap', type=int, default=4000)
    ap.add_argument('--box-cap', type=int, default=800)
    ap.add_argument('--pool', default='', help='候选池 parquet（默认 cand_nat；可用已特征化的超集）')
    a = ap.parse_args()
    pool_fp = a.pool or os.path.join(OUTD, 'cand_nat.parquet')
    df = pd.read_parquet(pool_fp)
    vd = json.load(open(os.path.join(OUTD, 'audit_verdict.json'), encoding='utf-8'))
    cell = vd['cells']
    df['k'] = df['band'].astype(str) + '_' + df['tier'].astype(str)
    df['verdict'] = [cell.get('%s|%s' % (z, k), {}).get('verdict', 'reject')
                     for z, k in zip(df['zone'], df['k'])]
    df['ratio'] = [float(cell.get('%s|%s' % (z, k), {}).get('ratio', 0.0))
                  for z, k in zip(df['zone'], df['k'])]
    emit('候选 %d；verdict 分布 %s' % (len(df), df['verdict'].value_counts().to_dict()))
    ok = df[df['verdict'] != 'reject'].copy()
    emit('可用（accept+shrink）%d 点，按 province × k 汇总' % len(ok))
    if len(ok) == 0:
        emit('无可采用格，退出'); return
    # 已判读点回避
    prev = []
    if os.path.exists(PREV_JUDGED):
        try:
            pv = pd.read_csv(PREV_JUDGED, encoding='utf-8-sig')
            prev = list(zip(pv['lon'].to_numpy(float), pv['lat'].to_numpy(float)))
        except Exception:
            prev = []
    # 正样本分配：按省可用候选数等比例（只 P 带）
    pos = ok[ok['band'] == 'P']
    pp = pos.groupby('prov').size().sort_values(ascending=False)
    wgt = (pp / pp.sum()) * a.target
    quota = {p: int(min(round(w), min(a.prov_cap, pp[p]))) for p, w in wgt.items()}
    emit('正样本目标 %d，按省配额 top10：%s' % (a.target, dict(list(sorted(quota.items(), key=lambda x: -x[1]))[:10])))
    rs = np.random.RandomState(20261011)
    chosen = []
    used = list(prev)
    for prov, q in quota.items():
        sub = pos[pos['prov'] == prov]
        if len(sub) == 0 or q <= 0:
            continue
        # 按格 ratio 加权抽取：先全排后按 ratio 截断（ratio=0.5 表示只取半数）
        sub = sub.iloc[rs.permutation(len(sub))]
        boxcnt, take = {}, []
        for r in sub.itertuples():
            if len(take) >= q:
                break
            if boxcnt.get(r.box, 0) >= a.box_cap:
                continue
            if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (MIN_KM / 111.0) ** 2
                   for u in used[-1500:]):
                continue
            used.append((r.lon, r.lat))
            boxcnt[r.box] = boxcnt.get(r.box, 0) + 1
            take.append(r.Index)
        sub2 = pos.loc[take]
        # ratio 收缩：只保留 ratio=1.0 与部分 ratio=0.5 的点
        if (sub2['ratio'] < 1.0).any():
            keep_full = sub2[sub2['ratio'] >= 1.0]
            keep_half = sub2[sub2['ratio'] < 1.0]
            n_half = int(round(len(keep_half) * 0.5))
            sub2 = pd.concat([keep_full, keep_half.sample(n=min(n_half, len(keep_half)),
                                                          random_state=rs)])
        chosen.append(sub2)
        emit('  %s 取 %d 点（目标 %d）' % (prov, len(sub2), q))
    posdf = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    # 硬负样本（N 带可用格）
    neg = ok[ok['band'] == 'N']
    negdf = pd.DataFrame()
    if len(neg) and a.neg > 0:
        np_ = neg.groupby('prov').size().sort_values(ascending=False)
        wg = (np_ / np_.sum()) * a.neg
        nq = {p: int(min(round(w), np_[p])) for p, w in wg.items()}
        negch, used2 = [], list(used)
        for prov, q in nq.items():
            sub = neg[neg['prov'] == prov]
            if len(sub) == 0 or q <= 0:
                continue
            sub = sub.iloc[rs.permutation(len(sub))]
            take, bc = [], {}
            for r in sub.itertuples():
                if len(take) >= q:
                    break
                if bc.get(r.box, 0) >= a.box_cap:
                    continue
                if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (MIN_KM / 111.0) ** 2
                       for u in used2[-1500:]):
                    continue
                used2.append((r.lon, r.lat)); bc[r.box] = bc.get(r.box, 0) + 1
                take.append(r.Index)
            sub2 = neg.loc[take]
            keep_full = sub2[sub2['ratio'] >= 1.0]
            keep_half = sub2[sub2['ratio'] < 1.0]
            n_half = int(round(len(keep_half) * 0.5))
            negch.append(pd.concat([keep_full, keep_half.sample(n=min(n_half, len(keep_half)),
                                                               random_state=rs)]))
        negdf = pd.concat(negch, ignore_index=True) if negch else pd.DataFrame()
    if len(posdf):
        posdf['sample_role'] = 'shrub_pos'
    if len(negdf):
        negdf['sample_role'] = 'hard_neg'
    allp = pd.concat([posdf, negdf], ignore_index=True) if len(posdf) or len(negdf) else pd.DataFrame()
    allp = allp.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    allp.insert(0, 'point_id', ['SN-%s-%06d' % ('P' if r.sample_role == 'shrub_pos' else 'N', i)
                                for i, r in enumerate(allp.itertuples())])
    allp.to_csv(os.path.join(OUTD, 'mass_points.csv'), index=False, encoding='utf-8-sig')
    plan = dict(target=a.target, neg=a.neg, prov_cap=a.prov_cap, box_cap=a.box_cap,
                n_pos=int(len(posdf)), n_neg=int(len(negdf)), n_all=int(len(allp)),
                pos_by_prov={p: int(n) for p, n in posdf.groupby('prov').size().items()} if len(posdf) else {},
                pos_by_zone={z: int(n) for z, n in posdf.groupby('zone').size().items()} if len(posdf) else {},
                pos_by_k={k: int(n) for k, n in posdf.groupby('k').size().items()} if len(posdf) else {},
                neg_by_prov={p: int(n) for p, n in negdf.groupby('prov').size().items()} if len(negdf) else {})
    json.dump(plan, open(os.path.join(OUTD, 'mass_plan.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ 正样本 %d + 硬负 %d = %d 点' % (len(posdf), len(negdf), len(allp)))


if __name__ == '__main__':
    main()
