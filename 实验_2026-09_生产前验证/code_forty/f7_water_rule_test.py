# -*- coding: utf-8 -*-
"""f7_water_rule_test.py — 水体规则判别力检验 + 样本集 v2 策略定标

* 输入：data/eco_gate/_water/water_occ_points.csv（JRC occ_pt / occ_90，6,638 点）
        data/m3/gap_points.csv + gap_imagery/judge_*.csv（判读真值）
        data/m3/refset_all.csv（exp 参考集真值）
        data/m3/layer_precision.json（层精度）
* 规则源（预注册）：对每条"水体邻域/丰度"候选规则，计算在判读真值上的
        **真阳保留率（召回）**与**假阳清除率**；只有当"清除率显著 > 0 且召回 ≥0.7"才采纳
* 门槛：每条规则判读样本 ≥8 才给结论
* 输出：data/m3/water_rule_test.json + .csv
* 用法：python f7_water_rule_test.py
"""
import json
import os
import sys
import time
import csv as csvmod
import collections

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

M3 = os.path.join(WORK, 'data', 'm3')
WATER = os.path.join(WORK, 'data', 'eco_gate', '_water', 'water_occ_points.csv')
OUTJ = os.path.join(M3, 'water_rule_test.json')
OUTC = os.path.join(M3, 'water_rule_test.csv')
NAMES = VC.V31_NAMES()


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    w = pd.read_csv(WATER, encoding='utf-8-sig')
    w['occ_pt'] = pd.to_numeric(w.occ_pt, errors='coerce')
    w['occ_90'] = pd.to_numeric(w.occ_90, errors='coerce')
    W = {r.point_id: (r.occ_pt, r.occ_90) for r in w.itertuples()}
    # 判读真值：gap（池内声明类） + refset（影像真值）
    gp = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    js = [pd.read_csv(os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i), encoding='utf-8-sig')
          for i in range(1, 6) if os.path.exists(os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i))]
    G = gp.merge(pd.concat(js, ignore_index=True), on='point_id')
    R = pd.read_csv(os.path.join(M3, 'refset_all.csv'), encoding='utf-8-sig')
    R['Q1'] = '是'
    R['want'] = R.truth_code.map(lambda c: '%d%s' % (c, NAMES.get(str(int(c)), ''))) if len(R) else ''

    rows, res = [], {}
    for code, nm in ((16, '湖河滩地'), (14, '木本沼泽'), (15, '草本沼泽')):
        a = G[G.want.str.startswith('%d' % code)][['point_id', 'Q1', 'want']]
        b = R[R.truth_code == code][['point_id', 'Q1', 'want']]
        m = pd.concat([a, b], ignore_index=True)
        m['occ_pt'] = [W.get(p, (np.nan, np.nan))[0] for p in m.point_id]
        m['occ_90'] = [W.get(p, (np.nan, np.nan))[1] for p in m.point_id]
        v = m[(m.Q1 != '无法判读') & m.occ_90.notna()]
        yes, no = v[v.Q1 == '是'], v[v.Q1 == '否']
        d = dict(n=len(m), judged=len(v), yes=len(yes), no=len(no),
                 occ90_yes_med=round(float(yes.occ_90.median()), 1) if len(yes) else None,
                 occ90_no_med=round(float(no.occ_90.median()), 1) if len(no) else None,
                 occpt_yes_med=round(float(yes.occ_pt.median()), 1) if len(yes) else None,
                 occpt_no_med=round(float(no.occ_pt.median()), 1) if len(no) else None)
        res[nm] = d
        emit('%s：真 n=%d occ_pt中位 %s / occ_90中位 %s ｜ 假 n=%d occ_pt中位 %s / occ_90中位 %s' % (
            nm, d['yes'], d['occpt_yes_med'], d['occ90_yes_med'],
            d['no'], d['occpt_no_med'], d['occ90_no_med']))
        # 阈值扫描：occ_pt（点自身水频率）作为"是水/不是滩地"的判据
        for thr in (5, 20, 50, 80):
            kept_yes = int((yes.occ_pt <= thr).sum()) if len(yes) else 0
            kept_no = int((no.occ_pt <= thr).sum()) if len(no) else 0
            r = dict(rule='%s: occ_pt<=%d（判为非法：点在常年水中）' % (nm, thr), thr=thr,
                     n_yes=len(yes), keep_yes=kept_yes, n_no=len(no), keep_no=kept_no,
                     recall=round(kept_yes / max(1, len(yes)), 3),
                     false_removed=round((len(no) - kept_no) / max(1, len(no)), 3),
                     final_prec=round(kept_yes / max(1, kept_yes + kept_no), 3))
            rows.append(r)
    VC.jsave(res, OUTJ)
    with open(OUTC, 'w', encoding='utf-8-sig', newline='') as f:
        wtr = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    emit('=== occ_pt 阈值扫描（"点在常年水中→非滩地"判据）===')
    for r in rows:
        emit('  %-34s 召回 %.2f ｜ 假点清除 %.2f ｜ 净化后精度 %.2f' % (
            r['rule'], r['recall'], r['false_removed'], r['final_prec']))
    emit('→ %s' % OUTC)


if __name__ == '__main__':
    main()
