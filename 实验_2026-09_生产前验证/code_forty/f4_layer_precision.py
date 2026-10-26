# -*- coding: utf-8 -*-
"""f4_layer_precision.py — 训练池"层精度"汇总（判读真值 → 各来源层的可信度）

* 输入：gap_imagery/judge_{1..5}.csv（255 缺口点判读）、gap_points.csv（池内声明类）
        refset_all.csv（exp 系列 1,022 点判读，含 truth_code 与来源层信息）
        gsnx_ref_judged.csv（甘宁 304 点，Q1=是否灌丛）
* 规则源：任务书——"先全部过一遍已有全国的样本，然后筛选一遍用生态规则，要保证样本尽量真实"
* 门槛：只统计有效判读（排除"无法判读"）；判"是"= 池内声明类与影像一致；判"否"= 不一致
* 输出：data/m3/layer_precision.json + layer_precision.csv
* 用法：python f4_layer_precision.py
"""
import collections
import csv
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

M3 = os.path.join(WORK, 'data', 'm3')
OUTJ = os.path.join(M3, 'layer_precision.json')
OUTC = os.path.join(M3, 'layer_precision.csv')
NAMES = VC.V31_NAMES()


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    res = {}
    rows = []
    # ---- A. 缺口点（255）----
    gp = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
    j = []
    for i in range(1, 6):
        fp = os.path.join(M3, 'gap_imagery', 'judge_%d.csv' % i)
        if os.path.exists(fp):
            j.append(pd.read_csv(fp, encoding='utf-8-sig'))
    J = pd.concat(j, ignore_index=True) if j else pd.DataFrame()
    emit('缺口点判读 %d 行' % len(J))
    G = gp.merge(J, on='point_id', how='inner')
    emit('缺口点合并 %d / %d' % (len(G), len(gp)))
    for (tag, want), sub in G.groupby(['tag', 'want']):
        v = sub[sub.Q1 != '无法判读']
        n = len(sub)
        res.setdefault('gap', {})['%s|%s' % (tag, want)] = dict(
            n=n, judged=len(v), yes=int((v.Q1 == '是').sum()), no=int((v.Q1 == '否').sum()),
            unknown=int((sub.Q1 == '无法判读').sum()),
            precision=round(float((v.Q1 == '是').mean()), 3) if len(v) else None,
            top_no=dict(collections.Counter(v[v.Q1 == '否'].Q2).most_common(4)))
        r = res['gap']['%s|%s' % (tag, want)]
        rows.append(dict(scope='gap', layer='%s|%s' % (tag, want), n=r['n'], judged=r['judged'],
                         yes=r['yes'], no=r['no'], unknown=r['unknown'],
                         precision=r['precision'], top_no=str(r['top_no'])))

    # ---- B. exp 参考集（1,022）----
    R = pd.read_csv(os.path.join(M3, 'refset_all.csv'), encoding='utf-8-sig')
    emit('参考集 %d 点' % len(R))
    for grp, sub in R.groupby('eval_group'):
        v = sub[sub.truth_code > 0]
        res.setdefault('refset', {})[grp] = dict(
            n=int(len(v)),
            classes={NAMES.get(str(c), str(c)): int(n) for c, n in
                     collections.Counter(v.truth_code).most_common()})
        rows.append(dict(scope='refset', layer=grp, n=int(len(v)), judged=int(len(v)),
                         yes=int(len(v)), no=0, unknown=0, precision=None,
                         top_no=str(res['refset'][grp]['classes'])))

    # ---- C. 甘宁干旱灌丛（304，Q1=是否灌丛）----
    gs = pd.read_csv(os.path.join(M3, 'gsnx_ref_judged.csv'), encoding='utf-8-sig')
    v = gs[gs.Q1 != '无法判读']
    # 该集抽样自 FCS10 灌丛层（w1/w3/w4 窗）与底座
    res['dryshrub'] = dict(n=int(len(gs)), judged=int(len(v)),
                           yes=int((v.Q1 == '是').sum()), no=int((v.Q1 == '否').sum()),
                           precision=round(float((v.Q1 == '是').mean()), 3),
                           note='甘宁 304 点：FCS10 灌丛层 + 底层对照；Q1=影像上是否真灌丛')
    rows.append(dict(scope='dryshrub', layer='FCS10灌丛层(甘宁)', n=res['dryshrub']['n'],
                     judged=res['dryshrub']['judged'], yes=res['dryshrub']['yes'],
                     no=res['dryshrub']['no'], unknown=int((gs.Q1 == '无法判读').sum()),
                     precision=res['dryshrub']['precision'], top_no=''))

    VC.jsave(res, OUTJ)
    with open(OUTC, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scope', 'layer', 'n', 'judged', 'yes', 'no', 'unknown',
                                          'precision', 'top_no'])
        w.writeheader()
        w.writerows(rows)
    emit('=== 层精度（判"是"率；排除无法判读）===')
    for k, v in res.get('gap', {}).items():
        emit('  %-26s n=%3d 判=%3d 是=%2d 否=%2d → 精度 %s' % (
            k, v['n'], v['judged'], v['yes'], v['no'], v['precision']))
    emit('  甘宁干旱灌丛层：n=%d 判=%d 是=%d → 精度 %s' % (
        res['dryshrub']['n'], res['dryshrub']['judged'], res['dryshrub']['yes'],
        res['dryshrub']['precision']))
    emit('→ %s' % OUTJ)


if __name__ == '__main__':
    main()
