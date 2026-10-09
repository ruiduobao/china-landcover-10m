# -*- coding: utf-8 -*-
"""f1_refset.py — 统一评估参考集组装 + 考察池训练点类别分布

* 输入：data/m3/exp{A,A2,B,C,C2,C3,C4}_judged.csv（1,109 点，含 82 维特征与 truth_code）
* 规则源：本文件头部（预注册）——只做"组装与清点"，不做任何筛选决策
        · 参考集 = 全部 truth_code 非空且 conf 有效的判读点；0 视为无效不参与
        · 三评估区（来自判读点 bbox）：森林=全国、湿润灌丛=南方(101–114°E,23–29°N)、湿地=三江(129–135°E,42–50°N)
* 门槛：每点须有 AEF64 完整；S1/S2/DEM 可缺（记录缺失率）
* 输出：data/m3/refset_all.csv（统一评估集）、data/m3/refset_plan.json（区域/类别清点）
* 用法：python f1_refset.py
"""
import collections
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
OUT = os.path.join(M3, 'refset_all.csv')
PLAN = os.path.join(M3, 'refset_plan.json')
NAMES = VC.V31_NAMES()
GROUPS = {
    '森林': ['expA_judged.csv', 'expA2_judged.csv'],
    '湿润灌丛': ['expB_judged.csv'],
    '湿地': ['expC_judged.csv', 'expC2_judged.csv', 'expC3_judged.csv', 'expC4_judged.csv'],
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    rows = []
    for g, files in GROUPS.items():
        for f in files:
            fp = os.path.join(M3, f)
            if not os.path.exists(fp):
                emit('缺 %s' % fp)
                continue
            d = pd.read_csv(fp)
            d['src_exp'] = f.replace('_judged.csv', '')
            d['eval_group'] = g
            rows.append(d)
    allx = pd.concat(rows, ignore_index=True, sort=False)
    # 有效性：truth_code 非空且 >0
    tc = pd.to_numeric(allx['truth_code'], errors='coerce')
    allx['truth_code'] = tc
    valid = allx[tc.notna() & (tc > 0)].copy()
    valid['truth_code'] = valid['truth_code'].astype(int)
    aef_ok = valid[[f for f in VC.FEATS]].notna().all(axis=1)
    emit('合并 %d 行 → 有效判读 %d 点（AEF 完整 %d）' % (len(allx), len(valid), int(aef_ok.sum())))
    valid = valid[aef_ok].copy()
    # 去重（同一点可能出现在多批）
    before = len(valid)
    valid = valid.drop_duplicates(subset=['point_id'], keep='first')
    if len(valid) != before:
        emit('按 point_id 去重：%d → %d' % (before, len(valid)))
    valid.to_csv(OUT, index=False, encoding='utf-8-sig')
    # 清点
    plan = {'n_points': int(len(valid)), 'groups': {}, 'coverage': {}}
    for g, sub in valid.groupby('eval_group'):
        plan['groups'][g] = dict(n=int(len(sub)),
                                 bbox=[round(float(sub.lon.min()), 3), round(float(sub.lat.min()), 3),
                                       round(float(sub.lon.max()), 3), round(float(sub.lat.max()), 3)],
                                 classes={NAMES.get(str(c), str(c)): int(n)
                                          for c, n in collections.Counter(sub.truth_code).most_common()})
    s1b = [b for b in ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s'] if b in valid.columns]
    s2b = [c for c in valid.columns if c.startswith(('ndvi_', 'ndwi_', 'mndwi_'))]
    demb = [c for c in ('dem', 'slope') if c in valid.columns]
    plan['coverage'] = dict(s1=round(float(valid[s1b].notna().all(axis=1).mean()), 4) if s1b else 0,
                            s2=round(float(valid[s2b].notna().all(axis=1).mean()), 4) if s2b else 0,
                            dem=round(float(valid[demb].notna().all(axis=1).mean()), 4) if demb else 0,
                            s1_cols=s1b, n_s2=len(s2b), dem_cols=demb)
    VC.jsave(plan, PLAN)
    emit('参考集 %d 点 → %s' % (len(valid), OUT))
    for g, v in plan['groups'].items():
        emit('  %-6s n=%4d bbox=%s' % (g, v['n'], v['bbox']))
        emit('        类别：%s' % dict(list(v['classes'].items())[:8]))
    emit('特征覆盖：%s' % plan['coverage'])


if __name__ == '__main__':
    main()
