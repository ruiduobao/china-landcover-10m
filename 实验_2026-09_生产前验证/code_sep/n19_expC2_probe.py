# -*- coding: utf-8 -*-
"""n19_expC2_probe.py — 实验 C 第二轮：合并 C+C2（+C3 若就绪）湿地关键对复测（预注册判据沿用）

* 输入：data/m3/expC_judged.csv + expC2_judged.csv（+ expC3_judged.csv 若存在）
* 规则源（**预注册，先写死再跑**，判据同第一轮 n15）：
  关键对：P1 草本沼泽 vs 草地；P2 木本沼泽 vs 森林（任一林中类）；P3 湖河滩地 vs 裸地（裸地/沙地）；
  5 折分层 CV，RF(300, balanced_subsample)，seed=7；特征集 F1=AEF；F2=+S2；F3=+S1；F4=+DEM（最佳特征集 AUC）。
  判据：三对 AUC 均 ≥0.85 → 保留湿地细分；否则 → 湿地只报合并类。样本 <15/类 的对只报分布。
  另报：P1 分 conf（仅 中/高 子集）复测——评估标签噪声影响。
* 输出：results/d2/expC2_probe.json（含两轮合并口径与轮次拆分口径）
* 用法：python n19_expC2_probe.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import v31_common as VC
from n12_probe_lib import pair_probe

M3D = r'F:/lc_work/v31_exp/data/m3'
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
S2B = ['%s_%s' % (k, s) for s in ['djf', 'mam', 'jja', 'son'] for k in ('ndvi', 'ndwi', 'mndwi')]
FSETS = {'F1_AEF': VC.FEATS, 'F2_AEF+S2': VC.FEATS + S2B,
         'F3_AEF+S2+S1': VC.FEATS + S2B + S1B, 'F4_F3+DEM': VC.FEATS + S2B + S1B + ['dem', 'slope']}
FOREST = ['常绿阔叶林', '落叶阔叶林', '常绿针叶林', '落叶针叶林', '针阔混交林']
PAIRS = {'P1_草本沼泽vs草地': ('草本沼泽', ['草地']),
         'P2_木本沼泽vs森林': ('木本沼泽', FOREST),
         'P3_湖河滩地vs裸地': ('湖河滩地', ['裸地', '沙地'])}


def load():
    fr = []
    for i, f in enumerate(['expC_judged.csv', 'expC2_judged.csv', 'expC3_judged.csv'], 1):
        fp = os.path.join(M3D, f)
        if os.path.exists(fp):
            d = pd.read_csv(fp, encoding='utf-8-sig')
            d['round'] = i
            fr.append(d)
    a = fr[0]
    cols = [c for c in a.columns if all(c in x.columns for x in fr)]
    return pd.concat([x[cols] for x in fr], ignore_index=True)


def run(name, df, pm, nm, out):
    npos, nneg = int(pm.sum()), int(nm.sum())
    print('\n== %s ==  正 %d / 负 %d' % (name, npos, nneg))
    if npos < 15 or nneg < 15:
        out[name] = {'n_pos': npos, 'n_neg': nneg, 'verdict': '样本不足（<15/类）'}
        print('  样本不足')
        return None
    ok = df[VC.FEATS].notna().all(axis=1)
    res, sub, y = pair_probe(df[ok], {k: [c for c in v if c in df.columns] for k, v in FSETS.items()},
                             pm[ok], nm[ok])
    for k, m in res.items():
        print('  %-14s AUC=%.3f F1=%.3f (n=%d)' % (k, m['AUC'], m['F1'], m['n']))
    best = max(res.items(), key=lambda kv: kv[1]['AUC'])
    out[name] = {'n_pos': npos, 'n_neg': nneg, 'results': res, 'best': best[0],
                 'best_AUC': best[1]['AUC'], 'best_F1': best[1]['F1'], 'AEF_AUC': res['F1_AEF']['AUC']}
    return best[1]['AUC']


def main():
    d = load()
    v = d[d.Q1.isin(['是', '否'])].copy()
    print('合并判读点 %d（valid %d）；轮次分布 %s' % (len(d), len(v), d['round'].value_counts().to_dict()))
    print('Q2 分布:', v.Q2.value_counts().to_dict())
    out = {'n_valid': int(len(v)), 'rounds': d['round'].value_counts().to_dict(),
           'q2_dist': v.Q2.value_counts().to_dict(), 'pairs': {}, 'verdict': ''}
    aucs = {}
    for name, (pos, negs) in PAIRS.items():
        a = run(name, v, v.Q2 == pos, v.Q2.isin(negs), out['pairs'])
        if a is not None:
            aucs[name] = a
    # P1 高置信子集复测（标签噪声评估）
    vh = v[v.conf.isin(['高', '中'])]
    run('P1_conf中高子集', vh, vh.Q2 == '草本沼泽', vh.Q2 == '草地', out['pairs'])
    if len(aucs) == len(PAIRS) and all(a >= 0.85 for a in aucs.values()):
        out['verdict'] = '三对均 ≥0.85 → 保留湿地细分'
    elif aucs:
        out['verdict'] = '三对未全部 ≥0.85（%s）→ 湿地只报合并类（湿地 1–2 类）' % (
            ', '.join('%s=%.3f' % (k.split('_')[0], a) for k, a in aucs.items()))
    else:
        out['verdict'] = '样本不足，无法裁决 → 保守走合并口径'
    print('\n判定：%s' % out['verdict'])
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'expC2_probe.json'))


if __name__ == '__main__':
    main()
