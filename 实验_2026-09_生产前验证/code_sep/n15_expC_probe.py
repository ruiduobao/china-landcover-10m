# -*- coding: utf-8 -*-
"""n15_expC_probe.py — 实验 C：湿地细分可分性探针（三江 403 仲裁点，预注册判据）

* 输入：data/m3/expC_judged.csv（子代理湿地细分真值 + AEF/S2/S1/DEM 特征）
* 规则源（**预注册，先写死再跑**）：
  关键对（每对用 5 折分层 CV，RF(300, balanced_subsample)，seed=7；样本 <15/类 的对只报分布不判）：
    P1 草本沼泽 vs 草地；P2 木本沼泽 vs 森林（任一林中类）；P3 湖河滩地 vs 裸地（裸地/沙地）。
  特征集：F1=AEF64；F2=AEF+S2四季；F3=F2+S1冬夏；F4=F3+DEM/坡度。每对取最佳特征集 AUC。
  判据：**三对 AUC 均 ≥0.85 → 保留湿地细分；否则 → 湿地只报合并类（湿地 1–2 类）**。
  附加对（不改判据，供解读）：草本沼泽 vs 水田、水体 vs 草本沼泽。
* 输出：results/d2/expC_probe.json
* 用法：python n15_expC_probe.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from n12_probe_lib import pair_probe

M3D = r'F:/lc_work/v31_exp/data/m3'
JUDGED = os.path.join(M3D, 'expC_judged.csv')
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
S2B = ['%s_%s' % (k, s) for s in ['djf', 'mam', 'jja', 'son'] for k in ('ndvi', 'ndwi', 'mndwi')]
FSETS = {'F1_AEF': VC.FEATS, 'F2_AEF+S2': VC.FEATS + S2B,
         'F3_AEF+S2+S1': VC.FEATS + S2B + S1B, 'F4_F3+DEM': VC.FEATS + S2B + S1B + ['dem', 'slope']}
FOREST = ['常绿阔叶林', '落叶阔叶林', '常绿针叶林', '落叶针叶林', '针阔混交林']
PAIRS = {
    'P1_草本沼泽vs草地': ('草本沼泽', ['草地']),
    'P2_木本沼泽vs森林': ('木本沼泽', FOREST),
    'P3_湖河滩地vs裸地': ('湖河滩地', ['裸地', '沙地']),
}
EXTRA = {
    'E1_草本沼泽vs水田': ('草本沼泽', ['水田']),
    'E2_水体vs草本沼泽': ('水体', ['草本沼泽']),
}


def main():
    df = pd.read_csv(JUDGED, encoding='utf-8-sig')
    df = df[df.Q1.notna()].copy()
    ok = df[VC.FEATS].notna().all(axis=1)
    print('判读点 %d（有效特征 %d）；Q2 分布 top15：' % (len(df), int(ok.sum())))
    print(df.Q2.value_counts().head(15).to_string())
    out = {'n_judged': int(len(df)), 'q2_dist': df.Q2.value_counts().to_dict(), 'pairs': {}, 'extra': {}, 'verdict': ''}
    aucs = {}
    for name, (pos, negs) in {**PAIRS, **EXTRA}.items():
        pm = df.Q2 == pos
        nm = df.Q2.isin(negs)
        npos, nneg = int(pm.sum()), int(nm.sum())
        print('\n== %s ==  正 %d / 负 %d' % (name, npos, nneg))
        tgt = out['pairs'] if name in PAIRS else out['extra']
        if npos < 15 or nneg < 15:
            tgt[name] = {'n_pos': npos, 'n_neg': nneg, 'verdict': '样本不足（<15/类）'}
            print('  样本不足')
            continue
        res, sub, y = pair_probe(df[ok], FSETS, pm[ok], nm[ok])
        for k, m in res.items():
            print('  %-14s AUC=%.3f F1=%.3f (n=%d 正=%d)' % (k, m['AUC'], m['F1'], m['n'], m['pos']))
        best = max(res.items(), key=lambda kv: kv[1]['AUC'])
        tgt[name] = {'n_pos': npos, 'n_neg': nneg, 'results': res, 'best': best[0],
                     'best_AUC': best[1]['AUC'], 'best_F1': best[1]['F1']}
        if name in PAIRS:
            aucs[name] = best[1]['AUC']
    # 预注册裁决
    if len(aucs) == len(PAIRS) and all(a >= 0.85 for a in aucs.values()):
        out['verdict'] = '三对均 ≥0.85 → 保留湿地细分'
    elif aucs:
        out['verdict'] = '三对未全部 ≥0.85（%s）→ 湿地只报合并类（湿地 1–2 类）' % (
            ', '.join('%s=%.3f' % (k.split('_')[0], a) for k, a in aucs.items()))
    else:
        out['verdict'] = '样本不足，无法裁决 → 保守走合并口径'
    print('\n判定：%s' % out['verdict'])
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'expC_probe.json'))


if __name__ == '__main__':
    main()
