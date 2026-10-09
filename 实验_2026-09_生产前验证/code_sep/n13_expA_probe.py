# -*- coding: utf-8 -*-
"""n13_expA_probe.py — 实验 A：森林"常绿/落叶 × 针叶/阔叶"可分性探针（预注册判据）

* 输入：data/m3/expA_judged.csv（子代理判读的林型真值 + AEF/S2/S1/DEM 特征）
* 规则源（**预注册，先写死再跑**）：
  样本：Q1=是 且 Q2 为五类林型之一的点（Q1=无法判读/非森林点排除）。
  任务：
    T1 常绿 vs 落叶：{常绿阔叶林, 常绿针叶林}=1 vs {落叶阔叶林, 落叶针叶林}=0（针阔混交林点不参与 T1，单列报告）；
    T2a 针 vs 阔（常绿组内）：常绿针叶林=1 vs 常绿阔叶林=0；
    T2b 针 vs 阔（落叶组内）：落叶针叶林=1 vs 落叶阔叶林=0。
  特征集：F1=AEF64；F2=AEF+S2四季(12)；F3=F2+S1冬夏(4)；F4=F3+DEM/坡度(2)。
  模型：5 折分层 CV，RF(300, balanced_subsample)，seed=7。
  判据（任一任务用其**最佳特征集**的 AUC）：
    常绿vs落叶 AUC ≥0.85 且 T2a/T2b 两任务 AUC 均 ≥0.85 → 保留 4 类（混交林并入多数类）；
    仅"常绿vs落叶" ≥0.85（T2 有任一 <0.85）→ 森林 2 类（常绿林/落叶林）；
    两者均 <0.85 → 森林 1 类。
  小样本约束：任一任务 n<30 或某一类 <15 点 → 该任务只报"样本不足"，判据按可得任务执行并在结论标注。
* 输出：results/d2/expA_probe.json + 控制台 AUC/F1 表
* 用法：python n13_expA_probe.py

补充分析 A2/A3（2026-10-09 预注册于本轮执行前；用于标注 A1 判读标签的物候核验）：
  A3 标签质量核验：把 A1 五个视觉林型与**物理物候类**交叉——物候类由 S2 四季 NDVI 振幅定义：
     振幅 = ndvi_jja − ndvi_djf；常绿 <0.15；落叶 >0.30；其余"模糊"（用于识别单期底图误判常绿/落叶）；
  A2 物候标签探针：仅取 Q1=是 的森林点，标签=物候类（常绿/落叶，模糊剔除），特征集同上。
     说明：S2 含在特征集内时对物候标签存在循环性，故 **以 F1(AEF) 为主报**，其余仅附。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from n12_probe_lib import pair_probe

M3D = r'F:/lc_work/v31_exp/data/m3'
JUDGED = os.path.join(M3D, 'expA_judged.csv')
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
SEAS = ['djf', 'mam', 'jja', 'son']
S2B = ['%s_%s' % (k, s) for s in SEAS for k in ('ndvi', 'ndwi', 'mndwi')]
FSETS = {'F1_AEF': VC.FEATS,
         'F2_AEF+S2': VC.FEATS + S2B,
         'F3_AEF+S2+S1': VC.FEATS + S2B + S1B,
         'F4_F3+DEM': VC.FEATS + S2B + S1B + ['dem', 'slope']}
EB, DB = '常绿阔叶林', '落叶阔叶林'
EN, DN, MX = '常绿针叶林', '落叶针叶林', '针阔混交林'


def main():
    df = pd.read_csv(JUDGED, encoding='utf-8-sig')
    df = df[df.Q1.notna()].copy()
    forest = df[df.Q2.isin([EB, DB, EN, DN, MX])].copy()
    F = [c for c in VC.FEATS + S2B + S1B + ['dem', 'slope'] if c in forest.columns]
    forest = forest[forest[VC.FEATS].notna().all(axis=1)].copy()
    print('判读点 %d；森林点 %d（五类林型）' % (df.Q1.notna().sum(), len(forest)))
    print('林型分布：', forest.Q2.value_counts().to_dict())

    out = {'n_forest': int(len(forest)), 'type_dist': forest.Q2.value_counts().to_dict(),
           'tasks': {}, 'verdict': ''}
    tasks = {
        'T1_常绿vs落叶': (forest.Q2.isin([EB, EN]), forest.Q2.isin([DB, DN])),
        'T2a_针vs阔_常绿': (forest.Q2 == EN, forest.Q2 == EB),
        'T2b_针vs阔_落叶': (forest.Q2 == DN, forest.Q2 == DB),
    }
    aucs = {}
    for name, (pm, nm) in tasks.items():
        npos, nneg = int(pm.sum()), int(nm.sum())
        print('\n== %s ==  正 %d / 负 %d' % (name, npos, nneg))
        if npos < 15 or nneg < 15:
            out['tasks'][name] = {'n_pos': npos, 'n_neg': nneg, 'verdict': '样本不足（<15/类），只报分布'}
            print('  样本不足：正 %d / 负 %d' % (npos, nneg))
            continue
        res, sub, y = pair_probe(forest, {k: [c for c in v if c in F] for k, v in FSETS.items()}, pm, nm)
        for k, m in res.items():
            print('  %-14s AUC=%.3f F1=%.3f (n=%d, 正=%d)' % (k, m['AUC'], m['F1'], m['n'], m['pos']))
        best = max(res.items(), key=lambda kv: kv[1]['AUC'])
        aucs[name] = best[1]['AUC']
        out['tasks'][name] = {'n_pos': npos, 'n_neg': nneg, 'results': res,
                              'best': best[0], 'best_AUC': best[1]['AUC'], 'best_F1': best[1]['F1']}
    # 判据执行（预注册）
    t1 = aucs.get('T1_常绿vs落叶')
    t2 = [aucs.get('T2a_针vs阔_常绿'), aucs.get('T2b_针vs阔_落叶')]
    t2_known = [a for a in t2 if a is not None]
    if t1 is None:
        out['verdict'] = '样本不足，无法裁决'
    elif t1 >= 0.85 and t2_known and all(a >= 0.85 for a in t2_known) and len(t2_known) == 2:
        out['verdict'] = '森林保留 4 类（常绿/落叶 × 针/阔 均 ≥0.85；混交林并入多数类）'
    elif t1 >= 0.85:
        out['verdict'] = '森林并成 2 类（仅常绿/落叶可分；针阔 ≤0.85）'
    else:
        out['verdict'] = '森林并成 1 类（常绿/落叶亦不可分）'
    print('\n判定：%s' % out['verdict'])
    print('（针阔混交林 %d 点：不参与 T1/T2，建议并入多数类）' % int((forest.Q2 == MX).sum()))
    out['mixed_n'] = int((forest.Q2 == MX).sum())

    # ---- A3 视觉标签 × 物候类 交叉核验 ----
    fj = forest[forest['ndvi_jja'].notna() & forest['ndvi_djf'].notna()].copy()
    fj['amp'] = fj['ndvi_jja'] - fj['ndvi_djf']
    fj['pheno'] = np.where(fj['amp'] < 0.15, '常绿', np.where(fj['amp'] > 0.30, '落叶', '模糊'))
    ct = pd.crosstab(fj.Q2, fj.pheno)
    out['A3_visual_x_pheno'] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict('index').items()}
    conf_n = int(((fj.Q2 == '常绿阔叶林') & (fj.pheno == '落叶')).sum())
    print('\n== A3 视觉"常绿阔叶林"判读数 %d，其中物候为落叶 %d（%.0f%%）==' % (
        int((fj.Q2 == '常绿阔叶林').sum()), conf_n,
        100 * conf_n / max(1, int((fj.Q2 == '常绿阔叶林').sum()))))
    print(ct.to_string())
    out['A3_amp_by_region'] = fj.groupby('region')['amp'].median().round(3).to_dict()

    # ---- A2 物候标签探针（常绿 vs 落叶）----
    sub = fj[fj.pheno.isin(['常绿', '落叶'])]
    pm = sub.pheno == '常绿'
    nm = sub.pheno == '落叶'
    print('\n== A2 物候标签探针：常绿 %d vs 落叶 %d ==' % (int(pm.sum()), int(nm.sum())))
    if int(pm.sum()) >= 15 and int(nm.sum()) >= 15:
        res, sub2, y = pair_probe(fj, {k: [c for c in v if c in F] for k, v in FSETS.items()}, pm, nm)
        for k, m in res.items():
            print('  %-14s AUC=%.3f F1=%.3f (n=%d)' % (k, m['AUC'], m['F1'], m['n']))
        out['A2_pheno_probe'] = res
        b = max(res.items(), key=lambda kv: kv[1]['AUC'])
        out['A2_best'] = {'set': b[0], 'AUC': b[1]['AUC'], 'F1': b[1]['F1'],
                          'F1_only_AEF': res.get('F1_AEF', {}).get('AUC')}
    else:
        out['A2_pheno_probe'] = {'verdict': '样本不足（常绿 %d / 落叶 %d）' % (int(pm.sum()), int(nm.sum()))}
        print('  样本不足')

    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'expA_probe.json'))


if __name__ == '__main__':
    main()
