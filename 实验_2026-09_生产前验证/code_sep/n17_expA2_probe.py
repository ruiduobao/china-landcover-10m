# -*- coding: utf-8 -*-
"""n17_expA2_probe.py — 实验 A 第二轮：针/阔可分性（合并两轮判读样本）+ 物候标签常绿/落叶探针

* 输入：data/m3/expA_judged.csv（第一轮 190 点）+ expA2_judged.csv（第二轮 188 点，FCS10 森林层分层选点）
* 规则源（**预注册，先写死再跑**）：
  样本合并：两轮均取 Q1=是 的森林点（判读林型为五类之一），特征列取两表共有（AEF64 + S2四季 + S1 + DEM）。
  T2a 针 vs 阔（常绿组内）：常绿针叶林=1 vs 常绿阔叶林=0（两轮合并）；
  T2b 针 vs 阔（落叶组内）：落叶针叶林=1 vs 落叶阔叶林=0；
  T1p 物候标签常绿/落叶（**物理标签**，解单期底图判不出物候的问题）：
      amp = ndvi_jja − ndvi_djf；常绿 amp<0.15；落叶 amp>0.30；其余剔除；仅森林点（Q1=是）。
 特征集：F1=AEF64（主报，无循环性）；F2=AEF+S2；F3=F2+S1；F4=F3+DEM。
  判据（沿用第一轮预注册梯子，按可得任务执行）：
    若 T1p(AEF) AUC ≥0.85 → "常绿/落叶"轴可分；
    若 T2a AUC ≥0.85（且 T2b 可得并 ≥0.85）→ 针/阔轴亦可得证；
    综合：两轴均 ≥0.85 → 支持 4 类；仅常绿/落叶 → 2 类；均 <0.85 → 1 类。
  小样本约束：任一任务任一类 <15 点 → 该任务只报分布。
  旁证：判读林型 × FCS10 选点分层交叉表（评估 FCS10 森林层在本样本的"眼见为林"率）。
* 输出：results/d2/expA2_probe.json
* 用法：python n17_expA2_probe.py
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
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
S2B = ['%s_%s' % (k, s) for s in ['djf', 'mam', 'jja', 'son'] for k in ('ndvi', 'ndwi', 'mndwi')]
FSETS = {'F1_AEF': VC.FEATS, 'F2_AEF+S2': VC.FEATS + S2B,
         'F3_AEF+S2+S1': VC.FEATS + S2B + S1B, 'F4_F3+DEM': VC.FEATS + S2B + S1B + ['dem', 'slope']}
EB, DB, EN, DN, MX = '常绿阔叶林', '落叶阔叶林', '常绿针叶林', '落叶针叶林', '针阔混交林'
FOREST = [EB, DB, EN, DN, MX]


def load():
    a = pd.read_csv(os.path.join(M3D, 'expA_judged.csv'), encoding='utf-8-sig')
    b = pd.read_csv(os.path.join(M3D, 'expA2_judged.csv'), encoding='utf-8-sig')
    a['round'] = 1
    b['round'] = 2
    cols = [c for c in a.columns if c in b.columns]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    return d


def run(name, df, pm, nm, out):
    npos, nneg = int(pm.sum()), int(nm.sum())
    print('\n== %s ==  正 %d / 负 %d' % (name, npos, nneg))
    if npos < 15 or nneg < 15:
        out[name] = {'n_pos': npos, 'n_neg': nneg, 'verdict': '样本不足（<15/类）'}
        print('  样本不足')
        return None
    res, sub, y = pair_probe(df, {k: [c for c in v if c in df.columns] for k, v in FSETS.items()}, pm, nm)
    for k, m in res.items():
        print('  %-14s AUC=%.3f F1=%.3f (n=%d)' % (k, m['AUC'], m['F1'], m['n']))
    best = max(res.items(), key=lambda kv: kv[1]['AUC'])
    out[name] = {'n_pos': npos, 'n_neg': nneg, 'results': res, 'best': best[0],
                 'best_AUC': best[1]['AUC'], 'best_F1': best[1]['F1'],
                 'AEF_AUC': res['F1_AEF']['AUC']}
    return best[1]['AUC']


def main():
    d = load()
    f = d[(d.Q1 == '是') & d.Q2.isin(FOREST)].copy()
    print('合并判读点 %d；森林点 %d' % (len(d), len(f)))
    print('林型分布（两轮）:', f.Q2.value_counts().to_dict())
    out = {'n_forest': int(len(f)), 'type_dist': f.Q2.value_counts().to_dict(), 'tasks': {}, 'verdict': ''}
    # 旁证：判读 × FCS10 分层
    if 'stratum' in f.columns:
        ct = pd.crosstab(f['stratum'].fillna('(round1)'), f.Q2)
        out['judged_x_stratum'] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict('index').items()}
        print('\n判读林型 × 选点分层：')
        print(ct.to_string())
    # T2a / T2b
    a2a = run('T2a_针vs阔_常绿', f, f.Q2 == EN, f.Q2 == EB, out['tasks'])
    a2b = run('T2b_针vs阔_落叶', f, f.Q2 == DN, f.Q2 == DB, out['tasks'])
    # T1p 物候标签
    fj = f[f['ndvi_jja'].notna() & f['ndvi_djf'].notna()].copy()
    fj['amp'] = fj['ndvi_jja'] - fj['ndvi_djf']
    fj['pheno'] = np.where(fj['amp'] < 0.15, '常绿', np.where(fj['amp'] > 0.30, '落叶', '模糊'))
    ct = pd.crosstab(fj.Q2, fj.pheno)
    out['pheno_x_type'] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict('index').items()}
    print('\n判读林型 × 物候类（amp 阈值 0.15/0.30）：')
    print(ct.to_string())
    t1p = run('T1p_物候常绿vs落叶', fj[fj.pheno.isin(['常绿', '落叶'])],
              fj.pheno == '常绿', fj.pheno == '落叶', out['tasks'])
    out['T1p_by_region'] = {str(k): int(v) for k, v in fj.groupby(['region', 'pheno']).size().to_dict().items()} if 'region' in fj.columns else {}
    # 裁决（沿用预注册梯子）
    if t1p is None:
        out['verdict'] = '物候样本不足，无法裁决'
    elif t1p >= 0.85 and a2a is not None and a2a >= 0.85 and a2b is not None and a2b >= 0.85:
        out['verdict'] = '两轴均 ≥0.85 → 支持森林 4 类'
    elif t1p >= 0.85:
        out['verdict'] = '仅常绿/落叶轴 ≥0.85（针/阔不足）→ 森林 2 类（常绿林/落叶林）'
    else:
        out['verdict'] = '常绿/落叶轴亦 <0.85 → 森林 1 类'
    print('\n判定：%s' % out['verdict'])
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'expA2_probe.json'))


if __name__ == '__main__':
    main()
