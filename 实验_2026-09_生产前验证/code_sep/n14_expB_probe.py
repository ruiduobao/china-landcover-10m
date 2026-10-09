# -*- coding: utf-8 -*-
"""n14_expB_probe.py — 实验 B：湿润区（南方丘陵 w3 + 云贵 w7）灌丛真伪复核与可分性探针（预注册判据）

* 输入：data/m3/expB_judged.csv（子代理灌丛判读真值 + AEF/S2/S1/DEM 特征）
* 规则源（**预注册，先写死再跑**）：
  步骤 1 真灌丛率：valid 点（Q1∈{是,否}）中 Q1=是 的比例 → 总体 + 分 zone（w3/w7）+ 分来源（m3a/fcs10layer*/wc20）。
  步骤 2 可分性探针：二分类"灌丛(Q1=是) vs 其它(Q1=否)"，5 折分层 CV，RF(300, balanced_subsample)，seed=7；
    特征集 F1=AEF64；F2=AEF+S2四季；F3=F2+S1；F4=F3+DEM。
  判据（预注册）：**真灌丛率 ≥0.60 且 最佳 AUC ≥0.85 且 F1 ≥0.55 → 湿润区灌丛可分、支持分区处置（湿润保留/干旱删除）；
    否则 → 全国统一不输出灌丛类（维持 R4 现状）**。
  分层旁证（不改判据）：wc20 来源点真灌丛率＝WorldCover 灌丛层在南方的实测精度；
    fcs10layer 来源点真灌丛率＝FCS10 灌丛层在南方是否真。
* 输出：results/d2/expB_probe.json
* 用法：python n14_expB_probe.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from n12_probe_lib import cv_eval

M3D = r'F:/lc_work/v31_exp/data/m3'
JUDGED = os.path.join(M3D, 'expB_judged.csv')
S1B = ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
S2B = ['%s_%s' % (k, s) for s in ['djf', 'mam', 'jja', 'son'] for k in ('ndvi', 'ndwi', 'mndwi')]
FSETS = {'F1_AEF': VC.FEATS, 'F2_AEF+S2': VC.FEATS + S2B,
         'F3_AEF+S2+S1': VC.FEATS + S2B + S1B, 'F4_F3+DEM': VC.FEATS + S2B + S1B + ['dem', 'slope']}


def rate(df):
    if not len(df):
        return None
    return dict(n=int(len(df)), yes=int((df.Q1 == '是').sum()),
                rate=round(float((df.Q1 == '是').mean()), 3))


def main():
    df = pd.read_csv(JUDGED, encoding='utf-8-sig')
    v = df[df.Q1.isin(['是', '否'])].copy()
    print('判读点 %d；valid %d；无法判读 %d' % (df.Q1.notna().sum(), len(v), int((df.Q1 == '无法判读').sum())))
    out = {'n_valid': int(len(v)), 'overall': rate(v),
           'by_zone': {z: rate(v[v.zone == z]) for z in sorted(v.zone.dropna().unique())},
           'by_src': {s: rate(v[v.src == s]) for s in sorted(v.src.dropna().unique())},
           'probe': {}, 'verdict': ''}
    print('真灌丛率（总）: %s' % out['overall'])
    for k, r in out['by_zone'].items():
        print('  zone %s: %s' % (k, r))
    for k, r in out['by_src'].items():
        print('  src  %s: %s' % (k, r))

    # 探针
    X0 = v[v[VC.FEATS].notna().all(axis=1)].copy()
    y = (X0.Q1 == '是').astype(int).to_numpy()
    print('\n探针样本 %d（灌丛 %d / 其它 %d）' % (len(X0), y.sum(), len(y) - y.sum()))
    for name, cols in FSETS.items():
        cols = [c for c in cols if c in X0.columns]
        X = np.nan_to_num(X0[cols].to_numpy('float32'), nan=-999)
        m, _ = cv_eval(X, y)
        if m:
            out['probe'][name] = m
            print('  %-14s AUC=%.3f F1=%.3f' % (name, m['AUC'], m['F1']))
    best = max(out['probe'].items(), key=lambda kv: kv[1]['AUC']) if out['probe'] else None
    r_all = out['overall']['rate'] if out['overall'] else 0
    if best and r_all >= 0.60 and best[1]['AUC'] >= 0.85 and best[1]['F1'] >= 0.55:
        out['verdict'] = ('湿润区灌丛可分（真灌丛率 %.2f，%s AUC %.3f/F1 %.3f）→ 支持分区处置' % (
            r_all, best[0], best[1]['AUC'], best[1]['F1']))
    else:
        out['verdict'] = ('不支持分区灌丛：真灌丛率 %.2f，最佳 %s AUC/F1=%s → 全国统一不输出灌丛类' % (
            r_all, best[0] if best else '—',
            '%.3f/%.3f' % (best[1]['AUC'], best[1]['F1']) if best else '—'))
    print('\n判定：%s' % out['verdict'])
    os.makedirs(os.path.join(VC.RES, 'd2'), exist_ok=True)
    VC.jsave(out, os.path.join(VC.RES, 'd2', 'expB_probe.json'))


if __name__ == '__main__':
    main()
