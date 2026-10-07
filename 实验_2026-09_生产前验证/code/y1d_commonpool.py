# -*- coding: utf-8 -*-
"""y1d_commonpool.py — 决定性诊断：年度差异来自"训练池多出来的点"还是"该年嵌入本身"？
设计：只用 4 年共同存在的训练点（同点、同标签、仅特征随年份变）训练 → 评估共同留出点（同年特征）
  · 若精度回到 2017 水平 → 差异来自"池中多出的点"（可用挑样本修）
  · 若仍低 → 差异来自该年 AEF 嵌入本身（只能用该方法/该年样本）
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4']
OUTD = os.path.join(VC.DATA, 'yearly')


def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                  max_features='sqrt', class_weight=None, n_jobs=15, random_state=7)


def main():
    plan = VC.jload(os.path.join(OUTD, 'plan.json'))
    L = ['# 训练池 vs 嵌入年：共同训练点诊断 %s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '| 窗 | 口径 | 2017 | 2020 | 2023 | 2024 |', '|---|---|---|---|---|---|']
    for w in WS:
        common = set(plan['common_eval'][w]['rids'])
        tr = {y: pd.read_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (y, w))) for y in YEARS}
        ev = {y: pd.read_parquet(os.path.join(OUTD, 'eval_%d_%s.parquet' % (y, w))) for y in YEARS}
        # 共同训练点 = 4 年训练表都有的 rid
        sets = [set(d.row_id.tolist()) for d in tr.values()]
        ctr = set.intersection(*sets)
        if len(ctr) < 500:
            VC.emit('%s 共同训练点仅 %d，跳过' % (w, len(ctr))); continue
        ctr = sorted(ctr)
        order = {y: {int(r): i for i, r in enumerate(tr[y].row_id.to_numpy())} for y in YEARS}
        # 评估
        pos = {y: {int(r): i for i, r in enumerate(ev[y].row_id.to_numpy())} for y in YEARS}
        cid = sorted(common & set.intersection(*[set(d.row_id.tolist()) for d in ev.values()]))
        ridx = {y: np.array([pos[y].get(int(r), -1) for r in cid]) for y in YEARS}
        ok = np.all([p >= 0 for p in ridx.values()], axis=0)
        cid = np.array(cid)[ok]
        ridx = {y: ridx[y][ok] for y in YEARS}
        # 标签（静态，用 2017 表）
        t = None
        a, b = {}, {}
        for y in YEARS:
            ty = np.array([order[y][int(r)] for r in ctr])
            Xtr_c = tr[y][VC.FEATS].to_numpy('float32')[ty]
            ytr_c = VC.to_v31(tr[y]['class_new'].to_numpy(int)[ty])
            clf_c = rf().fit(Xtr_c, ytr_c)
            Xev = ev[y][VC.FEATS].to_numpy('float32')[ridx[y]]
            tev = VC.to_v31(ev[y]['class_new'].to_numpy(int)[ridx[y]])
            a[y] = float((clf_c.predict(Xev) == tev).mean())          # 共同池（同点）
            clf_f = rf().fit(tr[y][VC.FEATS].to_numpy('float32'),
                             VC.to_v31(tr[y]['class_new'].to_numpy(int)))
            b[y] = float((clf_f.predict(Xev) == tev).mean())          # 全池（原口径）
        VC.emit('%s 共同训练点 %d ｜ 共同池: %s ｜ 全池: %s' % (
            w, len(ctr), {y: round(a[y], 4) for y in YEARS}, {y: round(b[y], 4) for y in YEARS}))
        L.append('| %s | 共同池(n_tr=%d) | %s |' % (w, len(ctr), ' / '.join('%.4f' % a[y] for y in YEARS)))
        L.append('|  | 全池（原口径） | %s |' % ' / '.join('%.4f' % b[y] for y in YEARS))
    fp = os.path.join(VC.REPT, '训练池诊断_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
