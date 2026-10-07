# -*- coding: utf-8 -*-
"""y1c_decompose.py — 年度差异归因（本地，秒级）：分离"特征年份效应"与"训练池年份效应"
A 固定模型（2017 池训练）→ 用各年特征评估：只变特征年
B 各年池训练 → 用固定特征（2017）评估：只变训练池年
评估对象：共同集（4 年同点），标签静态（母库单标签），故差异只来自特征/训练池
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
    L = ['# 年度差异归因（A 特征年 / B 训练池年）%s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '| 窗 | 特征年→（模型=2017池）OA | 训练池年→（特征=2017）OA |', '|---|---|---|']
    for w in WS:
        common = set(plan['common_eval'][w]['rids'])
        # 载入各年评估表，取共同集
        ev = {}
        for y in YEARS:
            fp = os.path.join(OUTD, 'eval_%d_%s.parquet' % (y, w))
            d = pd.read_parquet(fp)
            d = d[d.row_id.isin(common)].sort_values('row_id')
            ev[y] = d
        base_ids = ev[2017]['row_id'].to_numpy()
        keep = np.isin(base_ids, [ev[y]['row_id'].to_numpy() for y in YEARS][0])
        # 对齐所有年份到 2017 的 rid 顺序
        idx = {y: {int(r): i for i, r in enumerate(ev[y]['row_id'].to_numpy())} for y in YEARS}
        pos = {y: np.array([idx[y].get(int(r), -1) for r in base_ids]) for y in YEARS}
        ok = np.all([p >= 0 for p in pos.values()], axis=0)
        rid = base_ids[ok]
        t = VC.to_v31(ev[2017]['class_new'].to_numpy(int)[ok])
        # 训练池
        tr = {}
        for y in YEARS:
            d = pd.read_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (y, w)))
            tr[y] = d
        # A: 模型固定=2017 池
        clfA = rf().fit(tr[2017][VC.FEATS].to_numpy('float32'),
                        VC.to_v31(tr[2017]['class_new'].to_numpy(int)))
        a = {}
        for y in YEARS:
            X = ev[y][VC.FEATS].to_numpy('float32')[pos[y][ok]]
            a[y] = (clfA.predict(X) == t).mean()
        # B: 特征固定=2017
        X17 = ev[2017][VC.FEATS].to_numpy('float32')[ok]
        b = {}
        for y in YEARS:
            clf = rf().fit(tr[y][VC.FEATS].to_numpy('float32'),
                           VC.to_v31(tr[y]['class_new'].to_numpy(int)))
            b[y] = (clf.predict(X17) == t).mean()
        VC.emit('%s n=%d ｜ A(特征年): %s ｜ B(池年): %s' % (
            w, int(ok.sum()), {y: round(a[y], 4) for y in YEARS}, {y: round(b[y], 4) for y in YEARS}))
        L.append('| %s (n=%d) | %s | %s |' % (
            w, int(ok.sum()),
            ' / '.join('%.4f' % a[y] for y in YEARS),
            ' / '.join('%.4f' % b[y] for y in YEARS)))
    L += ['', '列序：2017 / 2020 / 2023 / 2024', '',
          '- A 行差异 = 只换评估特征年（训练池固定 2017）→ 特征年份效应',
          '- B 行差异 = 只换训练池年（评估特征固定 2017）→ 训练池年份效应']
    fp = os.path.join(VC.REPT, '年度归因_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
