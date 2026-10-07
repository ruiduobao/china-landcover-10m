# -*- coding: utf-8 -*-
"""y1e_screened.py — 来源筛选复测：排除 FCS10-2023 灌丛派生点后，年际差异是否消失？
对照：① 全池（原口径）② 排除 src 以 glc_fcs10 开头（同口径施于四年）③ 仅保留 tier=stable 类（若可用）
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
OUTD = os.path.join(VC.DATA, 'yearly')


def rf():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                  max_features='sqrt', class_weight=None, n_jobs=15, random_state=7)


def run(w, mode):
    plan = VC.jload(os.path.join(OUTD, 'plan.json'))
    common = set(plan['common_eval'][w]['rids'])
    res = {}
    for y in YEARS:
        tr = pd.read_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (y, w)))
        ev = pd.read_parquet(os.path.join(OUTD, 'eval_%d_%s.parquet' % (y, w)))
        if mode == 'screen':
            tr = tr[~tr.src.fillna('').str.startswith('glc_fcs10')]
            ev = ev[~ev.src.fillna('').str.startswith('glc_fcs10')]   # 评估点同样剔除，保证可比
        if len(tr) < 500 or len(ev) < 100:
            res[y] = None; continue
        clf = rf().fit(tr[VC.FEATS].to_numpy('float32'), VC.to_v31(tr['class_new'].to_numpy(int)))
        # 评估：共同集 ∩ 该年评估点
        rid = ev.row_id.to_numpy()
        sel = np.isin(rid, list(common))
        if sel.sum() < 100:
            res[y] = None; continue
        p = clf.predict(ev[VC.FEATS].to_numpy('float32')[sel])
        t = VC.to_v31(ev['class_new'].to_numpy(int)[sel])
        m = VC.metrics(t, p)
        res[y] = dict(n=int(sel.sum()), n_train=int(len(tr)), OA=m['OA'], macroF1=m['macroF1'])
    return res


def main():
    L = ['# 来源筛选复测（排除 FCS10-2023 灌丛派生点）%s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '| 窗 | 口径 | 2017 | 2020 | 2023 | 2024 |', '|---|---|---|---|---|---|']
    for w in WS:
        for mode, tag in [('all', '全池'), ('screen', '筛除FCS10')]:
            r = run(w, mode)
            cells = []
            for y in YEARS:
                v = r.get(y)
                cells.append('%.4f' % v['OA'] if v else '—')
            L.append('| %s | %s | %s |' % (w, tag, ' / '.join(cells)))
            VC.emit('%s %s: %s' % (w, tag, ' / '.join(cells)))
    fp = os.path.join(VC.REPT, '来源筛选复测_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
