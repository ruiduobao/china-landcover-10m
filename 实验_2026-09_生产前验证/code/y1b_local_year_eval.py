# -*- coding: utf-8 -*-
"""y1b_local_year_eval.py — 年度匹配检查 · 本地 RF 评估（零 GEE 成本，分钟级出结果）
与 GEE smileRandomForest 对齐：n_estimators=100、min_samples_leaf=2、max_leaf_nodes=5000、
max_features='sqrt'、class_weight=None（GEE 无类别权重）。训练标签先做 v31 映射。
评估口径：
  · 逐年：该年固定留出 rid（2023 窗留出 ∩ 该年存在）→ 逐年绝对水平
  · 共同集（主口径）：4 年均存在的同一批 rid → 严格的跨年可比
CPU 纪律：n_jobs=15，单进程串行。
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
OUTD = os.path.join(VC.DATA, 'yearly')
METD = os.path.join(VC.RES, 'yearly')
FOCUS = [9, 10, 2, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17]


def main():
    os.makedirs(METD, exist_ok=True)
    plan = VC.jload(os.path.join(OUTD, 'plan.json'))
    common = {w: set(plan['common_eval'][w]['rids']) for w in WS}
    names = VC.V31_NAMES()
    rows = []
    t00 = time.time()
    for w in WS:
        for year in YEARS:
            tr_fp = os.path.join(OUTD, 'train_%d_%s.parquet' % (year, w))
            ev_fp = os.path.join(OUTD, 'eval_%d_%s.parquet' % (year, w))
            if not (os.path.exists(tr_fp) and os.path.exists(ev_fp)):
                VC.emit('缺表 %d %s' % (year, w)); continue
            tr = pd.read_parquet(tr_fp)
            ev = pd.read_parquet(ev_fp)
            if len(tr) < 100 or len(ev) < 30:
                VC.emit('表太小 %d %s（tr=%d ev=%d）' % (year, w, len(tr), len(ev))); continue
            Xtr = tr[VC.FEATS].to_numpy('float32')
            ytr = VC.to_v31(tr['class_new'].to_numpy(int))
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                         max_leaf_nodes=5000, max_features='sqrt',
                                         class_weight=None, n_jobs=15, random_state=7)
            t0 = time.time()
            clf.fit(Xtr, ytr)
            fit_s = time.time() - t0
            Xev = ev[VC.FEATS].to_numpy('float32')
            pev = clf.predict(Xev)
            tev = VC.to_v31(ev['class_new'].to_numpy(int))
            rid = ev['row_id'].to_numpy()
            res = dict(window=w, year=year, n_train=len(tr), n_eval=len(ev), fit_s=round(fit_s, 1))
            m = VC.metrics(tev, pev)
            res['all'] = dict(OA=m['OA'], macroF1=m['macroF1'], per=m['per'])
            res['m9'] = VC.metrics(VC.to_macro(tev), VC.to_macro(pev))['OA']
            idx = np.isin(rid, list(common[w]))
            if idx.sum() >= 30:
                m2 = VC.metrics(tev[idx], pev[idx])
                res['common'] = dict(n=int(idx.sum()), OA=m2['OA'], macroF1=m2['macroF1'],
                                     per=m2['per'],
                                     m9=VC.metrics(VC.to_macro(tev[idx]), VC.to_macro(pev[idx]))['OA'])
            VC.jsave(res, os.path.join(METD, 'local_%d_%s.json' % (year, w)))
            VC.emit('%s %d: 训练 %d / 评估 %d ｜ 全集 OA=%.4f mF1=%.4f ｜ 共同集(n=%s) OA=%.4f mF1=%.4f（拟合 %.0fs）' % (
                w, year, len(tr), len(ev), res['all']['OA'], res['all']['macroF1'],
                res.get('common', {}).get('n', 0), res.get('common', {}).get('OA', 0),
                res.get('common', {}).get('macroF1', 0), fit_s))
            rows.append(res)
    # 汇总表
    L = ['# 年度匹配检查（本地 RF，与 GEE 配置对齐）%s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '> 训练：各年窗内 20k 点（年度子集，剔除固定留出 rid 与 Baseline 训练 rid）；标签 v31 映射',
         '> 评估：逐年固定留出点 / 共同集 = 4 年均存在的同一批点（严格跨年可比）', '']
    L += ['## 逐年 OA / macroF1（共同集口径，n=同一批点）', '',
          '| 窗 | n_common | 2017 | 2020 | 2023 | 2024 | 极差 |', '|---|---|---|---|---|---|---|']
    by = {}
    for r in rows:
        by.setdefault(r['window'], {})[r['year']] = r
    for w in WS:
        d = by.get(w, {})
        cells = []
        vals = []
        ncom = 0
        for y in YEARS:
            c = d.get(y, {}).get('common')
            if c:
                cells.append('%.4f/%.4f' % (c['OA'], c['macroF1']))
                vals.append(c['OA']); ncom = c['n']
            else:
                cells.append('—')
        rng = ('%.2fpp' % ((max(vals) - min(vals)) * 100)) if len(vals) > 1 else '—'
        L.append('| %s | %d | %s | %s |' % (w, ncom, ' | '.join(cells), rng))
    L += ['', '## 焦点类逐类 F1（共同集，按年）', '',
          '| 窗 | 类 | 2017 | 2020 | 2023 | 2024 |', '|---|---|---|---|---|---|']
    for w in WS:
        d = by.get(w, {})
        for c in FOCUS:
            cells = []
            for y in YEARS:
                cm = d.get(y, {}).get('common', {}).get('per', {})
                cells.append('%.3f' % cm[c]['f1'] if str(c) in cm else '—')
            if any(x != '—' for x in cells):
                L.append('| %s | %02d %s | %s |' % (w, c, names.get(str(c), '?'), ' | '.join(cells)))
    fp = os.path.join(VC.REPT, '年度匹配检查_%s.md' % time.strftime('%Y%m%d'))
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    VC.emit('汇总 → %s（用时 %.1f 分钟）' % (fp, (time.time() - t00) / 60))


if __name__ == '__main__':
    main()
