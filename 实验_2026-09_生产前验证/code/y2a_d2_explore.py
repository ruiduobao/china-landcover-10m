# -*- coding: utf-8 -*-
"""y2a_d2_explore.py — D2 决策所需的数据侦察（零成本）
问题：
  1) 各年 × 窗 训练池的来源构成（glc_fcs10* 占比、类别）
  2) 各年 × 窗 评估点的来源构成（FCS10 派生点是否在评估集里）
  3) 共同评估集（4 年同点）的 v31 类别分布（尤其灌丛 09/10）
  4) 若筛除 FCS10，各窗灌丛训练点剩多少
产物：data/yearly/d2_explore.json + reports/D2数据侦察_YYYYMMDD.md
"""
import os, sys, time, json
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
OUTD = os.path.join(VC.DATA, 'yearly')


def srckey(s):
    s = '' if s is None else str(s)
    for p in ('glc_fcs10', 'v2_glc_fcs30d', 'ng_', 'cvh', 'inat', 'gbif'):
        if s.startswith(p) or p in s:
            return p
    return s[:22] if s else '(空)'


def main():
    names = VC.V31_NAMES()
    plan = VC.jload(os.path.join(OUTD, 'plan.json'))
    common = {w: set(plan['common_eval'][w]['rids']) for w in WS}
    out = dict(common_n={w: len(common[w]) for w in WS}, train={}, ev={}, common_cls={}, shrub_after_screen={})
    L = ['# D2 数据侦察 %s' % time.strftime('%Y-%m-%d %H:%M'), '']
    for w in WS:
        L += ['## %s' % w, '', '### 训练池来源构成（行=年，列=来源族）', '',
              '| 年 | n | ' + ' | '.join(['glc_fcs10', 'v2_glc_fcs30d', 'ng_', '其他']) + ' | FCS10内类别 |', '|---|---|---|---|---|---|---|']
        for y in YEARS:
            tr = pd.read_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (y, w)))
            ev = pd.read_parquet(os.path.join(OUTD, 'eval_%d_%s.parquet' % (y, w)))
            src = tr['src'].fillna('').astype(str).map(srckey)
            cnt = Counter(src)
            f10 = tr[src.str.startswith('glc_fcs10')]
            cls_in_f10 = Counter(VC.to_v31(f10['class_new'].to_numpy(int)).tolist()) if len(f10) else {}
            cls_txt = ' '.join('%s:%d' % (names.get(str(k), k), v) for k, v in sorted(cls_in_f10.items(), key=lambda x: -x[1])[:5])
            # 灌丛训练点（v31 9/10）在筛除前后
            y31 = VC.to_v31(tr['class_new'].to_numpy(int))
            n_shrub_all = int(np.isin(y31, [9, 10]).sum())
            if len(f10):
                keep = tr[~src.str.startswith('glc_fcs10')]
                y31k = VC.to_v31(keep['class_new'].to_numpy(int))
                n_shrub_scr = int(np.isin(y31k, [9, 10]).sum())
            else:
                n_shrub_scr = n_shrub_all
            out['train'].setdefault(w, {})[str(y)] = dict(n=len(tr), src=cnt, f10_cls=cls_in_f10,
                                                          shrub_all=n_shrub_all, shrub_after=n_shrub_scr)
            L.append('| %d | %d | %d (%.0f%%) | %d | %d | %d | %s |' % (
                y, len(tr), cnt.get('glc_fcs10', 0), 100.0 * cnt.get('glc_fcs10', 0) / max(1, len(tr)),
                cnt.get('v2_glc_fcs30d', 0), cnt.get('ng_', 0), len(tr) - sum(cnt.get(k, 0) for k in ('glc_fcs10', 'v2_glc_fcs30d', 'ng_')),
                cls_txt or '—'))
            VC.emit('%s %d 训练 %d（FCS10 %d）灌丛训练点 %d→%d' % (w, y, len(tr), cnt.get('glc_fcs10', 0), n_shrub_all, n_shrub_scr))
            # 评估集来源
            esrc = ev['src'].fillna('').astype(str).map(srckey)
            ecnt = Counter(esrc)
            sel = np.isin(ev.row_id.to_numpy(), list(common[w]))
            out['ev'].setdefault(w, {})[str(y)] = dict(n=len(ev), src=ecnt, n_common=int(sel.sum()))
        # 共同集类别分布（用 2023 年子集代表）
        tr23 = pd.read_parquet(os.path.join(OUTD, 'train_2023_%s.parquet' % w))
        ev23 = pd.read_parquet(os.path.join(OUTD, 'eval_2023_%s.parquet' % w))
        sel = np.isin(ev23.row_id.to_numpy(), list(common[w]))
        y31 = VC.to_v31(ev23['class_new'].to_numpy(int)[sel])
        cnt = Counter(y31.tolist())
        out['common_cls'][w] = {int(k): int(v) for k, v in cnt.items()}
        txt = ' '.join('%s:%d' % (names.get(str(k), k), v) for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:12])
        L += ['### 共同评估集(n=%d) 类别分布：%s' % (int(sel.sum()), txt), '']
        VC.emit('%s 共同集 n=%d 类别: %s' % (w, int(sel.sum()), txt))
    VC.jsave(out, os.path.join(OUTD, 'd2_explore.json'))
    fp = os.path.join(VC.REPT, 'D2数据侦察_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
