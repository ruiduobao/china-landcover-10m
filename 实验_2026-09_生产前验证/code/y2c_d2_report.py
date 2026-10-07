# -*- coding: utf-8 -*-
"""y2c_d2_report.py — 从 results/d2/*.json 生成扩展报告（不重跑拟合）
追加内容：
  ① 逐类 F1（共同集，R0 vs R1，2023/2024 分年）
  ② 去向分析：R0 下被预测为 09/10 的点，其真实类分布（= 谁在为"灌丛"付出代价）
  ③ 决策表：R0/R1/R2/R3 的 OA 增益 vs 灌丛存活的量化权衡
"""
import os, sys, time, json, glob
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np

RESD = os.path.join(VC.RES, 'd2')
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
YEARS = [2017, 2020, 2023, 2024]
POLS = ['R0', 'R1', 'R2', 'R3', 'R4']


def load(w, y, p):
    fp = os.path.join(RESD, 'p_%s_%d_%s.json' % (w, y, p))
    return VC.jload(fp) if os.path.exists(fp) else None


def main():
    names = VC.V31_NAMES()
    L = ['# D2 灌丛策略复测 · 扩展分析 %s' % time.strftime('%Y-%m-%d %H:%M'), '']
    # ①/③ 决策表
    L += ['## 决策表（2023 · 共同集 · 底座惯例口径）', '',
          '| 窗 | 策略 | 保留FCS10 | OA | ΔOA vs R0 | 灌丛预测率 | 年度极差 |',
          '|---|---|---|---|---|---|---|']
    for w in WS:
        base = None
        for p in POLS:
            d = load(w, 2023, p)
            if not d:
                continue
            oas = [load(w, y, p)['OA'] for y in YEARS if load(w, y, p)]
            rng = (max(oas) - min(oas)) * 100
            if p == 'R0':
                base = d['OA']
            L.append('| %s | %s | %d | %.4f | %+.4f | %.3f%% | %.2fpp |' % (
                w, p, d['n_f10_kept'], d['OA'], d['OA'] - base, d['shrub_pred_rate'] * 100, rng))
    # ② 去向分析
    L += ['', '## 去向分析：R0（2023）下被预测为灌丛的点，其真实类 Top8', '',
          '| 窗 | 预测灌丛点数 | 真实类 Top8（占预测灌丛的%） |', '|---|---|---|']
    for w in WS:
        d = load(w, 2023, 'R0')
        if not d:
            continue
        C = np.array(d['conf']).reshape(25, 25)
        col = C[:, 9] + C[:, 10]
        tot = col.sum()
        top = sorted([(i, int(col[i])) for i in range(25) if col[i] > 0], key=lambda x: -x[1])[:8]
        txt = ' '.join('%s:%.0f%%' % (names.get(str(i), i), 100.0 * v / max(1, tot)) for i, v in top)
        L.append('| %s | %d | %s |' % (w, tot, txt))
    # ① 逐类 F1
    for y in (2023, 2024):
        L += ['', '## 逐类 F1（%d 共同集，R0/R1/R4，仅列 n≥20 的类）' % y, '',
              '| 窗 | 类 | R0 F1 | R1 F1 | R4 F1 | n |', '|---|---|---|---|---|---|']
        for w in WS:
            a, b, d4 = load(w, y, 'R0'), load(w, y, 'R1'), load(w, y, 'R4')
            if not (a and b):
                continue
            codes = sorted(set(a['per'].keys()) | set(b['per'].keys()), key=lambda x: int(x))
            for c in codes:
                ca, cb = a['per'].get(c), b['per'].get(c)
                c4 = (d4 or {}).get('per', {}).get(c)
                n = (ca or cb or {}).get('n', 0)
                if n < 20:
                    continue
                fa, fb = (ca or {}).get('f1', 0), (cb or {}).get('f1', 0)
                L.append('| %s | %02d %s | %.3f | %.3f | %s | %d |' % (
                    w, int(c), names.get(str(c), '?'), fa, fb,
                    ('%.3f' % c4['f1']) if c4 else '—', n))
    fp = os.path.join(VC.REPT, 'D2灌丛策略复测_扩展_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
