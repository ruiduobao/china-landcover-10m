# -*- coding: utf-8 -*-
"""k7_phys_screen.py — 在 FCS10 框评估点（E，n=87）上量测物理屏幕的提纯效果

* 输入：data/shrub_nat/eval_fcs10_scored.parquet（E：doc57 的 FCS10 层随机抽样，87 判读点）
* 规则源（预注册屏幕组合）：
        S1 rh98∈[1.5,8]（GEDI 冠层高度灌丛带）
        S2 rh98_mx11<8（1.1 km 邻域无高冠层）
        S3 tc<30（Hansen 树覆盖低）
        D1 dwp≥0.2（DW 灌丛概率不低）
        D2 dwl==6（DW 众数即灌丛）
        F1 ndvi_amp∈[0.05,0.55]（物候振幅在灌丛合理带）
* 门槛：逐组合报 n/真灌丛/precision/Wilson CI；按省分层报；n<15 只作参考不裁决
* 输出：data/shrub_nat/phys_screen.json + data/shrub_nat/phys_screen.md
* 用法：python k7_phys_screen.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    e = pd.read_parquet(os.path.join(OUTD, 'eval_fcs10_scored.parquet'))
    if 'prov' not in e.columns:
        d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
        pm = {(round(a, 6), round(b, 6)): c for a, b, c in
              zip(d57.lon.to_numpy(float), d57.lat.to_numpy(float), d57['prov'])}
        e['prov'] = [pm.get((round(a, 6), round(b, 6)), '?')
                     for a, b in zip(e.lon.to_numpy(float), e.lat.to_numpy(float))]
    y = e['y'].to_numpy(int)
    emit('E：n=%d 真灌丛 %d（precision=%.3f）' % (len(e), y.sum(), y.mean()))
    r98 = e['rh98'].to_numpy(float)
    mx = e['rh98_mx11'].to_numpy(float)
    tc = e['tc'].to_numpy(float)
    dwp = e['dwp'].to_numpy(float)
    dwl = e['dwl'].to_numpy(float)
    amp = e['ndvi_amp'].to_numpy(float)
    S1 = np.isfinite(r98) & (r98 >= 1.5) & (r98 <= 8.0)
    S1n = np.isfinite(r98)
    S2 = np.isfinite(mx) & (mx < 8.0)
    S3 = np.isfinite(tc) & (tc < 30)
    D1 = dwp >= 0.2
    D2 = dwl == 6
    F1 = amp >= 50 & (amp <= 550)
    combos = {
        'E(无屏幕)': np.ones(len(e), bool),
        'E+S1': S1,
        'E+S2': S2,
        'E+S3': S3,
        'E+S1+S2': S1 & S2,
        'E+S1+S2+S3': S1 & S2 & S3,
        'E+S2+S3': S2 & S3,
        'E+D1': D1,
        'E+D1+D2': D1 & D2,
        'E+S1+S2+D1': S1 & S2 & D1,
        'E+S1+S2+S3+D1': S1 & S2 & S3 & D1,
        'E+S1+S2+S3+F1': S1 & S2 & S3 & F1,
        'E+S1+S2+S3+D1+F1': S1 & S2 & S3 & D1 & F1,
    }
    out = dict(n_e=int(len(e)), pos_e=int(y.sum()), base=round(float(y.mean()), 4), combos={},
               by_prov={})
    lines = ['# FCS10 框物理屏幕提纯（E，doc57 抽样）%s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- E＝doc57 的 FCS10 灌丛层随机抽样判读点，n=%d，其中真灌丛 %d（precision=%.3f）' % (
                 len(e), y.sum(), y.mean()),
             '- 屏幕定义见 `k7_phys_screen.py` 顶部预注册块', '',
             '| 组合 | n | 真灌丛 | precision | 95%CI | 相对基线 |', '|---|---|---|---|---|---|']
    for tag, m in combos.items():
        n, k = int(m.sum()), int(y[m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        out['combos'][tag] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                  wilson=[lo, hi], delta=(round(prec - y.mean(), 4) if prec is not None else None))
        lines.append('| %s | %d | %d | %s | [%.2f,%.2f] | %s |' % (
            tag, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi,
            ('%+.3f' % (prec - y.mean())) if prec is not None else '—'))
        emit('%s：n=%d 灌=%d precision=%s' % (tag, n, k, ('%.3f' % prec) if prec is not None else '—'))
    # 按省（最优组合）
    best = 'E+S1+S2+S3'
    if out['combos'][best]['n'] == 0:
        best = max(combos, key=lambda t: out['combos'][t]['n'])
    mb = combos[best]
    lines += ['', '## 最优组合 `%s` 按省' % best, '', '| 省 | n | 真灌丛 | precision |', '|---|---|---|---|']
    for p, g in e.groupby('prov'):
        idx = e.index.isin(g.index)
        n = int((mb & idx).sum())
        k = int(y[mb & idx].sum())
        out['by_prov'][p] = dict(n=n, shrub=k, precision=(round(k / n, 4) if n else None))
        lines.append('| %s | %d | %d | %s |' % (p, n, k, ('%.3f' % (k / n)) if n else '—'))
    json.dump(out, open(os.path.join(OUTD, 'phys_screen.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'phys_screen.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ phys_screen.json / phys_screen.md')


if __name__ == '__main__':
    main()
