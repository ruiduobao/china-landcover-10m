# -*- coding: utf-8 -*-
"""h5_audit_stats.py — 目视判读合并 + 分层精度估计 + 预注册阈值裁决

* 输入：data/shrub_nat/imagery/judge_*.csv（各判读子代理输出，表头 point_id,Q1_shrub,
        Q2_evergreen,Q3_other,conf,note）+ audit_points.csv（指标与分层）
* 规则源（预注册阈值，写死本脚本；改动须在报告说明）：
        P_A 格（多源正向）：precision≥0.70 且 n_judged≥20 → **接受**；0.50~0.70 → 减半采用；<0.50 → 拒绝
        P_B 格（单源正向）：≥0.75 → 接受；0.60~0.75 → 减半采用；<0.60 → 拒绝
        N_A/N_B 格（负样本）：precision<0.30 → 接受为负样本；0.30~0.45 → 减半采用；>0.45 → 拒绝
        ——precision = Q1=是 /（是+否），"无法判读"剔除出分母但单独报告判读率
* 门槛：报告每格 n/nd/判读率/precision 及 95% 置信区间（Wilson），合并级（n<20 只报合并级）
* 输出：data/shrub_nat/audit_verdict.json + data/shrub_nat/audit_stats.md
* 用法：python h5_audit_stats.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
IMGD = os.path.join(OUTD, 'imagery')


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


def verdict_of(k, prec, nd):
    """pre-registered 阈值裁决，返回 (决定, 采用比例)。"""
    if k == 'P_A':
        if nd >= 20 and prec >= 0.70:
            return 'accept', 1.0
        if prec >= 0.50:
            return 'shrink', 0.5
        return 'reject', 0.0
    if k == 'P_B':
        if nd >= 20 and prec >= 0.75:
            return 'accept', 1.0
        if prec >= 0.60:
            return 'shrink', 0.5
        return 'reject', 0.0
    # N 带：precision 越低越适合当负样本
    if prec < 0.30:
        return 'accept', 1.0
    if prec <= 0.45:
        return 'shrink', 0.5
    return 'reject', 0.0


def main():
    import glob
    files = sorted(glob.glob(os.path.join(IMGD, 'judge_*.csv')))
    if not files:
        emit('没有 judge_*.csv'); return
    js = []
    for f in files:
        d = pd.read_csv(f, encoding='utf-8-sig')
        d['_from'] = os.path.basename(f)
        js.append(d)
    jd = pd.concat(js, ignore_index=True)
    jd = jd.drop_duplicates(subset=['point_id'], keep='first')
    emit('判读 %d 点（来自 %d 份）' % (len(jd), len(files)))
    pts = pd.read_csv(os.path.join(OUTD, 'audit_points.csv'), encoding='utf-8-sig')
    pts['point_id'] = pts['point_id'].astype(str)
    mg = pts.merge(jd, on='point_id', how='left', suffixes=('', '_j'))
    mg['Q1_shrub'] = mg['Q1_shrub'].fillna('未判读')
    n_all = len(mg)
    n_jud = int((mg['Q1_shrub'] != '未判读').sum())
    mg['judged'] = mg['Q1_shrub'].isin(['是', '否'])
    emit('审计点 %d，判读 %d（%.0f%%），可判读 %d' % (n_all, n_jud, 100 * n_jud / max(1, n_all),
                                            int(mg['judged'].sum())))
    out = dict(n_all=n_all, n_judged=n_jud, cells={})
    rows = []
    for (cell, k), g in mg.groupby(['aud_cell', 'k']):
        nd = int(g['judged'].sum())
        ny = int((g['Q1_shrub'] == '是').sum())
        prec = ny / nd if nd else float('nan')
        lo, hi = wilson(ny, nd)
        dec, ratio = verdict_of(k, prec if nd else 0.0, nd)
        # 常绿/落叶（仅确认灌丛）
        ev = g[g['Q1_shrub'] == '是']['Q2_evergreen'].value_counts().to_dict()
        rec = dict(cell=cell, k=k, n=int(len(g)), judged=nd, shrub=ny,
                   undecidable=int((g['Q1_shrub'] == '无法判读').sum()),
                   precision=(round(prec, 4) if nd else None), wilson=[lo, hi],
                   verdict=dec, ratio=ratio, evergreen={kk: int(vv) for kk, vv in ev.items()})
        out['cells']['%s|%s' % (cell, k)] = rec
        rows.append((cell, k, len(g), nd, ny, rec['precision'], lo, hi, dec, ratio,
                     int((g['Q1_shrub'] == '无法判读').sum()),
                     '/'.join('%s%d' % (kk, vv) for kk, vv in ev.items())))
        emit('%s %s：n=%d 判=%d 灌=%d precision=%.3f[%.2f,%.2f] → %s×%.1f' % (
            cell, k, len(g), nd, ny, prec if nd else -1, lo, hi, dec, ratio))
    # 汇总
    ag = mg[mg['judged']]
    out['overall'] = dict(n=int(len(ag)), shrub=int((ag['Q1_shrub'] == '是').sum()),
                          precision=round(float((ag['Q1_shrub'] == '是').mean()), 4))
    acc = [r for r in rows if r[8] == 'accept']
    out['accepted_cells'] = ['%s|%s' % (r[0], r[1]) for r in acc]
    out['shrunk_cells'] = ['%s|%s' % (r[0], r[1]) for r in rows if r[8] == 'shrink']
    json.dump(out, open(os.path.join(OUTD, 'audit_verdict.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    L = ['# 全国灌丛样本目视审计 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '- 判读来源：%d 份子代理输出；审计点 %d，判读 %d，可判读 %d' % (len(files), n_all, n_jud, int(mg['judged'].sum())),
         '- precision = Q1=是 /（是+否）；区间为 Wilson 95%；裁决阈值见 `h5_audit_stats.py` 预注册块', '',
         '| 审计格 | 层 | n | 判读 | 灌丛 | precision | 95%CI | 裁决 | 采用比 | 无法判读 | 常绿/落叶 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append('| %s | %s | %d | %d | %d | %s | [%.2f,%.2f] | %s | %.1f | %d | %s |' % (
            r[0], r[1], r[2], r[3], r[4], ('%.3f' % r[5]) if r[5] is not None else '—',
            r[6], r[7], r[8], r[9], r[10], r[11]))
    L += ['', '## 汇总', '',
          '- 整体 precision（全部可判读点）= %.4f（n=%d）' % (out['overall']['precision'], out['overall']['n']),
          '- 接受格 %d 个：%s' % (len(out['accepted_cells']), '，'.join(out['accepted_cells'])),
          '- 减半格 %d 个：%s' % (len(out['shrunk_cells']), '，'.join(out['shrunk_cells']))]
    fp = os.path.join(OUTD, 'audit_stats.md')
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    emit('→ ' + fp)


if __name__ == '__main__':
    main()
