# -*- coding: utf-8 -*-
"""a2_land_report.py — 生成 doc67 用的质检汇总表（不写文档，只产出 markdown 片段）

* 输入：out/land_qc_{crop,veg,water,urban}.parquet
* 输出：out/land_qc_tables.md（总表 / 逐类 reason / src×verdict / 关键证据分位）
* 用法：python a2_land_report.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/v31_exp/data/land_qc'; OUT = os.path.join(BASE, 'out')
NAME = {1: '草本旱地', 3: '灌溉耕地', 11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 22: '裸地',
        16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 23: '水体',
        21: '不透水面', 24: '永久冰雪'}
KEY = {16: 'max_300', 17: 'max_300', 20: 'max_300', 19: 'max_300', 18: 'trees', 23: 'max_extent',
       11: 'ndvi', 13: 'ndvi', 22: 'ndvi', 12: 'dem', 1: 'crops', 3: 'crops', 21: 'dw_built_150',
       24: 'snow_and_ice'}


def q(s, p):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return np.nan if not len(s) else round(float(np.percentile(s, p)), 2)


def main():
    fs = sorted(glob.glob(os.path.join(OUT, 'land_qc_*.parquet')))
    fs = [f for f in fs if os.path.basename(f) not in ('land_qc_all.parquet',)]
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d['point_id'] = d.point_id.astype(str)
    L = []
    L.append('### T1 逐类总表\n')
    L.append('| class24 | 名称 | n | keep | flag | quarantine | keep% | quar% |')
    L.append('|---|---|---|---|---|---|---|---|')
    for cl in sorted(NAME):
        g = d[d.class24 == cl]
        if len(g) == 0:
            L.append('| %d | %s | 0 | 0 | 0 | 0 | — | — |' % (cl, NAME[cl])); continue
        k = int((g.verdict == 'keep').sum()); fg = int((g.verdict == 'flag').sum())
        qr = int((g.verdict == 'quarantine').sum())
        L.append('| %d | %s | %d | %d | %d | %d | %.2f | %.2f |' % (cl, NAME[cl], len(g), k, fg, qr,
                                                                    100 * k / len(g), 100 * qr / len(g)))
    L.append('\n### T2 逐类 quarantine 理由（top5）\n')
    L.append('| class24 | 名称 | reason（n, 占比） |')
    L.append('|---|---|---|')
    for cl in sorted(NAME):
        g = d[(d.class24 == cl) & (d.verdict == 'quarantine')]
        if not len(g):
            L.append('| %d | %s | — |' % (cl, NAME[cl])); continue
        vc = g.reason.value_counts()
        L.append('| %d | %s | %s |' % (cl, NAME[cl],
                 '; '.join('%s(%d, %.1f%%)' % (r, n, 100 * n / len(g)) for r, n in vc.head(5).items())))
    L.append('\n### T3 src × verdict（合计）\n')
    d['srcg'] = d.src.where(d.src.isin(d.src.value_counts().head(12).index), 'other')
    piv = pd.crosstab(d.srcg, d.verdict)
    for c in ('keep', 'flag', 'quarantine'):
        if c not in piv:
            piv[c] = 0
    piv['quar%'] = (100 * piv.quarantine / piv.sum(axis=1)).round(2)
    L.append(piv[['keep', 'flag', 'quarantine', 'quar%']].sort_values('quar%', ascending=False).to_markdown())
    L.append('\n### T4 关键证据分位（keep+flag vs quarantine）\n')
    L.append('| class24 | 证据 | all p10/p50/p90 | keep+flag p50 | quar p50 |')
    L.append('|---|---|---|---|---|')
    for cl in sorted(NAME):
        g = d[d.class24 == cl]; c = KEY[cl]
        if c not in g.columns:
            continue
        kf = g[g.verdict != 'quarantine'][c]; qq = g[g.verdict == 'quarantine'][c]
        L.append('| %d | %s | %s / %s / %s | %s | %s |' % (cl, c, q(g[c], 10), q(g[c], 50), q(g[c], 90),
                                                           q(kf, 50), q(qq, 50)))
    txt = '\n'.join(L)
    open(os.path.join(OUT, 'land_qc_tables.md'), 'w', encoding='utf-8').write(txt)
    print(txt[:4000])


if __name__ == '__main__':
    main()
