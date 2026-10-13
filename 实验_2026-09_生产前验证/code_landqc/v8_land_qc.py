# -*- coding: utf-8 -*-
"""v8_land_qc.py — 其余地类样本的统一质检（预注册判据 keep/quarantine/flag 三态 + 逐类统计）

* 输入：--group <crop|veg|water|urban>；点表 lists/grpN_*.parquet；证据 feats/<tag>/*.parquet
* 判据：out/criteria_land.json（**先冻结后跑**；如需修改须在 reports/land_qc_2026-10-13/
        预注册判据.md 追加带时间戳的修正记录）
* 三态：keep=通过正向证据；quarantine=证据与标签矛盾（剔除）；flag=证据不足/中间态（保留但标低置信）
* 输出：out/land_qc_<tag>.parquet（逐点三态+派生列）
        out/stats_<tag>.json（逐类 keep/quar/flag 计数与占比）
        qc/quarantine_<tag>.csv（人工可核）
* 用法：python v8_land_qc.py --group crop
"""
import argparse
import json
import os
import sys
import time
import glob

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
BASE = os.path.join(WORK, 'data', 'land_qc')
OUT = os.path.join(BASE, 'out')
QCD = os.path.join(BASE, 'qc')
CRIT = os.path.join(OUT, 'criteria_land.json')
GIDX = {'crop': 1, 'veg': 2, 'water': 3, 'urban': 4}
DW8 = ('water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice', 'flooded_vegetation', 'shrub')
NEI = ('occ_300', 'max_300', 'm07_150', 'm01_150', 'dw_built_150', 'dw_water_150', 'dw_bare_150')
SENT = ('s2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong', 'dem', 'slope', 'relief500')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def derive(d):
    """哨兵归一：GSW 点级 -1（无数据）→ NaN（判据里两边均为 False → 归 flag）；
    邻域与 DW 概率负值 → 0（"无证据"）。真实 max_extent∈{0,1}、occurrence∈[0,100]。"""
    for c in ('occurrence', 'max_extent', 'seasonality', 'recurrence'):
        d[c] = pd.to_numeric(d[c], errors='coerce').where(lambda x: x >= 0)
    for c in NEI:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).clip(lower=0)
    for c in DW8:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(-1).clip(0, 1)
    for c in SENT:
        d[c] = pd.to_numeric(d[c], errors='coerce').where(lambda x: x > -9000)
    d['veg'] = d.trees + d.grass
    d['ndvi'] = d.s2_ndvi_xia.fillna(d.s2_ndvi_dong)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=list(GIDX))
    a = ap.parse_args()
    tag = a.group
    crit = json.load(open(CRIT, encoding='utf-8'))
    pts = pd.read_parquet(os.path.join(BASE, 'lists', 'grp%d_%s.parquet' % (GIDX[tag], tag)))
    pts['point_id'] = pts.point_id.astype(str)
    fs = sorted(glob.glob(os.path.join(BASE, 'feats', tag, 'land_p*_c*.parquet')))
    if not fs:
        raise SystemExit('缺 feats/%s/*.parquet' % tag)
    f = pd.concat([pd.read_parquet(x) for x in fs], ignore_index=True)
    f['point_id'] = f.point_id.astype(str)
    f = f.drop_duplicates('point_id')
    d = pts.merge(f, on='point_id', how='left', indicator='_m')
    d['has_ev'] = (d._m == 'both').astype('int8')
    d = d.drop(columns=['_m'])
    have_ev = d.has_ev == 1
    emit('%s：点 %d / 证据块 %d / 合并后 %d（无证据 %d）' % (tag, len(pts), len(fs), len(d), int((~have_ev).sum())))
    d = derive(d)
    d['verdict'] = 'flag'
    d['reason'] = np.where(d.has_ev == 1, '', 'NO_EVIDENCE')
    st = {}
    for cl_s, rules in crit['classes'].items():
        cl = int(cl_s)
        g = d[(d.class24 == cl) & (d.has_ev == 1)]
        if len(g) == 0:
            continue
        K = g.eval(rules['keep'])
        Q = g.eval(rules['quar'])
        d.loc[g.index[K], 'verdict'] = 'keep'
        qidx = g.index[Q & ~K]
        d.loc[qidx, 'verdict'] = 'quarantine'
        d.loc[qidx, 'reason'] = rules.get('reason', 'EVIDENCE_CONFLICT')
        st[cl_s] = dict(n=int(len(g)), keep=int(K.sum()), quar=int((Q & ~K).sum()),
                        flag=int((~K & ~Q).sum()),
                        keep_pct=round(100.0 * K.sum() / len(g), 2),
                        quar_pct=round(100.0 * (Q & ~K).sum() / len(g), 2))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(QCD, exist_ok=True)
    keep_cols = ['point_id', 'lon', 'lat', 'class_new', 'src', 'class24', 'has_ev', 'verdict', 'reason',
                 'max_extent', 'occurrence', 'seasonality', 'occ_300', 'max_300', 'm07_150', 'm01_150',
                 'water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice', 'flooded_vegetation',
                 'shrub',
                 'dw_built_150', 'dw_water_150', 'dw_bare_150', 'wc', 'modis_lc', 'dem', 'slope',
                 'relief500', 's2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong', 'veg', 'ndvi']
    out = d[[c for c in keep_cols if c in d.columns]]
    out.to_parquet(os.path.join(OUT, 'land_qc_%s.parquet' % tag), index=False)
    out[out.verdict == 'quarantine'].to_csv(os.path.join(QCD, 'quarantine_%s.csv' % tag),
                                            index=False, encoding='utf-8-sig')
    summ = dict(group=tag, n=int(len(d)), by_class=st,
                totals={v: int((d.verdict == v).sum()) for v in ('keep', 'quarantine', 'flag')},
                criteria_version=crit.get('version'), time=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(summ, open(os.path.join(OUT, 'stats_%s.json' % tag), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('=== %s ===' % tag)
    for cl, s in st.items():
        emit(' 类%-3s n=%-7d keep %6.2f%%  quar %5.2f%%  flag %5.2f%%' %
             (cl, s['n'], s['keep_pct'], s['quar_pct'], 100 - s['keep_pct'] - s['quar_pct']))
    emit('合计 %s' % summ['totals'])


if __name__ == '__main__':
    main()
