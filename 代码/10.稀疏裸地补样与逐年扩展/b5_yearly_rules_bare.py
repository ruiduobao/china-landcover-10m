# -*- coding: utf-8 -*-
"""b5_yearly_rules_bare.py — 南方多源裸地候选的逐年可用性判定（2017–2024）

* 输入：data/final/bare_south_cand.parquet；data/evidence_yearly_bare/<year>/；data/aef_yearly_bare/<year>/
* 规则（与 2023 裸地口径一致 + 正向证据）：
    keep ⇔ NDVI<200 且 water<0.30 且 built<0.30 且 snow<0.30 且 crops<0.35 且 DW bare≥0.25
    NDVI：夏(6–9) → 宽窗(4–11) → 冬 依次兜底；缺证据年继承最近年
* 输出：data/yearly/topup_bare_south_<year>.parquet（pid/lon/lat/cl/fam/A00..A63/ndvi/evidence）
        data/yearly/bare_south_counts.csv
* 用法：python b5_yearly_rules_bare.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
YEARS = list(range(2017, 2025))
FEATS = ['A%02d' % i for i in range(64)]


def main():
    fin = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'bare_south_cand.parquet'))
    fin = fin[['pid', 'lon', 'lat']].copy()
    fin['cl'] = 22
    fin['fam'] = 'bare'
    rows, counts = [], []
    for y in YEARS:
        fs = sorted(glob.glob(os.path.join(BASE, 'data', 'evidence_yearly_bare', str(y), '*.parquet')))
        if not fs:
            print('%d 缺证据' % y); continue
        ev = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates('pid')
        d = fin.merge(ev.drop(columns=[c for c in ('lon', 'lat') if c in ev.columns]),
                      on='pid', how='left')
        for c in ['water', 'built', 'crops', 'snow_and_ice', 'bare']:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(-1).clip(0, 1)
        for c in ['s2_ndvi_xia', 's2_ndvi_wide', 's2_ndvi_dong']:
            d[c] = pd.to_numeric(d[c], errors='coerce').where(lambda x: x > -9000)
        d['ndvi'] = d.s2_ndvi_xia.fillna(d.s2_ndvi_wide).fillna(d.s2_ndvi_dong)
        d['used'] = np.where(d.s2_ndvi_xia.notna(), 'summer',
                             np.where(d.s2_ndvi_wide.notna(), 'wide',
                                      np.where(d.s2_ndvi_dong.notna(), 'winter', 'none')))
        keep = ((d['ndvi'] < 200) & (d['water'] < 0.30) & (d['built'] < 0.30) &
                (d['snow_and_ice'] < 0.30) & (d['crops'] < 0.35) & (d['bare'] >= 0.25))
        d['status'] = np.where(keep, 'keep', 'drop')
        d['evidence'] = np.where(d['ndvi'].notna(), 'direct', 'none')
        d['year'] = y
        rows.append(d[['pid', 'lon', 'lat', 'cl', 'fam', 'status', 'evidence', 'used',
                       'ndvi', 'water', 'built', 'crops', 'snow_and_ice', 'bare', 'year']])
        counts.append({'year': y, 'n': len(d), 'keep': int(keep.sum()), 'no_ev': int((d.evidence == 'none').sum()),
                       'band': {k: int(v) for k, v in d.used.value_counts().items()}})
        print('%d: keep %d/%d | band %s' % (y, int(keep.sum()), len(d), counts[-1]['band']), flush=True)
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # 缺证据继承最近年
    for pid, g in panel.groupby('pid'):
        direct = g[g.evidence == 'direct']
        if direct.empty:
            continue
        dy = direct.year.to_numpy(); ds = direct.status.to_numpy()
        for i, r in g.iterrows():
            if r.evidence == 'direct':
                continue
            k = int(np.argmin(np.abs(dy - r.year)))
            panel.loc[i, 'status'] = ds[k]
            panel.loc[i, 'evidence'] = 'inherit(%d)' % dy[k]
    # 逐年层（含 AEF）
    out_rows = []
    for y in YEARS:
        g = panel[(panel.year == y) & (panel.status == 'keep')]
        aef = sorted(glob.glob(os.path.join(BASE, 'data', 'aef_yearly_bare', str(y), '*.parquet')))
        if not aef or g.empty:
            continue
        ae = pd.concat([pd.read_parquet(f) for f in aef], ignore_index=True).drop_duplicates('pid')
        o = g.merge(ae[['pid'] + FEATS], on='pid', how='left')
        miss = int(o[FEATS].isna().all(axis=1).sum())
        o['year'] = y
        o.to_parquet(os.path.join(BASE, 'data', 'yearly', 'topup_bare_south_%d.parquet' % y), index=False)
        out_rows.append({'year': y, 'n_keep': int(len(o)), 'aef_missing': miss})
        print('%d 逐年层 %d 点（AEF 缺失 %d）' % (y, len(o), miss), flush=True)
    pd.DataFrame(counts).to_csv(os.path.join(BASE, 'data', 'yearly', 'bare_south_counts.csv'),
                                index=False, encoding='utf-8-sig')
    print(pd.DataFrame(out_rows).to_string(index=False))


if __name__ == '__main__':
    main()
