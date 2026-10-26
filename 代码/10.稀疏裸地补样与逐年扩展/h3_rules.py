# -*- coding: utf-8 -*-
"""h3_rules.py — 候选点质量裁定（预注册判据 → keep / quarantine / flag）

* 输入：--cand 候选 parquet（逗号分隔）；--ev 证据目录（h2 的 ev_p*_b*.parquet）
* 判据源：F:/lc_work/bare_sparse_v3/spec/预注册判据.md §3（沿用 doc67 v1.1 两类口径）
    稀疏植被 keep ⇔ (NDVI≥60 或 shrub≥0.25) 且 NDVI<400 且 water<0.40 且 built<0.30
                 且 NDVI≥30 且 WC∉{70,80,100}
    裸地     keep ⇔ NDVI<200 且 water<0.30 且 built<0.30 且 snow<0.30
                 且 GSW occurrence<0.30 且 crops<0.35
    quarantine ⇔ NDVI≥450(稀疏)/350(裸地) 或 water≥0.5 或 built≥0.5 或 snow≥0.5
    flag ⇔ 关键证据缺失（NDVI 无值）
* 输出：--outdir/cand_qc.parquet（含 status/reason/ndvi 等），qc_stats.json
* 用法：python h3_rules.py --cand ../data/candidates/cand_north.parquet,../data/candidates/cand_east.parquet \
        --ev ../data/evidence --outdir ../data/final
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SENT = -9999.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--ev', required=True)
    ap.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'final'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cand = pd.concat([pd.read_parquet(p) for p in a.cand.split(',')], ignore_index=True)
    cand['pid'] = [f'{r.fam}_{r.lon:.6f}_{r.lat:.6f}' for r in cand.itertuples()]
    fs = sorted(glob.glob(os.path.join(a.ev, 'ev_p*_b*.parquet')))
    if not fs:
        raise SystemExit('缺证据块：%s' % a.ev)
    ev = pd.concat([pd.read_parquet(fp) for fp in fs], ignore_index=True)
    ev = ev.drop_duplicates('pid')
    print('候选 %d ｜ 证据块 %d ｜ 证据点 %d' % (len(cand), len(fs), len(ev)), flush=True)
    d = cand.merge(ev.drop(columns=['lon', 'lat', 'fam', 'class'], errors='ignore'),
                   on='pid', how='left')

    # 证据归一
    for c in ['s2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong', 'dem', 'slope']:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce').where(lambda x: x > -9000)
    d['ndvi'] = d['s2_ndvi_xia'].fillna(d['s2_ndvi_dong'])
    for c in ['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice',
              'flooded_vegetation', 'shrub']:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(-1).clip(0, 1)

    status = np.full(len(d), 'flag', dtype=object)
    reason = np.full(len(d), '', dtype=object)

    sp = d.fam == 'sparse'
    keep_sp = ((d['ndvi'] >= 60) | (d['shrub'] >= 0.25)) & (d['ndvi'] < 400) & \
              (d['water'] < 0.40) & (d['built'] < 0.30) & (d['ndvi'] >= 30) & \
              (~d['wc'].isin([70, 80, 100]))
    quar_sp = (d['ndvi'] >= 450) | (d['water'] >= 0.5) | (d['built'] >= 0.5) | \
              (d['snow_and_ice'] >= 0.5)
    bare = d.fam == 'bare'
    keep_ba = (d['ndvi'] < 200) & (d['water'] < 0.30) & (d['built'] < 0.30) & \
              (d['snow_and_ice'] < 0.30) & (d['occurrence'] < 0.30) & (d['crops'] < 0.35)
    quar_ba = (d['ndvi'] >= 350) | (d['water'] >= 0.5) | (d['built'] >= 0.5) | \
              (d['snow_and_ice'] >= 0.5)

    status[sp & quar_sp] = 'quarantine'; reason[sp & quar_sp] = 'sparse:quarantine'
    status[bare & quar_ba] = 'quarantine'; reason[bare & quar_ba] = 'bare:quarantine'
    status[sp & keep_sp & ~quar_sp] = 'keep'; reason[sp & keep_sp & ~quar_sp] = 'sparse:keep'
    status[bare & keep_ba & ~quar_ba] = 'keep'; reason[bare & keep_ba & ~quar_ba] = 'bare:keep'
    no_ev = d['ndvi'].isna()
    status[no_ev] = 'flag'; reason[no_ev] = 'missing_ndvi'
    d['status'] = status; d['reason'] = reason

    d.to_parquet(os.path.join(a.outdir, 'cand_qc.parquet'), index=False)
    stats = {}
    for (fam, sub), g in d.groupby(['fam', 'subwin']):
        stats.setdefault(fam, {})[str(int(sub))] = {
            'n': int(len(g)), 'keep': int((g.status == 'keep').sum()),
            'quarantine': int((g.status == 'quarantine').sum()),
            'flag': int((g.status == 'flag').sum())}
    tot = {fam: {'n': int((d.fam == fam).sum()),
                 'keep': int(((d.fam == fam) & (d.status == 'keep')).sum()),
                 'quarantine': int(((d.fam == fam) & (d.status == 'quarantine')).sum()),
                 'flag': int(((d.fam == fam) & (d.status == 'flag')).sum())}
           for fam in ('sparse', 'bare')}
    json.dump({'total': tot, 'by_subwin': stats}, open(os.path.join(a.outdir, 'qc_stats.json'),
              'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('总计:', tot, flush=True)
    print('→', os.path.join(a.outdir, 'cand_qc.parquet'), flush=True)


if __name__ == '__main__':
    main()
