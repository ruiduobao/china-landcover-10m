# -*- coding: utf-8 -*-
"""o21_aef_fetch.py — 茶园/橡胶候选 AEF64（2023）抓取（GEE，块级可续跑，多进程分片）

* 输入：--src tea|rubber → data/orchard/{tea_2022_cn_points,rubber_sea_cn_points}.parquet
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023 mosaic，scale=10，逐 band unmask(-9999)
* 输出：data/orchard/aef/{tea,rubber}_p{ci}.parquet（lon,lat,A00-A63,aef_ok）
* 用法：python o21_aef_fetch.py --src tea [--part 0 --nparts 3 --acct x]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
OUTD = os.path.join(WORK, 'data', 'orchard', 'aef')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, choices=['tea', 'rubber', 'aomc', 'citrus'])
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    fp = os.path.join(WORK, 'data', 'orchard',
                      {'tea': 'tea_2022_cn_points.parquet', 'rubber': 'rubber_sea_cn_points.parquet',
                       'aomc': 'aomc_apple_points.parquet', 'citrus': 'citrus_2020_cn_points.parquet'}[a.src])
    d = pd.read_parquet(fp).reset_index(drop=True)
    chunks = [d.iloc[i:i + CHUNK] for i in range(0, len(d), CHUNK)]
    mine = [(i, c) for i, c in enumerate(chunks) if i % a.nparts == a.part]
    emit('%s：%d 点 / %d 块；本进程 %d 块' % (a.src, len(d), len(chunks), len(mine)))
    if not mine:
        return
    import ee
    VC.ensure_ctx(a.acct)
    aef = (ee.ImageCollection(AEF).filterDate('2023-01-01', '2024-01-01')
           .mosaic().select(VC.FEATS))
    for ci, c in mine:
        out = os.path.join(OUTD, '%s_p%02d.parquet' % (a.src, ci))
        if os.path.exists(out):
            emit('  块 %02d 已存在' % ci)
            continue
        for attempt in range(4):
            try:
                feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'i': int(i)})
                         for i, r in c.iterrows()]
                got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10,
                                        geometries=False, tileScale=4).getInfo()['features']
                df = pd.DataFrame([{'i': g['properties']['i'], **{k: g['properties'].get(k) for k in VC.FEATS}}
                                   for g in got])
                m = c.reset_index().rename(columns={'index': 'i'}).merge(df, on='i', how='left')
                m['aef_ok'] = m[VC.FEATS].notna().all(axis=1)
                m.drop(columns=['i'], errors='ignore').to_parquet(out, index=False)
                emit('  块 %02d: %d/%d 命中' % (ci, int(m.aef_ok.sum()), len(c)))
                break
            except Exception as e:
                emit('  块 %02d 重试%d: %s' % (ci, attempt, str(e)[:140]))
                time.sleep(10)


if __name__ == '__main__':
    main()
