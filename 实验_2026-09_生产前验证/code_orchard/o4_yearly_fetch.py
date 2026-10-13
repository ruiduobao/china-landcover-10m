# -*- coding: utf-8 -*-
"""o4_yearly_fetch.py — 乔灌园地：母库园地点 2017-2024 年匹配 AEF64 面板（GEE，块级可续跑）

* 输入：data/orchard/mk_orchard_points.csv（row_id,lon,lat,src；母库园地点并集）
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL，逐年 mosaic（filterDate Y-01-01~Y+1-01-01），
          逐 band unmask(-9999) 防丢点；scale=10（与母库同源）
* 门槛：每（年,块）5000 点；块落盘 data/orchard/yearly_parts/y{Y}/aef_p{pp}.parquet（存在即跳过）
* 输出：data/orchard/yearly_parts/y{Y}/aef_p{pp}.parquet（row_id,lon,lat,src,A00-A63,aef_ok）
* 用法：python o4_yearly_fetch.py [--acct x] [--part 0 --nparts 4] [--years 2017,2018]
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
PTS = os.path.join(WORK, 'data', 'orchard', 'mk_orchard_points.csv')
OUTD = os.path.join(WORK, 'data', 'orchard', 'yearly_parts')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEARS = list(range(2017, 2025))
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def fetch_year_chunk(ee, pts, year):
    aef = (ee.ImageCollection(AEF).filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
           .mosaic().select(VC.FEATS))
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'row_id': int(r.row_id)})
             for r in pts.itertuples()]
    got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                            tileScale=4).getInfo()['features']
    recs = [f['properties'] for f in got]
    df = pd.DataFrame([{'row_id': int(r['row_id']), **{k: r.get(k) for k in VC.FEATS}} for r in recs])
    df['aef_ok'] = df[VC.FEATS].notna().all(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--years', default='2017,2018,2019,2020,2021,2022,2023,2024')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    d = pd.read_csv(PTS, encoding='utf-8-sig')
    d['row_id'] = d.row_id.astype(int)
    import ee
    VC.ensure_ctx(a.acct)
    todo = []
    for y in years:
        chunks = [d.iloc[i:i + CHUNK] for i in range(0, len(d), CHUNK)]
        for ci, c in enumerate(chunks):
            if ci % a.nparts != a.part:
                continue
            yd = os.path.join(OUTD, 'y%d' % y)
            os.makedirs(yd, exist_ok=True)
            fp = os.path.join(yd, 'aef_p%02d.parquet' % ci)
            if os.path.exists(fp):
                continue
            todo.append((y, ci, c, fp))
    emit('待抓 %d 块（点 %d × 年 %s）' % (len(todo), len(d), years))
    for y, ci, c, fp in todo:
        for attempt in range(4):
            try:
                df = fetch_year_chunk(ee, c, y)
                out = c[['row_id', 'lon', 'lat', 'src']].merge(df, on='row_id', how='left')
                out.to_parquet(fp, index=False)
                emit('  y%d 块%02d: %d/%d 命中' % (y, ci, int(out.aef_ok.sum()), len(c)))
                break
            except Exception as e:
                emit('  y%d 块%02d 重试%d: %s' % (y, ci, attempt, str(e)[:140]))
                time.sleep(10)
        else:
            emit('  y%d 块%02d 连续失败' % (y, ci))


if __name__ == '__main__':
    main()
