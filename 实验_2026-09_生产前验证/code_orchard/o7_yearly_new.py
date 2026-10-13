# -*- coding: utf-8 -*-
"""o7_yearly_new.py — 乔灌园地：新增选点 2017-2024 年匹配 AEF64 抓取（块级可续跑）

* 输入：data/orchard/orchard_samples_2023.csv（o6 选点，非 mk 部分）
* 规则源：同 o4（GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 逐年 mosaic，scale=10，逐 band unmask）
* 门槛：每（年,块）5000 点；块落盘 data/orchard/yearly_new/y{Y}/aef_p{pp}.parquet
* 输出：同上（point_id,lon,lat,prov,tier,A00-A63,aef_ok）
* 用法：python o7_yearly_new.py [--part 0 --nparts 4 --acct x]
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
SEL = os.path.join(WORK, 'data', 'orchard', 'orchard_samples_2023.csv')
OUTD = os.path.join(WORK, 'data', 'orchard', 'yearly_new')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--years', default='2017,2018,2019,2020,2021,2022,2023,2024')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    d = pd.read_csv(SEL, encoding='utf-8-sig')
    d = d[d.tier != 'P0_legacy'].reset_index(drop=True)
    if not len(d):
        emit('无新增选点（请先跑 o6）')
        return
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
            if not os.path.exists(fp):
                todo.append((y, ci, c, fp))
    emit('新增点 %d × 年 %s；待抓 %d 块' % (len(d), years, len(todo)))
    for y, ci, c, fp in todo:
        for attempt in range(4):
            try:
                aef = (ee.ImageCollection(AEF).filterDate('%d-01-01' % y, '%d-01-01' % (y + 1))
                       .mosaic().select(VC.FEATS))
                feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                    {'point_id': r.point_id}) for r in c.itertuples()]
                got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                                        tileScale=4).getInfo()['features']
                df = pd.DataFrame([{'point_id': r['properties']['point_id'],
                                    **{k: r['properties'].get(k) for k in VC.FEATS}} for r in got])
                out = c[['point_id', 'lon', 'lat', 'prov', 'tier']].merge(df, on='point_id', how='left')
                out['aef_ok'] = out[VC.FEATS].notna().all(axis=1)
                out.to_parquet(fp, index=False)
                emit('  y%d 块%02d: %d/%d → %s' % (y, ci, int(out.aef_ok.sum()), len(c), os.path.basename(fp)))
                break
            except Exception as e:
                emit('  y%d 块%02d 重试%d: %s' % (y, ci, attempt, str(e)[:140]))
                time.sleep(10)
        else:
            emit('  y%d 块%02d 连续失败' % (y, ci))


if __name__ == '__main__':
    main()
