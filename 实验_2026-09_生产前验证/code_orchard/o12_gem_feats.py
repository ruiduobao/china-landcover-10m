# -*- coding: utf-8 -*-
"""o12_gem_feats.py — GEM 候选点 AEF64 特征（2020 + 2023，GEE，块级可续跑）

* 输入：data/orchard/gem_cn.parquet（中国境内候选 610 点）
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL，2020 与 2023 年 mosaic；scale=10；逐 band unmask(-9999)
* 输出：data/orchard/gem/feats_y{2020,2023}.parquet（lon,lat,prov,A00-A63,aef_ok）
* 用法：python o12_gem_feats.py [--acct zixen8v8]
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    d = pd.read_parquet(os.path.join(ORD, 'gem_cn.parquet'))
    gd = os.path.join(ORD, 'gem')
    import ee
    VC.ensure_ctx('zixen8v8')
    for year in (2020, 2023):
        fp = os.path.join(gd, 'feats_y%d.parquet' % year)
        if os.path.exists(fp):
            emit('y%d 已存在' % year)
            continue
        aef = (ee.ImageCollection(AEF).filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
               .mosaic().select(VC.FEATS))
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'i': i})
                 for i, r in enumerate(d.itertuples())]
        for attempt in range(4):
            try:
                got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                                        tileScale=4).getInfo()['features']
                recs = [{'i': g['properties']['i'], **{k: g['properties'].get(k) for k in VC.FEATS}} for g in got]
                df = pd.DataFrame(recs)
                d2 = d.reset_index(drop=True)
                d2['i'] = np.arange(len(d2))
                out = d2.merge(df, on='i', how='left')
                out['aef_ok'] = out[VC.FEATS].notna().all(axis=1)
                out = out.drop(columns=['i', 'index'], errors='ignore')
                out.to_parquet(fp, index=False)
                emit('y%d: %d/%d 命中' % (year, int(out.aef_ok.sum()), len(d)))
                break
            except Exception as e:
                emit('y%d 重试%d: %s' % (year, attempt, str(e)[:140]))
                time.sleep(10)


if __name__ == '__main__':
    main()
