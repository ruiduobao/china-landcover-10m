# -*- coding: utf-8 -*-
"""f8_eco_features.py — 给判读参考点补"生态特征"（bio01/bio12/elev/slope/水频率），供特征集对比

* 输入：data/m3/gap_points.csv、refset_all.csv、gsnx_ref_judged.csv（判读点）
* 规则源：与生产 GEE 侧同源——WORLDCLIM/V1/BIO（bio01/bio12）、USGS/SRTMGL1_003（elev/slope）、
        JRC/GSW1_4/GlobalSurfaceWater（occurrence，点 + 150 m 邻域最大值）
* 门槛：每批 ≤3000 点
* 输出：data/m3/eco_features_ref.csv（point_id + bio01,bio12,elev,slope,occ_pt,occ_90）
* 用法：python f8_eco_features.py [--acct bx15mw]
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd, v31_common as VC
M3 = os.path.join(WORK, 'data', 'm3')
OUT = os.path.join(M3, 'eco_features_ref.csv')

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acct', default='bx15mw'); a = ap.parse_args()
    import ee
    VC.ensure_ctx(a.acct)
    dfs = []
    for f in ('gap_points.csv', 'refset_all.csv', 'gsnx_ref_judged.csv'):
        p = os.path.join(M3, f)
        if os.path.exists(p):
            d = pd.read_csv(p, encoding='utf-8-sig')
            d = d[['point_id', 'lon', 'lat']].copy()
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates('point_id')
    emit('判读点合计 %d' % len(df))
    # 分三批取（各波段掩膜不同，合在一起会让缺任一波段的点整体丢失——2026-10-10 实测 279/1581）
    dem = ee.Image('USGS/SRTMGL1_003')
    groups = [
        ('dem', ee.Image.cat([dem.rename('elev'), ee.Terrain.slope(dem).rename('slope')]), ['elev', 'slope']),
        ('bio', ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12']), ['bio01', 'bio12']),
        ('occ', ee.Image.cat([
            ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').rename('occ_pt'),
            ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
            .reduceNeighborhood(ee.Reducer.max(), ee.Kernel.square(2, 'pixels')).rename('occ_90')]).unmask(0),
         ['occ_pt', 'occ_90']),
    ]
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    frames = {}
    step = 3000
    for tag, img, want in groups:
        rows = []
        for k in range(0, len(pts), step):
            for att in range(4):
                try:
                    got = img.sampleRegions(collection=ee.FeatureCollection(pts[k:k+step]),
                                            scale=100, geometries=False, tileScale=4).getInfo()['features']
                    rows += [dict(point_id=f['properties']['point_id'],
                                  **{c: f['properties'].get(c) for c in want}) for f in got]
                    break
                except Exception as e:
                    emit('  %s 重试%d %s' % (tag, att+1, str(e)[:80])); time.sleep(10)
        frames[tag] = pd.DataFrame(rows)
        emit('  %s 命中 %d/%d' % (tag, len(frames[tag]), len(pts)))
    out = df[['point_id']].copy()
    for tag in ('dem', 'bio', 'occ'):
        out = out.merge(frames[tag], on='point_id', how='left')
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    emit('生态特征 %d 点 → %s' % (len(out), OUT))
    emit('  覆盖：elev %d, bio01 %d, occ_90 %d' % (
        int(out.elev.notna().sum()) if len(out) else 0,
        int(out.bio01.notna().sum()) if len(out) else 0,
        int(out.occ_90.notna().sum()) if len(out) else 0))

if __name__ == '__main__': main()
