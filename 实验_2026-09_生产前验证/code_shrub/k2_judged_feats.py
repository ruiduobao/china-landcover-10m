# -*- coding: utf-8 -*-
"""k2_judged_feats.py — 给 1,852 个目视判读点取全量特征（分层源 + 物理层 + AEF64 + 物候）

* 输入：data/shrub_nat/judged_pool.csv（k1 产物）
* 规则源：与全国候选同源同参（保证规则可迁移）——
        分层源：DW v1 2023（label 众数 + shrub_and_scrub 年均）、WC v200(20)、LC100(20)、MODIS IGBP6/7
        物理层：Hansen treecover2000、GEDI rh98（点值 + 四对角 1.1 km 最大值）、SRTM DEM/坡度、
                WorldClim bio01/bio12、JRC GSW occurrence
        贵层：AEF64（2023 年度嵌入口径，与生产同源）、S2 djf/jja 中位 NDVI
* 门槛：分块 2,000 点；任一波段 unmask 后再 cat（避免 sampleRegions 丢点）；失败重试 5 次
* 输出：data/shrub_nat/judged_feats.parquet
* 用法：python k2_judged_feats.py [--acct bx15mw]
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def s2_band(col, d0, d1, tag):
    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
    c = (col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
    return c.median().normalizedDifference(['B8', 'B4']).rename(tag).multiply(1000).toFloat()


def main():
    global ee
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    import ee as _ee
    ee = _ee
    VC.ensure_ctx(a.acct)
    df = pd.read_csv(os.path.join(OUTD, 'judged_pool.csv'), encoding='utf-8-sig')
    df['point_id'] = df['point_id'].astype(str)
    pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
           for pid, x, y in zip(df['point_id'], df['lon'].to_numpy(float), df['lat'].to_numpy(float))]
    emit('判读点 %d' % len(df))
    s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    def cheap_stack():
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2023-12-31')
        dwi = ee.Image.cat([dw.select('label').mode().rename('dwl'),
                            dw.select('shrub_and_scrub').mean().rename('dwp')])
        wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        lc = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification')
        mcd = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2023-01-01', '2023-12-31').first().select('LC_Type1')
        dem = ee.Image('USGS/SRTMGL1_003')
        tc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').select('treecover2000')
        occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
        bio = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'])
        return ee.Image.cat([dwi.unmask(-999), wc.eq(20).unmask(0).rename('wc20'),
                             lc.eq(20).unmask(0).rename('lc20'),
                             mcd.eq(6).Or(mcd.eq(7)).unmask(0).rename('mcd'), tc.unmask(-1).rename('tc'),
                             dem.unmask(-999).rename('dem'), ee.Terrain.slope(dem).unmask(-999).rename('slope'),
                             bio.unmask(-999), occ.unmask(-1).rename('occ')]).toFloat()

    def dear_stack():
        aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
            .filterDate('2023-01-01', '2024-01-01').mosaic()
        gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').select('rh98').median().rename('rh98').unmask(-1)
        return ee.Image.cat([aef.select(VC.FEATS),
                             s2_band(s2col, '2022-12-01', '2023-03-01', 'ndvi_djf'),
                             s2_band(s2col, '2023-06-01', '2023-09-01', 'ndvi_jja'),
                             gedi]).toFloat()

    cs, ds = cheap_stack(), dear_stack()
    cwant = ['dwp', 'dwl', 'wc20', 'lc100', 'mcd', 'tc', 'dem', 'slope', 'bio01', 'bio12', 'occ']
    dwant = list(VC.FEATS) + ['ndvi_djf', 'ndvi_jja', 'rh98']
    got = {t: [] for t in ('c', 'd')}
    for k in range(0, len(pts), 2000):
        fc = ee.FeatureCollection(pts[k:k + 2000])
        for tag, img, want in (('c', cs, cwant), ('d', ds, dwant)):
            for att in range(5):
                try:
                    r = img.sampleRegions(collection=fc, scale=20, geometries=False, tileScale=4).getInfo()['features']
                    got[tag] += [f['properties'] for f in r]
                    break
                except Exception as e:
                    emit('  %s 块 %d 重试%d %s' % (tag, k, att + 1, str(e)[:70]))
                    time.sleep(8 * (att + 1))
        emit('  便宜层 %d/%d；贵层 %d/%d' % (min(k + 2000, len(pts)), len(pts),
                                            min(k + 2000, len(pts)), len(pts)))
    fc_ = pd.DataFrame(got['c']).drop_duplicates(subset=['point_id'])
    fd_ = pd.DataFrame(got['d']).drop_duplicates(subset=['point_id'])
    fd_ = fd_[['point_id'] + [c for c in dwant if c in fd_.columns]]
    out = df.merge(fc_, on='point_id', how='left').merge(fd_, on='point_id', how='left')
    out['aef_ok'] = out['A00'].notna().astype(int)
    out['ndvi_amp'] = (out['ndvi_jja'].astype(float) - out['ndvi_djf'].astype(float)).abs()
    # 邻域最高冠层高度：四对角 ±0.01°
    nb = {}
    for i, (dlon, dlat) in enumerate(((-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01))):
        fpts = [ee.Feature(ee.Geometry.Point([float(x + dlon), float(y + dlat)]), {'point_id': pid})
                for pid, x, y in zip(df['point_id'], df['lon'].to_numpy(float), df['lat'].to_numpy(float))]
        for k in range(0, len(fpts), 2000):
            for att in range(5):
                try:
                    r = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').select('rh98').median().rename('rh98').unmask(-1) \
                        .sampleRegions(collection=ee.FeatureCollection(fpts[k:k + 2000]), scale=50,
                                      geometries=False, tileScale=4).getInfo()['features']
                    for f in r:
                        nb.setdefault(f['properties']['point_id'], []).append(f['properties'].get('rh98', np.nan))
                    break
                except Exception as e:
                    emit('  邻域块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                    time.sleep(8 * (att + 1))
    if nb:
        out['rh98_mx11'] = [max(v) if v else np.nan for v in nb.get(out['point_id'], [np.nan])] if False else \
            [max(nb.get(p, [np.nan])) if p in nb else np.nan for p in out['point_id']]
    out.to_parquet(os.path.join(OUTD, 'judged_feats.parquet'), index=False)
    emit('→ judged_feats.parquet（%d 行，AEF 命中 %d，非空 dwp %d）' % (
        len(out), int(out['aef_ok'].sum()), int(out['dwp'].notna().sum())))


if __name__ == '__main__':
    main()
