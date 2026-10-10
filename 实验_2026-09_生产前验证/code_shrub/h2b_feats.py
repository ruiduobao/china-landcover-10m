# -*- coding: utf-8 -*-
"""h2b_feats.py — 给入选点补取贵层特征（AEF64 + S2 物候 + GEDI 冠层高度邻域）

* 输入：--in  csv/parquet（至少含 point_id,lon,lat）+ 其余列原样带回
* 规则源：AEF2023 = GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023（生产同源特征）
        S2 = COPERNICUS/S2_SR_HARMONIZED，djf(2022-12~2023-03)/jja(2023-06~09)
             中位 NDVI（SCL 剔云影；CLOUDY_PIXEL_PERCENTAGE<60）
        GEDI = LARSE/GEDI/GEDI02_A_002_MONTHLY rh98 中位（2019-2024）+ 4 个对角 0.01°(~1.1km)
              偏移取 max 作 rh98_mx11（邻域最高冠层高度，识别附近乔木）
* 门槛：分块 4,000 点（getInfo 5,000 要素上限）；失败重试 5 次；缺 AEF 的点标记 aef_ok=0
* 输出：--out  parquet（原列 + aef_ok + A00..A63 + ndvi_djf,ndvi_jja,ndvi_amp,rh98,rh98_mx11）
* 用法：python h2b_feats.py --in 点表.csv --out 结果.parquet [--acct bx15mw]
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

OUTS = None


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def s2_band(col, d0, d1, tag):
    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
    c = (col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
    return c.median().normalizedDifference(['B8', 'B4']).rename(tag).multiply(1000).toFloat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--acct', default='bx15mw')
    ap.add_argument('--nbr', type=int, default=1, help='是否补邻域冠层高度（1=补，费 1 倍计算）')
    ap.add_argument('--acct0', default='bx15mw', help='多分片时的账号池起点')
    ap.add_argument('--part', type=int, default=-1, help='分片索引（-1=不分片）')
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    import ee as _ee
    ee = _ee
    VC.ensure_ctx(a.acct)

    if a.inp.endswith('.parquet'):
        df = pd.read_parquet(a.inp)
    else:
        df = pd.read_csv(a.inp, encoding='utf-8-sig')
    df['point_id'] = df['point_id'].astype(str)
    if a.part >= 0:
        df = df.iloc[a.part::a.nparts].reset_index(drop=True)
        emit('分片 %d/%d：%d 点' % (a.part, a.nparts, len(df)))
    # 输入已带 AEF/物候列（如母池导出）时先去掉，避免 merge 出 _x/_y 后缀
    drop = [c for c in df.columns if c in set(VC.FEATS) | {'ndvi_djf', 'ndvi_jja', 'ndvi_amp',
                                                           'rh98', 'rh98_mx11', 'aef_ok'}]
    if drop:
        df = df.drop(columns=drop)
        emit('剔除输入已有的特征列 %d 个' % len(drop))
    pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
           for pid, x, y in zip(df['point_id'], df['lon'].to_numpy(float), df['lat'].to_numpy(float))]
    emit('入选点 %d' % len(df))

    def s2_band_local(col, d0, d1, tag):
        def msk(im):
            scl = im.select('SCL')
            return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
        c = (col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
        return c.median().normalizedDifference(['B8', 'B4']).rename(tag).multiply(1000).toFloat()

    s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate('2023-01-01', '2024-01-01').mosaic()
    gedi_global = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').select('rh98') \
        .median().rename('rh98').unmask(-1)
    main_img = ee.Image.cat([
        aef.select(VC.FEATS),
        s2_band_local(s2col, '2022-12-01', '2023-03-01', 'ndvi_djf'),
        s2_band_local(s2col, '2023-06-01', '2023-09-01', 'ndvi_jja'),
        gedi_global,
    ]).toFloat()
    want = list(VC.FEATS) + ['ndvi_djf', 'ndvi_jja', 'rh98']

    got = []
    step = 4000
    for k in range(0, len(pts), step):
        fc = ee.FeatureCollection(pts[k:k + step])
        for att in range(5):
            try:
                r = main_img.sampleRegions(collection=fc, scale=20, geometries=False, tileScale=4).getInfo()['features']
                got += [f['properties'] for f in r]
                break
            except Exception as e:
                emit('  主栈块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                time.sleep(8 * (att + 1))
        emit('  特征 %d/%d' % (min(k + step, len(pts)), len(pts)))
    fe = pd.DataFrame(got)
    if len(fe):
        for c in want:
            if c not in fe.columns:
                fe[c] = np.nan
        fe = fe[['point_id'] + want]
    if a.nbr:
        # 邻域最高冠层：四对角 ±0.01°
        nbf = []
        for i, (dlon, dlat) in enumerate(((-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01))):
            fpts = [ee.Feature(ee.Geometry.Point([float(x + dlon), float(y + dlat)]),
                               {'point_id': pid}) for pid, x, y in
                    zip(df['point_id'], df['lon'].to_numpy(float), df['lat'].to_numpy(float))]
            for k in range(0, len(fpts), step):
                fc = ee.FeatureCollection(fpts[k:k + step])
                for att in range(5):
                    try:
                        r = gedi_global.sampleRegions(collection=fc, scale=50, geometries=False,
                                                     tileScale=4).getInfo()['features']
                        nbf += [dict(point_id=f['properties']['point_id'],
                                     **{('rh98' if i == 0 else 'rh98_%d' % i):
                                        f['properties'].get('rh98', np.nan)})
                                for f in r]
                        break
                    except Exception as e:
                        emit('  邻域块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                        time.sleep(8 * (att + 1))
        if nbf:
            nb = pd.DataFrame(nbf).groupby('point_id').first()
            cols = [c for c in nb.columns if c.startswith('rh98')]
            nb['rh98_mx11'] = nb[cols].max(axis=1)
            nb = nb[['rh98_mx11']].reset_index()
            fe = fe.merge(nb, on='point_id', how='left')
    out = df.merge(fe, on='point_id', how='left')
    out['aef_ok'] = out['A00'].notna().astype(int)
    out['ndvi_amp'] = (out['ndvi_jja'].astype(float) - out['ndvi_djf'].astype(float)).abs()
    ofp = a.out
    if a.part >= 0:
        base, ext = os.path.splitext(a.out)
        ofp = base + '_p%02d' % a.part + ext
    out.to_parquet(ofp, index=False)
    emit('→ %s（%d 行，AEF 命中 %d）' % (a.out, len(out), int(out['aef_ok'].sum())))


if __name__ == '__main__':
    main()
