# -*- coding: utf-8 -*-
"""v7_land_fetch.py — 其余地类样本质检的统一证据抓取（26 波段；逐块幂等）

* 输入：--list <parquet>（point_id/lon/lat/class24）
* 波段（一次 sampleRegions，scale=30）：
    GSW 点级：occurrence, max_extent, seasonality, recurrence（-1=从未有水/无数据，unmask(-1)）
    GSW 邻域（unmask(0) 后 focal）：occ_300, max_300, m07_150, m01_150（7/1 月 150m 水频）
    DW 2023 点级：water, trees, grass, crops, built, bare, snow_and_ice, flooded_vegetation（unmask(-1)）
    DW 邻域：dw_built_150, dw_water_150, dw_bare_150
    WC（WorldCover Map）+ MODIS LC_Type1
    SRTM：dem, slope, relief500（dem−focal_min 17px 方形，unmask(-9999)）
    S2 2023：s2_ndvi_xia, s2_ndwi_xia（夏 6-9）, s2_ndvi_dong（冬 12-2）
* 分片：--part/--nparts；块 4000；已存在块跳过
* 用法：python v7_land_fetch.py --list ../data/land_qc/lists/grp1_crop.parquet --part 0 --nparts 7 --acct ccstqqb
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'; BASE = os.path.join(WORK, 'data', 'land_qc'); SENT = -9999.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), str(m)), flush=True)


def build(ee):
    gswi = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    pt = ee.Image.cat([gswi.select(['occurrence','max_extent','seasonality','recurrence']).unmask(-1)])
    g0 = gswi.select(['occurrence','max_extent']).unmask(0)
    n3 = g0.focal_mean(radius=10, kernelType='circle').rename(['occ_300','max_300'])
    mh = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory').filterDate('2015-01-01','2022-01-01')
    def mon(m):
        c = mh.filter(ee.Filter.calendarRange(m, m, 'month')).map(lambda im: im.select('water').eq(2))
        return c.mean().unmask(0).focal_mean(radius=5, kernelType='circle').rename('m%02d_150' % m)
    moni = ee.Image.cat([mon(7), mon(1)])
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01','2024-01-01')
          .select(['water','trees','grass','crops','built','bare','snow_and_ice','flooded_vegetation',
                   'shrub_and_scrub'])
          .mean().unmask(-1)
          .rename(['water','trees','grass','crops','built','bare','snow_and_ice','flooded_vegetation','shrub']))
    dw150 = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
             .filterDate('2023-01-01','2024-01-01')
             .select(['built','water','bare']).mean().unmask(0)
             .focal_mean(radius=15, kernelType='circle')
             .rename(['dw_built_150','dw_water_150','dw_bare_150']))
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('wc').unmask(-1)
    mod = (ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2022-01-01','2023-01-01')
           .first().select('LC_Type1').rename('modis_lc').unmask(-1))
    dem = ee.Image('USGS/SRTMGL1_003')
    slp = ee.Terrain.slope(dem).rename('slope')
    r500 = dem.subtract(dem.focal_min(radius=17, kernelType='square')).rename('relief500')
    s2c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    def msk(im):
        scl = im.select('SCL'); return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
    def s2w(d0, d1):
        c = s2c.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60)).map(msk).median()
        return ee.Image.cat([c.normalizedDifference(['B8','B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_xia'),
                             c.normalizedDifference(['B3','B8']).multiply(1000).unmask(SENT).rename('s2_ndwi_xia')]).toFloat()
    s2x = s2w('2023-06-01','2023-09-01')
    cd = s2c.filterDate('2022-12-01','2023-02-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60)).map(msk).median()
    s2d = cd.normalizedDifference(['B8','B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_dong').toFloat()
    img = ee.Image.cat([pt, n3, moni, dw, dw150, wc, mod,
                        dem.rename('dem').unmask(SENT), slp.unmask(SENT), r500.unmask(SENT),
                        s2x, s2d]).toFloat()
    cols = ['occurrence','max_extent','seasonality','recurrence','occ_300','max_300','m07_150','m01_150',
            'water','trees','grass','crops','built','bare','snow_and_ice','flooded_vegetation','shrub',
            'dw_built_150','dw_water_150','dw_bare_150','wc','modis_lc',
            'dem','slope','relief500','s2_ndvi_xia','s2_ndwi_xia','s2_ndvi_dong']
    return img, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=7)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--step', type=int, default=4000)
    ap.add_argument('--sub', type=int, default=2000,
                    help='每块内部再切的请求粒度（4000 点单请求会撞 300s 交互上限；实测 2000 点≈80s）')
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(WORK, 'code'))
    import ee, v31_common as VC
    VC.ensure_ctx(a.acct)
    d = pd.read_parquet(a.list, columns=['point_id','lon','lat'])
    d['point_id'] = d.point_id.astype(str)
    mine = d.iloc[a.part::a.nparts].reset_index(drop=True)
    os.makedirs(a.outdir, exist_ok=True)
    img, cols = build(ee)
    nch = (len(mine) + a.step - 1) // a.step
    emit('本片 %d 点 / %d 块（每块 %d 点，请求粒度 %d）' % (len(mine), nch, a.step, a.sub))
    done = 0
    for k in range(0, len(mine), a.step):
        fp = os.path.join(a.outdir, 'land_p%02d_c%04d.parquet' % (a.part, k // a.step))
        if os.path.exists(fp):
            done += 1; continue
        sub_all = mine.iloc[k:k + a.step]
        rows, failed = [], False
        for s0 in range(0, len(sub_all), a.sub):
            part = sub_all.iloc[s0:s0 + a.sub]
            pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': p})
                   for p, x, y in zip(part.point_id, part.lon.astype(float), part.lat.astype(float))]
            got, err = None, ''
            for att in range(6):
                try:
                    r = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=30,
                                          geometries=False, tileScale=4).getInfo()['features']
                    got = {f['properties']['point_id']: f['properties'] for f in r}; break
                except Exception as e:
                    err = str(e)[:120]
                    emit('  块%d 子%d 重试%d %s' % (k // a.step, s0 // a.sub, att + 1, err))
                    time.sleep(8 * (att + 1))
            if got is None:
                with open(os.path.join(a.outdir, '_fail_p%02d.json' % a.part), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(part=a.part, chunk=k // a.step, sub=s0 // a.sub, err=err),
                                       ensure_ascii=False) + '\n')
                failed = True; break
            rows += [dict(point_id=p, **{c: got.get(p, {}).get(c) for c in cols}) for p in part.point_id]
        if failed:
            continue
        o = pd.DataFrame(rows)
        for c in cols:
            o[c] = pd.to_numeric(o[c], errors='coerce')
        for c in ('s2_ndvi_xia','s2_ndwi_xia','s2_ndvi_dong','dem','slope','relief500'):
            o.loc[o[c] <= -9000, c] = np.nan
        o.to_parquet(fp, index=False); done += 1
        if done % 4 == 0:
            emit('  %d/%d 块' % (done, nch))
    emit('完成 p%02d：%d/%d 块 / %d 点' % (a.part, done, nch, len(mine)))


if __name__ == '__main__':
    main()
