# -*- coding: utf-8 -*-
"""m4_mangrove_fetch.py — 红树林扩充 第4步：候选点证据抓取（GEE；分片幂等）

* 输入：data/mangrove/cand/candidates.parquet（cid/lon/lat）或 --list 指定
* 波段（两组，各一次 sampleRegions，scale=10）：
  证据组（20）：wc95（WC Map==95）、giri（Giri 2000==1）、mang10（两者之一）、
    gsw_occ/gsw_max/gsw_seas/gsw_rec、max_300/occ_300（300m 邻域）、dw 9 概率（water/trees/grass/crops/
    built/bare/snow_and_ice/flooded_vegetation/shrub）、dem/slope、s2_ndvi_xia/s2_ndwi_xia/s2_ndvi_dong
  AEF 组（64）：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023 的 A00..A63（逐 band unmask(-9999)）
* 幂等：out/feats/{ev,aef}/m_p{part}_c{chunk}.parquet；分片 --part/--nparts；块 1200。
* 用法：python m4_mangrove_fetch.py --part 0 --nparts 8 --acct x2zdn8
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
SENT = -9999.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=8)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--list', default=os.path.join(WORK, 'cand', 'candidates.parquet'))
    ap.add_argument('--step', type=int, default=1200)
    ap.add_argument('--tag', default='', help='输出子目录前缀（如 orig → feats/orig_ev）')
    a = ap.parse_args()
    import ee, v31_common as VC
    VC.ensure_ctx(a.acct)
    d = pd.read_parquet(a.list)
    d['cid'] = d.cid.astype(str)
    mine = d.iloc[a.part::a.nparts].reset_index(drop=True)
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    giri = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS').first().select('1')
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    half = gsw.select(['occurrence', 'max_extent']).unmask(0)
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
          .select(['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice',
                   'flooded_vegetation', 'shrub_and_scrub']).mean().unmask(-1)
          .rename(['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice',
                   'flooded_vegetation', 'shrub']))
    dem = ee.Image('USGS/SRTMGL1_003')
    slp = ee.Terrain.slope(dem).rename('slope')
    s2c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    def msk(im):
        s = im.select('SCL')
        return im.updateMask(s.neq(3).And(s.neq(8)).And(s.neq(9)))
    c = (s2c.filterDate('2023-06-01', '2023-09-01')
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk).median())
    s2x = ee.Image.cat([c.normalizedDifference(['B8', 'B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_xia'),
                        c.normalizedDifference(['B3', 'B8']).multiply(1000).unmask(SENT).rename('s2_ndwi_xia')])
    cd = (s2c.filterDate('2022-12-01', '2023-02-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk).median())
    s2d = cd.normalizedDifference(['B8', 'B4']).multiply(1000).unmask(SENT).rename('s2_ndvi_dong')
    ev = ee.Image.cat([
        wc.eq(95).unmask(0).rename('wc95'), giri.eq(1).unmask(0).rename('giri'),
        gsw.select(['occurrence', 'max_extent', 'seasonality', 'recurrence']).unmask(-1),
        half.focal_mean(radius=30, kernelType='circle').rename(['occ_300', 'max_300']),
        dw, dem.rename('dem').unmask(SENT), slp.unmask(SENT), s2x, s2d]).toFloat()
    ev = ev.addBands(ev.select('wc95').add(ev.select('giri')).gt(0).rename('mang10'))
    EV_COLS = ['wc95', 'giri', 'mang10', 'occurrence', 'max_extent', 'seasonality', 'recurrence',
               'occ_300', 'max_300', 'water', 'trees', 'grass', 'crops', 'built', 'bare',
               'snow_and_ice', 'flooded_vegetation', 'shrub', 'dem', 'slope',
               's2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong']
    aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate('2023-01-01', '2024-01-01').mosaic())
    aef = ee.Image.cat([aef.select(b).unmask(SENT).rename(b) for b in VC.FEATS]).toFloat()
    for stage, img, cols in (('ev', ev, EV_COLS), ('aef', aef, list(VC.FEATS))):
        outd = os.path.join(WORK, 'out', 'feats', (a.tag + '_' + stage) if a.tag else stage)
        os.makedirs(outd, exist_ok=True)
        nch = (len(mine) + a.step - 1) // a.step
        done = 0
        for k in range(0, len(mine), a.step):
            fp = os.path.join(outd, 'm_p%02d_c%04d.parquet' % (a.part, k // a.step))
            if os.path.exists(fp):
                done += 1; continue
            sub = mine.iloc[k:k + a.step]
            pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'cid': c_})
                   for c_, x, y in zip(sub.cid, sub.lon, sub.lat)]
            got, err = None, ''
            for att in range(5):
                try:
                    r = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=10,
                                          geometries=False, tileScale=4).getInfo()['features']
                    got = {f['properties']['cid']: f['properties'] for f in r}; break
                except Exception as e:
                    err = str(e)[:120]; emit('  %s 块%d 重试%d %s' % (stage, k // a.step, att + 1, err))
                    time.sleep(8 * (att + 1))
            if got is None:
                with open(os.path.join(outd, '_fail_p%02d.json' % a.part), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(stage=stage, part=a.part, chunk=k // a.step, err=err),
                                       ensure_ascii=False) + chr(10))
                continue
            o = pd.DataFrame([dict(cid=c_, **{cc: got.get(c_, {}).get(cc) for cc in cols}) for c_ in sub.cid])
            for cc in cols:
                o[cc] = pd.to_numeric(o[cc], errors='coerce')
            if stage in ('ev', 'aef'):
                for cc in ('dem', 'slope', 's2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong') + (tuple(VC.FEATS) if stage == 'aef' else ()):
                    if cc in o.columns:
                        o.loc[o[cc] <= -9000, cc] = np.nan
            o.to_parquet(fp, index=False); done += 1
        emit('完成 %s p%02d：%d/%d 块' % (stage, a.part, done, nch))


if __name__ == '__main__':
    main()
