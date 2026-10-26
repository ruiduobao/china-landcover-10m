# -*- coding: utf-8 -*-
"""b1_candidates_gee.py — 南方/东部裸地多源候选（GEE，逐 1° 块幂等）

* 目标：补 FCS10/CLCD 在湿润区缺失的"裸地"——裸露矿/采石场、裸土工地、裸岩/石漠化
* 多源掩膜（S2/DW 10 m，WC/GSW/MODIS 辅助）：
    DW 2023: bare≥0.40, water<0.15, built<0.50, crops<0.30, trees<0.25
    S2 2023 夏: NDVI<0.28, NDWI<0.05, 亮度(B2/B3/B4 均值)>0.10
    WC v200: ∉{80 水, 90 湿地, 95 红树林, 100 苔原}
    GSW occurrence<0.05（非历史水面）
* 每 1° 块随机采样（掩膜外自动丢弃），带坐标
* 输出：data/bare_cand/<tag>/bc_p<part>_b<block>.parquet
* 用法：python b1_candidates_gee.py --tag south --acct <acct> --pid <pid> --part 0 --nparts 4
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')
BASE = r'F:/lc_work/bare_sparse_v3'
REGIONS = {
    'south': (102.0, 20.0, 116.0, 30.0),
    'east': (112.0, 26.0, 122.0, 32.0),
    'north_plain': (113.0, 32.0, 122.0, 40.5),
    'loess': (107.0, 34.0, 113.5, 38.5),
}
DWN = ['bare', 'water', 'built', 'crops', 'trees']
COLS = ['pid', 'lon', 'lat'] + DWN + ['s2_ndvi_xia', 's2_ndwi_xia', 's2_bright',
                                      'wc', 'gsw', 'modis_lc', 'dem']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build(ee):
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01', '2024-01-01').select(DWN).mean())
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate('2023-06-01', '2023-09-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
          .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                        .And(im.select('SCL').neq(9)))).median())
    ndvi = s2.normalizedDifference(['B8', 'B4']).multiply(1000).rename('s2_ndvi_xia')
    ndwi = s2.normalizedDifference(['B3', 'B8']).multiply(1000).rename('s2_ndwi_xia')
    bright = s2.select(['B2', 'B3', 'B4']).divide(10000).reduce(ee.Reducer.mean()).multiply(1000).rename('s2_bright')
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('wc').unmask(-1)
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(-1).rename('gsw')
    modis = (ee.ImageCollection('MODIS/061/MCD12Q1').filterDate('2022-01-01', '2023-01-01')
             .first().select('LC_Type1').rename('modis_lc').unmask(-1))
    dem = ee.Image('USGS/SRTMGL1_003').rename('dem').unmask(-9999)
    img = ee.Image.cat([dw, ndvi, ndwi, bright, wc, gsw, modis, dem])
    # 多源并集：任一裸地证据成立，再排除水/密建/耕地/林地/湿地
    bare_ev = (dw.select('bare').gte(0.30).Or(wc.eq(60)).Or(modis.eq(16))
               .Or(ndvi.lt(220).And(bright.gt(130)).And(ndwi.lt(0))))
    excl = (dw.select('water').gte(0.20).Or(dw.select('built').gte(0.55))
            .Or(dw.select('crops').gte(0.45)).Or(dw.select('trees').gte(0.50))
            .Or(gsw.gte(0.10)).Or(wc.eq(80)).Or(wc.eq(90)).Or(wc.eq(95)).Or(wc.eq(100)))
    mask = bare_ev.And(excl.Not())
    return img.updateMask(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, choices=list(REGIONS))
    ap.add_argument('--acct', required=True)
    ap.add_argument('--pid', default=None)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--per-block', type=int, default=3000)
    a = ap.parse_args()
    import ee
    outdir = os.path.join(BASE, 'data', 'bare_cand', a.tag)
    os.makedirs(outdir, exist_ok=True)
    x0, y0, x1, y1 = REGIONS[a.tag]
    blocks = [(x, y) for y in range(int(np.floor(y0)), int(np.ceil(y1)))
              for x in range(int(np.floor(x0)), int(np.ceil(x1)))]
    blocks = [b for i, b in enumerate(blocks) if i % a.nparts == a.part]
    emit('%s part %d/%d：%d 个 1° 块' % (a.tag, a.part, a.nparts, len(blocks)))
    img = None
    t0 = time.time()
    for bi, (bx, by) in enumerate(blocks):
        bx1 = min(bx + 1, x1); by1 = min(by + 1, y1)
        fp = os.path.join(outdir, 'bc_p%02d_b%04d.parquet' % (a.part, bi))
        if os.path.isfile(fp):
            continue
        for attempt in range(4):
            try:
                if img is None:
                    if a.pid:
                        from prod_conf import apply_account_env
                        apply_account_env(a.acct)
                        ee.Initialize(project=a.pid)
                    else:
                        sys.path.insert(0, r'F:/lc_work/v31_exp/code')
                        import v31_common as VC
                        VC.ctx(a.acct)
                    img = build(ee)
                region = ee.Geometry.Rectangle([bx, by, bx1, by1])
                fc = img.sample(region=region, scale=30, numPixels=a.per_block,
                                seed=7 + bi, geometries=True, dropNulls=True, tileScale=4)
                feats = fc.getInfo()['features']
                rows = []
                for f in feats:
                    c = f['geometry']['coordinates']
                    p = dict(f['properties'])
                    p['lon'] = c[0]; p['lat'] = c[1]
                    rows.append(p)
                d = pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLS)
                for c in COLS:
                    if c not in d.columns:
                        d[c] = np.nan
                if len(d):
                    d['pid'] = ['bare_%.6f_%.6f' % (r.lon, r.lat) for r in d.itertuples()]
                d[COLS].to_parquet(fp, index=False)
                if (bi + 1) % 10 == 0:
                    emit('  %d/%d 块，累计点 %d（%.0fs）' % (
                        bi + 1, len(blocks), sum(len(pd.read_parquet(
                            os.path.join(outdir, f))) for f in os.listdir(outdir)), time.time() - t0))
                break
            except Exception as e:
                emit('  块 %d 尝试 %d 失败：%s' % (bi + 1, attempt + 1, str(e)[:120]))
                time.sleep(8 + 8 * attempt)
                img = None
        else:
            emit('  !! 块 %d 连续失败' % (bi + 1))
    emit('%s part %d 完成（%.0fs）' % (a.tag, a.part, time.time() - t0))


if __name__ == '__main__':
    main()
