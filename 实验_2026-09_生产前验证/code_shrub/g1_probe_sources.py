# -*- coding: utf-8 -*-
"""g1_probe_sources.py — 探源：四省可用的"灌丛"专题数据源盘点（GEE 侧 + OSM）

* 输入：GEE 各灌丛专题数据集（见 SOURCES）；四省 bbox
* 规则源：无（仅探测可用性与覆盖规模）
* 门槛：每个源报"该省 bbox 内灌丛类像元数/要素数"与数据年份
* 输出：data/shrub/probe_sources.json + 打印表
* 用法：python g1_probe_sources.py [--acct bx15mw]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import v31_common as VC
OUTD = os.path.join(WORK, 'data', 'shrub')
PROV = {'宁夏': (104.0, 35.0, 108.0, 40.0), '四川': (97.0, 26.0, 109.0, 34.5),
        '黑龙江': (121.0, 43.0, 135.5, 53.5), '福建': (115.5, 23.0, 120.5, 28.5)}

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acct', default='bx15mw'); a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    import ee
    VC.ensure_ctx(a.acct)
    res = {}
    # 1) Dynamic World v1（10 m，含 shrub_and_scrub 概率；2015-06 起）
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    # 2) ESA WorldCover v200（10 m，20=Shrubland）
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    # 3) Copernicus LC100（100 m，20=Shrubland）
    try:
        lc100 = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification')
    except Exception:
        lc100 = None
    # 4) MODIS MCD12Q1（500 m，IGBP 6/7=shrub）
    mcd = ee.ImageCollection('MODIS/061/MCD12Q1').first().select('LC_Type1')
    for p, box in PROV.items():
        rect = ee.Geometry.Rectangle(list(box))
        r = {}
        try:
            dwc = dw.filterDate('2023-01-01', '2023-12-31').filterBounds(rect).select('label').mode()
            n = (dwc.eq(6).rename('s').reduceRegion(ee.Reducer.sum(), rect, scale=100, maxPixels=1e10,
                                                   bestEffort=True, tileScale=4).getInfo() or {}).get('s')
            r['dynamicworld_shrub_px@100m'] = n
        except Exception as e:
            r['dynamicworld'] = 'err ' + str(e)[:60]
        try:
            n = (wc.eq(20).rename('s').reduceRegion(ee.Reducer.sum(), rect, scale=100, maxPixels=1e10,
                                                   bestEffort=True, tileScale=4).getInfo() or {}).get('s')
            r['worldcover20_px@100m'] = n
        except Exception as e:
            r['worldcover'] = 'err ' + str(e)[:60]
        if lc100 is not None:
            try:
                n = (lc100.eq(20).rename('s').reduceRegion(ee.Reducer.sum(), rect, scale=200, maxPixels=1e10,
                                                          bestEffort=True, tileScale=4).getInfo() or {}).get('s')
                r['lc100_20_px@200m'] = n
            except Exception as e:
                r['lc100'] = 'err ' + str(e)[:60]
        try:
            v = mcd.eq(6).Or(mcd.eq(7)).rename('s')
            n = (v.reduceRegion(ee.Reducer.sum(), rect, scale=500, maxPixels=1e10, bestEffort=True,
                                tileScale=4).getInfo() or {}).get('s')
            r['modis_shrub_px@500m'] = n
        except Exception as e:
            r['modis'] = 'err ' + str(e)[:60]
        res[p] = r
        emit('%s：%s' % (p, r))
    VC.jsave(res, os.path.join(OUTD, 'probe_sources.json'))

if __name__ == '__main__': main()
