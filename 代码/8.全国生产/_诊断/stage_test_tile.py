# -*- coding: utf-8 -*-
"""stage_test_tile.py — 用同一套（asset-训练）模型交互式产出一个测试瓦片，喂给 p6_mosaic 验证拼接链路"""
import os, sys, io, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('zixen8v8')
import ee
ee.Initialize(project=C.ACCOUNTS['zixen8v8']['proj'])
ASSET = 'projects/braided-horizon-508210-a5/assets/lcf_a01/samples_5000_000'
fc = ee.FeatureCollection(ASSET)
kw = dict(numberOfTrees=150, minLeafPopulation=2, maxNodes=200000, seed=42)
clf = ee.Classifier.smileRandomForest(**kw).train(fc, 'cl', C.FEATS)
x0, y0 = 116.5, 36.0
box = ee.Geometry.Rectangle([x0, y0, x0 + 0.3, y0 + 0.3])
img = (ee.ImageCollection(C.AEF).filterDate('2022-01-01', '2023-01-01')
       .filterBounds(box).mosaic().select(C.FEATS))
cls = img.classify(clf).rename('class').uint8()
t0 = time.time()
url = cls.getDownloadURL(dict(region=box, scale=10, crs='EPSG:4326', format='GEO_TIFF'))
r = requests.get(url, proxies=C.PROXY, timeout=1800); r.raise_for_status()
outdir = os.path.join(C.STAGE, '2022'); os.makedirs(outdir, exist_ok=True)
fp = os.path.join(outdir, 'T999_2022.tif')
open(fp, 'wb').write(r.content)
print('测试瓦片 %.1f MB / %.0fs → %s' % (len(r.content)/1e6, time.time()-t0, fp))
