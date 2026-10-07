# -*- coding: utf-8 -*-
"""p5_call_test.py — 单独验证 p5 的下载调用（与 p5_fetch 内部完全同一 API/参数）
p5_fetch 对每个子块调用的就是 getDownloadURL(region=subbox, scale=10, crs=EPSG:4326, format=GEO_TIFF)；
这里对同一调用做 0.25° 子块（p5 默认 SUBTILE_DEG）的实测，确认体积与耗时在预算内。"""
import os, sys, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('zixen8v8')
import ee
ee.Initialize(project=C.ACCOUNTS['zixen8v8']['proj'])
clf = ee.Classifier.smileRandomForest(numberOfTrees=150, minLeafPopulation=2,
                                      maxNodes=200000, seed=42).train(
    ee.FeatureCollection('projects/braided-horizon-508210-a5/assets/lcf_a01/samples_5000_000'),
    'cl', C.FEATS)
x0, y0 = 116.5, 36.0
for deg in [0.25, 0.35, 0.45]:
    box = ee.Geometry.Rectangle([x0, y0, x0 + deg, y0 + deg])
    img = (ee.ImageCollection(C.AEF).filterDate('2022-01-01', '2023-01-01')
           .filterBounds(box).mosaic().select(C.FEATS))
    cls = img.classify(clf).rename('class').uint8()
    t0 = time.time()
    try:
        url = cls.getDownloadURL(dict(region=box, scale=10, crs='EPSG:4326', format='GEO_TIFF'))
        r = requests.get(url, proxies=C.PROXY, timeout=1800); r.raise_for_status()
        out = os.path.join(C.STAGE, '2022', 'probe_%.2f.tif' % deg)
        open(out, 'wb').write(r.content)
        print('✅ %.2f° @10m：%.2f MB，%.0f s（p5 默认子块 0.25°）' % (deg, len(r.content)/1e6, time.time()-t0))
    except Exception as e:
        print('❌ %.2f°：%.0f s  %s' % (deg, time.time()-t0, str(e)[:110]))
