# -*- coding: utf-8 -*-
"""cap_probe.py — 实测 GEE 能吃下的最大 smileRandomForest 规格（决定生产模型参数）"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('ief3nj')
import ee
ee.Initialize(project=C.ACCOUNTS['ief3nj']['proj'])
ROOT = 'projects/temporal-web-508410-h0/assets/lcf_c02'
ids = ['%s/smp_v1_%03d' % (ROOT, i) for i in range(29)]
fc = ee.FeatureCollection(ids[0])
for _a in ids[1:]:
    fc = fc.merge(ee.FeatureCollection(_a))
print('样本行数:', int(fc.size().getInfo()), flush=True)
reg = ee.Geometry.Rectangle([120.0, 40.0, 120.15, 40.15])   # 辽宁 0.15° 抽查窗
img = (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
       .filterBounds(reg).mosaic().select(C.FEATS))
for trees, mx, leaf in [(150, 200000, 2), (150, 20000, 2), (150, 5000, 2),
                        (100, 2000, 5), (50, 1000, 10)]:
    t0 = time.time()
    try:
        clf = ee.Classifier.smileRandomForest(numberOfTrees=trees, minLeafPopulation=leaf,
                                              maxNodes=mx).train(fc, 'cl', C.FEATS)
        h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=60,
                                           maxPixels=10**9, tileScale=4,
                                           bestEffort=True).getInfo()
        hh = (h or {}).get('classification', {}) or {}
        print('✅ 树%d/节点%s/叶%d → %.0fs，%d 类' % (trees, mx, leaf, time.time()-t0, len(hh)), flush=True)
    except Exception as e:
        print('❌ 树%d/节点%s/叶%d → %.0fs  %s' % (trees, mx, leaf, time.time()-t0, str(e)[:90]), flush=True)
