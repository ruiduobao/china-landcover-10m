# -*- coding: utf-8 -*-
"""size_probe.py — 找 GEE 分类器训练的最大样本量（50k/100k/150k）"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('ief3nj')
import ee
ee.Initialize(project=C.ACCOUNTS['ief3nj']['proj'])
fc = ee.FeatureCollection('projects/temporal-web-508410-h0/assets/lcf_c02/smp_merged')
print('合并样本总数:', int(fc.size().getInfo()), flush=True)
reg = ee.Geometry.Rectangle([120.0, 40.0, 121.0, 41.0])     # 1° 区（比抽查窗大）
img = (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
       .filterBounds(reg).mosaic().select(C.FEATS))
for frac, n in [(0.25, '5万'), (0.5, '10万'), (0.75, '15万')]:
    sub = fc.randomColumn('r', 7).filter('r < %s' % frac)
    t0 = time.time()
    try:
        clf = ee.Classifier.smileRandomForest(numberOfTrees=100, minLeafPopulation=2,
                                              maxNodes=5000).train(sub, 'cl', C.FEATS)
        h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=100,
                                           maxPixels=10**9, tileScale=4,
                                           bestEffort=True).getInfo()
        print('✅ ~%s点(frac=%s) 1°区 → %.0fs，%d 类' % (n, frac, time.time()-t0,
              len((h or {}).get('classification', {}) or {})), flush=True)
    except Exception as e:
        print('❌ ~%s点(frac=%s) → %.0fs  %s' % (n, frac, time.time()-t0, str(e)[:80]), flush=True)
