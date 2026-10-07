# -*- coding: utf-8 -*-
"""isolate.py — 隔离测试：分类器大小 vs 区域大小，谁触发 Computed value is too large"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('ief3nj')
import ee
ee.Initialize(project=C.ACCOUNTS['ief3nj']['proj'])
MERGED = 'projects/temporal-web-508410-h0/assets/lcf_c02/smp_merged'

def trial(tag, fc_train, deg, trees, mx):
    t0 = time.time()
    try:
        clf = ee.Classifier.smileRandomForest(numberOfTrees=trees, minLeafPopulation=2,
                                              maxNodes=mx).train(fc_train, 'cl', C.FEATS)
        reg = ee.Geometry.Rectangle([120.0, 40.0, 120.0 + deg, 40.0 + deg])
        img = (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
               .filterBounds(reg).mosaic().select(C.FEATS))
        h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=200,
                                           maxPixels=10**9, tileScale=4,
                                           bestEffort=True).getInfo()
        hh = (h or {}).get('classification', {}) or {}
        print('✅ %-28s %.0fs  %d 类' % (tag, time.time()-t0, len(hh)), flush=True)
        return True
    except Exception as e:
        print('❌ %-28s %.0fs  %s' % (tag, time.time()-t0, str(e)[:80]), flush=True)
        return False

fc = ee.FeatureCollection(MERGED)
for frac, label in [(0.02, '训练2万点'), (0.10, '训练10万点'), (1.0, '训练20万点')]:
    sub = fc.randomColumn('r', 7).filter('r < %s' % frac) if frac < 1 else fc
    for deg in [0.05, 0.5, 1.5]:
        ok = trial('%s / %.2f°区 / 100树x5000' % (label, deg), sub, deg, 100, 5000)
        if not ok and deg >= 0.5:
            break
