# -*- coding: utf-8 -*-
"""acl_read.py — 以被授权账号身份读共享资产，验证共享真的生效"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
ASSET = 'projects/braided-horizon-508210-a5/assets/lcf_a01/samples_5000_000'
C.apply_account_env('y30b63ye')
import ee
ee.Initialize(project=C.ACCOUNTS['y30b63ye']['proj'])
try:
    n = int(ee.FeatureCollection(ASSET).size().getInfo())
    print('✅ 被授权账号 y30b63ye 读到共享资产：%d 行' % n)
    # 真正用它训练一次，确认可用于 smileRandomForest
    reg = ee.Geometry.Rectangle([116.80, 36.50, 116.90, 36.60])
    img = (ee.ImageCollection(C.AEF).filterDate('2022-01-01', '2023-01-01')
           .filterBounds(reg).mosaic().select(C.FEATS))
    t0 = time.time()
    clf = ee.Classifier.smileRandomForest(numberOfTrees=100, minLeafPopulation=2).train(
        ee.FeatureCollection(ASSET), 'cl', C.FEATS)
    h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=60,
                                       maxPixels=10**9, tileScale=4, bestEffort=True).getInfo()
    print('✅ 跨账号共享资产直接训练+分类成功：%.0fs，区内 %d 类'
          % (time.time()-t0, len((h or {}).get('classification', {}) or {})))
except Exception as e:
    print('❌ 读取失败:', str(e)[:180])
