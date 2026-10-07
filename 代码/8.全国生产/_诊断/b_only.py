# -*- coding: utf-8 -*-
"""b_only.py — 只测 (b) GEE 端训练+分类耗时（样本资产已在上一步导出）"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
C.apply_account_env('zixen8v8')
import ee
ee.Initialize(project=C.ACCOUNTS['zixen8v8']['proj'])
ASSET = 'projects/braided-horizon-508210-a5/assets/lcf_a01/samples_5000_000'
fc = ee.FeatureCollection(ASSET)
print('资产行数:', int(fc.size().getInfo()), flush=True)
reg = ee.Geometry.Rectangle([116.80, 36.50, 117.00, 36.70])
img = (ee.ImageCollection(C.AEF).filterDate('2022-01-01', '2023-01-01')
       .filterBounds(reg).mosaic().select(C.FEATS))
for ntrees, mx in [(100, None), (150, None), (150, 200000)]:
    t0 = time.time()
    try:
        clf = ee.Classifier.smileRandomForest(numberOfTrees=ntrees, minLeafPopulation=2,
                                              maxNodes=mx).train(fc, 'cl', C.FEATS)
        h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg,
                                           scale=60, maxPixels=10**9, tileScale=4,
                                           bestEffort=True).getInfo()
        hh = (h or {}).get('classification', {}) or {}
        print('树=%s maxNodes=%s → %.0fs；区内 %d 类 %s'
              % (ntrees, mx, time.time()-t0, len(hh), sorted(int(k) for k in hh)), flush=True)
    except Exception as e:
        print('树=%s 失败 %.0fs: %s' % (ntrees, time.time()-t0, str(e)[:130]), flush=True)
print('=== 测资产 ACL 共享接口 ===', flush=True)
try:
    ee.data.setAssetAcl(ASSET, {'readers': ['ppzynq@example.com']})
    print('setAssetAcl 可调用')
except Exception as e:
    print('setAssetAcl:', str(e)[:130])
