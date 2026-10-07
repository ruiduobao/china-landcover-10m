# -*- coding: utf-8 -*-
"""与 ESA WorldCover v200 的逐窗对照（6 个典型地貌窗口）"""
import sys, os
sys.path.insert(0, '.'); sys.path.insert(0, '../0.本地流水线')
import numpy as np, rasterio
from lc_conf import CLASSES
import prod_conf as C

a = np.load(r'F:/lc_work/ne100m.npy')
with rasterio.open(r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/生产输出/成品/东北三省_2023_100m.tif') as s:
    tr = s.transform; H, W = s.height, s.width

OURS = {'乔木林':[51,52,61,62,71,72,81,82,91,92], '灌丛':[120,121], '草地':[130],
        '耕地':[10,11,12], '建成区':[190,200], '裸地/稀疏':[140,150,201],
        '水体':[202], '湿地':[180,181,182,183,184,185,186], '冰雪':[220]}
WC = {10:'乔木林', 20:'灌丛', 30:'草地', 40:'耕地', 50:'建成区', 60:'裸地/稀疏',
      80:'水体', 90:'湿地', 70:'冰雪', 95:'湿地', 100:'裸地/稀疏'}

C.apply_account_env('zixen8v8')
import ee, requests, json
ee.Initialize(project='braided-horizon-508210-a5')
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')

WINS = [('大兴安岭北·针叶林', 122.5, 51.5, 123.5, 52.5),
        ('小兴安岭·针阔混交', 128.0, 47.5, 129.0, 48.5),
        ('三江平原·湿地水田', 132.0, 46.5, 133.0, 47.5),
        ('长白山·阔叶混交', 127.5, 42.0, 128.5, 43.0),
        ('松嫩平原·旱作农区', 124.5, 45.5, 125.5, 46.5),
        ('辽河平原·农耕城市', 122.5, 41.0, 123.5, 42.0)]

print('%-22s %-8s %s' % ('窗口', '来源', '各地类占比（%）'))
print('-' * 100)
for name, x0, y0, x1, y1 in WINS:
    r0 = int(round((tr.f - y1) / abs(tr.e))); r1 = int(round((tr.f - y0) / abs(tr.e)))
    c0 = int(round((x0 - tr.c) / tr.a)); c1 = int(round((x1 - tr.c) / tr.a))
    r0, r1 = max(0, r0), min(H, r1); c0, c1 = max(0, c0), min(W, c1)
    blk = a[r0:r1, c0:c1]
    tot_o = int((blk > 0).sum())
    ours = {}
    if tot_o:
        for g, cs in OURS.items():
            n = int(np.isin(blk, cs).sum())
            if n: ours[g] = 100.0 * n / tot_o
    reg = ee.Geometry.Rectangle([x0, y0, x1, y1])
    h = wc.reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=100,
                        maxPixels=10**9, bestEffort=True).getInfo()
    hh = (h or {}).get('Map', {}) or {}
    tot_w = sum(int(v) for v in hh.values())
    wcv = {}
    for k, v in hh.items():
        g = WC.get(int(k))
        if g and tot_w:
            wcv[g] = wcv.get(g, 0) + 100.0 * int(v) / tot_w
    keys = sorted(set(ours) | set(wcv), key=lambda k: -max(ours.get(k, 0), wcv.get(k, 0)))
    s1 = '  '.join('%s %.0f' % (k, ours.get(k, 0)) for k in keys[:6])
    s2 = '  '.join('%s %.0f' % (k, wcv.get(k, 0)) for k in keys[:6])
    print('%-22s %-8s %s' % (name, '本项目', s1))
    print('%-22s %-8s %s' % ('', 'WorldCover', s2))
    print()
