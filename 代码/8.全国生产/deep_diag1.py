# -*- coding: utf-8 -*-
"""深度诊断 1：把 ESA WorldCover 拉成 500m 覆盖全 NE，与我们 100m 成品做**全域逐类交叉表**"""
import sys, os
sys.path.insert(0, '.'); sys.path.insert(0, '../0.本地流水线')
import numpy as np, rasterio, requests, json
import prod_conf as C
WC_FP = r'F:/lc_work/wc_ne_500m.tif'
BOXP = (118.5, 38.5, 135.0, 54.0)
if not os.path.isfile(WC_FP):
    import ee
    C.apply_account_env('zixen8v8')
    ee.Initialize(project='braided-horizon-508210-a5')
    box = ee.Geometry.Rectangle(list(BOXP))
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    url = wc.getDownloadURL(dict(region=box, scale=500, crs='EPSG:4326', format='GEO_TIFF'))
    r = requests.get(url, proxies=C.PROXY, timeout=1800); r.raise_for_status()
    open(WC_FP, 'wb').write(r.content)
    print('WorldCover 500m 下载 %.1f MB' % (len(r.content)/1e6))
with rasterio.open(WC_FP) as s:
    w = s.read(1); wtr = s.transform
    print('WC 栅格 %d×%d 像元 %.4f°' % (s.width, s.height, wtr.a))
print('WC 类别码:', sorted(int(x) for x in np.unique(w) if x))
np.save(r'F:/lc_work/wc_ne.npy', w)
json.dump({'transform': [wtr.c, wtr.a, wtr.f, wtr.e], 'shape': list(w.shape)},
          open(r'F:/lc_work/wc_ne_meta.json', 'w'))
