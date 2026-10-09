# -*- coding: utf-8 -*-
"""f5b_bio_national.py — 全国 1 km 气候（WorldClim bio01/bio12）本地栅格（生态规则全国化用）

* 输入：GEE WORLDC LIM/V1/BIO（bio01 年均温×10℃、bio12 年降水 mm）
* 规则源：与 e1/f5 同法——分纬带 getDownloadURL，2° 带（避免 48MB 上限）
* 输出：data/eco_gate/_dem/bio1000_chn.tif（2 波段）
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, rasterio, v31_common as VC
OUTD = os.path.join(WORK, 'data', 'eco_gate', '_dem')
BOX = [73.0, 18.0, 135.0, 54.0]
BANDS = [(y, y + 2) for y in range(18, 54, 2)]

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    fp = os.path.join(OUTD, 'bio1000_chn.tif')
    if os.path.exists(fp):
        emit('已存在，跳过'); return
    import ee
    VC.ensure_ctx('bx15mw')
    s, _ = VC.sess('bx15mw')
    bio = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'])
    parts = []
    for i, (y0, y1) in enumerate(BANDS):
        rect = ee.Geometry.Rectangle([BOX[0], y0, BOX[2], y1])
        p = os.path.join(OUTD, 'bio_b%d.tif' % i)
        for att in range(4):
            try:
                url = bio.getDownloadURL({'scale': 1000, 'crs': 'EPSG:4326', 'region': rect,
                                          'format': 'GEO_TIFF'})
                b = s.get(url, timeout=900).content
                if b[:2] in (b'II', b'MM') and len(b) > 1e5:
                    open(p, 'wb').write(b); parts.append(p)
                    emit('  纬带 %d-%d°N %.2f MB' % (y0, y1, len(b)/2**20)); break
                emit('  带 %d-%d 异常 %.50s' % (y0, y1, b[:50]))
            except Exception as e:
                emit('  带 %d-%d 第%d次失败 %s' % (y0, y1, att+1, str(e)[:70]))
            time.sleep(8)
    arrs, prof = [], None
    for p in parts:
        with rasterio.open(p) as ds:
            arrs.append(ds.read()); prof = ds.profile.copy()
    arr = np.concatenate(arrs, axis=1)
    prof.update(height=arr.shape[1], width=arr.shape[2],
                transform=rasterio.transform.from_origin(BOX[0], BOX[3], 0.01, 0.01),
                compress='deflate', tiled=True)
    with rasterio.open(fp, 'w', **prof) as dst: dst.write(arr)
    for p in parts: os.remove(p)
    emit('→ %s %s bio01[%d,%d] bio12[%d,%d]' % (fp, arr.shape, arr[0].min(), arr[0].max(), arr[1].min(), arr[1].max()))

if __name__ == '__main__': main()
