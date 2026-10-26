# -*- coding: utf-8 -*-
"""g0_probe_mask.py — 诊断：computePixels 是否保留 clip/updateMask 掩膜（一次性探针）

目的：G30N_seed7 单请求返回的 legacy/planar 在 box 外仍有值 → 需判定是
  (a) computePixels 丢掩膜，还是 (b) clip 几何在 AEF 原生投影下未生效。
做法：3 波段常量图（默认 EPSG:4326 投影）：
  b1 = constant(1)（无掩膜，恒 1）
  b2 = constant(1).clip(planar 小框)（小框内 1、框外应 0）
  b3 = constant(1).updateMask(constant(0))（全掩 → 应全 0）
导出窗口 = 小框 ± 外扩，逐行统计 0 的个数。
用法：python g0_probe_mask.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import numpy as np
import geom_planar as GP

ACCT = 'k3r6ncvd'
OUTD = os.path.join(HERE, 'out')
REF_TIF = r'F:/lc_work/prod2nd_2023/rasters_batch/T1206_10m.tif'
BOX = [97.0, 30.0, 99.0, 32.0]


def main():
    import v31_common as VC
    ee, pid = VC.ctx(ACCT)
    ee.data.setMaxRetries(0)
    g = dict(zip(['px', 'py', 'x0', 'ytop', 'W', 'H'], GP.grid_of_tif(REF_TIF)))
    # 窗口：cols[10854,11411) rows[22192,22285)
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], 10854, 22192, 557, 93)
    sub = [97.98, 30.001, 98.02, 30.005]          # 小框：完全落在窗口内
    b1 = ee.Image.constant(1).byte().rename('c1')
    b2 = ee.Image.constant(1).byte().rename('clip_planar').clip(GP.planar_rect(sub, ee=ee))
    b3 = ee.Image.constant(1).byte().rename('updmask0').updateMask(ee.Image.constant(0))
    b4 = ee.Image.constant(1).byte().rename('clip_legacy').clip(GP.legacy_rect(sub, ee=ee))
    img = b1.addBands(b2).addBands(b3).addBands(b4)
    ee.data.setDeadline(60000)
    t0 = time.time()
    params = {'expression': img, 'fileFormat': 'GEO_TIFF',
              'grid': {'dimensions': {'width': int(w), 'height': int(h)},
                       'affineTransform': {'scaleX': tr[0], 'shearX': tr[1], 'translateX': tr[2],
                                           'shearY': tr[3], 'scaleY': tr[4], 'translateY': tr[5]},
                       'crsCode': 'EPSG:4326'}}
    b = ee.data.computePixels(params)
    dt = time.time() - t0
    p = os.path.join(OUTD, 'probe_mask.tif')
    open(p, 'wb').write(b)
    import rasterio
    with rasterio.open(p) as ds:
        a = ds.read()
        t = ds.transform
        names = ds.descriptions
    out = dict(acct=ACCT, proj=pid, dt_s=round(dt, 2), shape=list(a.shape), names=list(names),
               band_uniq=[sorted(set(a[i].ravel().tolist()))[:6] for i in range(a.shape[0])],
               band_zero_count=[int((a[i] == 0).sum()) for i in range(a.shape[0])])
    # 逐行：clip_planar(b2) 的 0 计数，验证掩膜是否随行变化（小框外应全 0）
    lat = lambda r: t.f - (r + 0.5) * (-t.e)
    rows = []
    for r in range(a.shape[1]):
        rows.append(dict(row=r, lat=round(lat(r), 7),
                         c1_0=int((a[0, r] == 0).sum()), clip_planar_0=int((a[1, r] == 0).sum()),
                         updmask0_0=int((a[2, r] == 0).sum()), clip_legacy_0=int((a[3, r] == 0).sum())))
    out['rows_sample'] = [rows[0], rows[20], rows[45], rows[70], rows[92]]
    out['verdict'] = dict(
        constant_ok=out['band_zero_count'][0] == 0,
        clip_planar_masks=(out['band_zero_count'][1] > 0),
        updmask_masks=(out['band_zero_count'][2] == a.shape[1] * a.shape[2]),
        clip_legacy_masks=(out['band_zero_count'][3] > 0),
        computePixels_keeps_mask=bool(out['band_zero_count'][2] == a.shape[1] * a.shape[2]),
    )
    json.dump(out, open(os.path.join(OUTD, 'probe_mask.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
