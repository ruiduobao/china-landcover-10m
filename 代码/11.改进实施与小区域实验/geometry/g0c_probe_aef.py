# -*- coding: utf-8 -*-
"""g0c_probe_aef.py — AEF 路径下 clip 几何/掩膜解码判定（一次性探针，决定性）

背景：
  probe_geom（常量图，EPSG:4326 1° 投影）显示 `Rectangle([4 数字])` 与 `Rectangle(box,None,True)`
  逐行相同、且在南缘 lat<30.0038 全行置零（=测地拱），而 BBox/geodesic=False 无零；
  → 客户端 `Rectangle([数字])` 序列化为 `GeometryConstructors.Polygon`（数字坐标）→ 服务端
    Polygon 默认 geodesic=**true**，即生产 L220 确为测地（seams 报告归因成立）。
但 G30N_seed7（AEF classify 图）两臂均零 nodata → 需判定 AEF 原生投影路径下 clip 是否生效，
以及 computePixels 对 masked 值的解码。

做法：同一 AEF A00 图（原生投影）6 波段，全部 clip 到同一 box=[97,30,99,32]：
  b1 a00 无 clip | b2 clip(Rectangle[数字],测地) | b3 clip(BBox,平面)
  b4 clip(Rectangle,None,True) | b5 clip(Rectangle,None,False) | b6 = b2.unmask(0)（显式填 0 解码对照）
窗口 = T1206 南缘中心小窗。逐行统计每波段「==0」个数 + NaN/掩膜统计。
用法：python g0c_probe_aef.py
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
AEF_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'


def main():
    import v31_common as VC
    ee, pid = VC.ctx(ACCT)
    ee.data.setMaxRetries(0)
    g = dict(zip(['px', 'py', 'x0', 'ytop', 'W', 'H'], GP.grid_of_tif(REF_TIF)))
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], 10854, 22192, 557, 93)
    aef = (ee.ImageCollection(AEF_ID).filterDate('2023-01-01', '2024-01-01')
           .filterBounds(ee.Geometry.BBox(*BOX)).mosaic().select(['A00']))
    gN = ee.Geometry.Rectangle(BOX)
    gB = ee.Geometry.BBox(*BOX)
    gT = ee.Geometry.Rectangle(BOX, None, True)
    gF = ee.Geometry.Rectangle(BOX, None, False)
    arms = [('a00_noclip', aef),
            ('clip_rectNum', aef.clip(gN)),
            ('clip_bbox', aef.clip(gB)),
            ('clip_geoTrue', aef.clip(gT)),
            ('clip_geoFalse', aef.clip(gF)),
            ('clip_rectNum_unmask0', aef.clip(gN).unmask(0))]
    img = arms[0][1].rename(arms[0][0])
    for nm, im in arms[1:]:
        img = img.addBands(im.rename(nm))
    ee.data.setDeadline(60000)
    t0 = time.time()
    params = {'expression': img, 'fileFormat': 'GEO_TIFF',
              'grid': {'dimensions': {'width': int(w), 'height': int(h)},
                       'affineTransform': {'scaleX': tr[0], 'shearX': tr[1], 'translateX': tr[2],
                                           'shearY': tr[3], 'scaleY': tr[4], 'translateY': tr[5]},
                       'crsCode': 'EPSG:4326'}}
    b = ee.data.computePixels(params)
    dt = time.time() - t0
    p = os.path.join(OUTD, 'probe_aef.tif')
    open(p, 'wb').write(b)
    import rasterio
    with rasterio.open(p) as ds:
        a = ds.read()
        t = ds.transform
    lat = lambda r: t.f - (r + 0.5) * (-t.e)
    names = [nm for nm, _ in arms]
    out = dict(acct=ACCT, proj=pid, dt_s=round(dt, 2), shape=list(a.shape), names=names,
               wedge_h_deg=GP.wedge_height_deg(BOX), wedge_h_rows=GP.wedge_height_deg(BOX) / g['py'],
               band_zero_total={nm: int((a[i] == 0).sum()) for i, nm in enumerate(names)},
               band_finite_min={nm: float(np.nanmin(a[i])) for i, nm in enumerate(names)})
    out['per_row'] = [dict(row=r, lat=round(lat(r), 7),
                           **{nm: int((a[i][r] == 0).sum()) for i, nm in enumerate(names)})
                      for r in list(range(0, 8)) + list(range(36, 60)) + list(range(66, 93))]
    z = {nm: (a[i] == 0).sum(1) for i, nm in enumerate(names)}
    out['cmp'] = dict(
        rectNum_eq_bbox=bool((z['clip_rectNum'] == z['clip_bbox']).all()),
        rectNum_eq_geoTrue=bool((z['clip_rectNum'] == z['clip_geoTrue']).all()),
        rectNum_eq_geoFalse=bool((z['clip_rectNum'] == z['clip_geoFalse']).all()),
        unmask0_eq_clip=bool((z['clip_rectNum_unmask0'] == z['clip_rectNum']).all()))
    out['verdict'] = ('AEF 路径 clip 生效' if out['band_zero_total']['clip_rectNum'] > 0
                      else 'AEF 路径 clip 完全未生效（零置零）')
    json.dump(out, open(os.path.join(OUTD, 'probe_aef.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ['dt_s', 'wedge_h_deg', 'wedge_h_rows', 'band_zero_total',
                                          'band_finite_min', 'cmp', 'verdict']},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
