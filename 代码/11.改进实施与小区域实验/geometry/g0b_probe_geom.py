# -*- coding: utf-8 -*-
"""g0b_probe_geom.py — 判定「哪个 clip 几何真的带测地大圆拱」（一次性探针）

背景：g0_probe_mask 显示 `Rectangle([4 数字])` 与 `BBox` 的 clip 掩膜**逐行完全相同**；
客户端序列化核对（ee 1.7.42）显示：
    Rectangle([4 数字])            → GeometryConstructors.Polygon，**无 geodesic 键** → 服务端 Polygon 默认（平面）
    BBox(...)                      → GeometryConstructors.Polygon, geodesic=false → 平面
    Rectangle(box, None, False)    → GeometryConstructors.Rectangle, geodesic=false → 平面
    Rectangle(box, None, True)     → GeometryConstructors.Rectangle, geodesic=true  → **测地（真 legacy）**
→ 故「生产 L220 用 Rectangle(meta['box']) 即测地」的代码级归因需要修正；本探针用常量图直接实测。

做法：4 波段 constant(1) 分别 clip 上述 4 种几何（**同一 box=[97,30,99,32]**），
导出 T1206 南缘中心小窗，逐行统计 0 的个数：测地臂应在南边中部多出「拱形」零带。
用法：python g0b_probe_geom.py
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
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], 10854, 22192, 557, 93)
    one = ee.Image.constant(1).byte()
    arms = [
        ('rect_num', one.clip(ee.Geometry.Rectangle(BOX))),
        ('bbox', one.clip(ee.Geometry.BBox(*BOX))),
        ('rect_geoTrue', one.clip(ee.Geometry.Rectangle(BOX, None, True))),
        ('rect_geoFalse', one.clip(ee.Geometry.Rectangle(BOX, None, False))),
    ]
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
    p = os.path.join(OUTD, 'probe_geom.tif')
    open(p, 'wb').write(b)
    import rasterio
    with rasterio.open(p) as ds:
        a = ds.read()
        t = ds.transform
    lat = lambda r: t.f - (r + 0.5) * (-t.e)
    names = [nm for nm, _ in arms]
    z = {nm: (a[i] == 0).sum(1) for i, nm in enumerate(names)}
    # 每行「相对 rect_num 多出的零」→ 测地拱签名
    arc_rows = [r for r in range(a.shape[1]) if z['rect_geoTrue'][r] > z['rect_num'][r]]
    out = dict(acct=ACCT, proj=pid, dt_s=round(dt, 2), shape=list(a.shape), names=names,
               wedge_h_deg=GP.wedge_height_deg(BOX), wedge_h_rows=GP.wedge_height_deg(BOX) / g['py'],
               arc_extra_zero_rows=len(arc_rows), arc_row_range=[min(arc_rows), max(arc_rows)] if arc_rows else None,
               bands_identical=dict(
                   rect_num_eq_bbox=bool((z['rect_num'] == z['bbox']).all()),
                   rect_num_eq_geoFalse=bool((z['rect_num'] == z['rect_geoFalse']).all()),
                   rect_num_eq_geoTrue=bool((z['rect_num'] == z['rect_geoTrue']).all())),
               verdict='')
    if arc_rows:
        rr = np.array(arc_rows)
        extra = np.array([z['rect_geoTrue'][r] - z['rect_num'][r] for r in arc_rows])
        out['arc_max_extra_zero'] = int(extra.max())
        out['arc_max_extra_at_lat'] = round(lat(int(rr[int(np.argmax(extra))])), 7)
        out['verdict'] = ('geodesic=True 是唯一带拱几何；Rectangle([数字]) 与 BBox/geodesic=False 均为平面'
                          if (out['bands_identical']['rect_num_eq_bbox'] and
                              not out['bands_identical']['rect_num_eq_geoTrue'])
                          else '混合：见 bands_identical')
    else:
        out['verdict'] = '四种几何的 clip 掩膜逐行无差异（本窗内均未出现拱）→ 楔形非来自 clip 几何'
    # 逐行表（南缘 30 行 + 拱区）
    rows = []
    for r in sorted(set(list(range(0, 12)) + list(range(40, a.shape[1])))):
        rows.append(dict(row=r, lat=round(lat(r), 7), **{nm: int(z[nm][r]) for nm in names}))
    out['rows'] = rows
    json.dump(out, open(os.path.join(OUTD, 'probe_geom.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ['dt_s', 'wedge_h_deg', 'wedge_h_rows', 'arc_extra_zero_rows',
                                          'arc_row_range', 'bands_identical', 'arc_max_extra_zero',
                                          'verdict']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
