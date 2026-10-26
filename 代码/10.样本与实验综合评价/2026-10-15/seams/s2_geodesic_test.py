# -*- coding: utf-8 -*-
"""s2_geodesic_test — 测地矩形（ee.Geometry.Rectangle 默认 geodesic=True）南边大圆拱 vs 瓦片实际 nodata 掩膜

* 输入：rasters_batch/{tile}_10m.tif + plan/tiles_secondrun.csv 的 box
* 规则源：ee.Geometry.Rectangle 数字构造 → geodesic 默认 True（EE docs 本地库已核）→
    南/北边为球面大圆；经度方向两个端点同纬，故大圆顶点（最高纬度）在中经线上：
        tan(phi_arc(lon)) = tan(phi0) * cos(lon - lon_m) / cos(dlon/2)
    东/西边两点同经度 → 大圆即该经线（无拱）
* 门槛：只读；预测掩膜 = 像元中心不在 region 内（lat < phi_arc_south 或 lat > phi_arc_north）
* 输出：CSV/JSON（每瓦 Jaccard、逐列深度误差）
* 用法：python s2_geodesic_test.py
"""
import os
import sys
import csv
import json
import math

import numpy as np
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
TILES = r'F:/lc_work/prod2nd_2023/rasters_batch'
PLAN = r'F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv'
PXDEG = 8.983152841195215e-05


def load_boxes():
    out = {}
    with open(PLAN, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['tile']] = dict(x0=float(r['x0']), y0=float(r['y0']),
                                  x1=float(r['x1']), y1=float(r['y1']),
                                  prov=r['prov'], land_frac=float(r['land_frac']))
    return out


def arc_lat(lat0_deg, lon, lon_a, lon_b):
    """同纬两端点 (lat0,lon_a)-(lat0,lon_b) 的球面大圆在 lon 处的纬度（度）。"""
    t0 = math.tan(math.radians(lat0_deg))
    dlon2 = math.radians(abs(lon_b - lon_a)) / 2.0
    lonm = (lon_a + lon_b) / 2.0
    return math.degrees(math.atan(t0 * math.cos(math.radians(lon - lonm)) / math.cos(dlon2)))


def predict_nodata(box, t, W, H, rows_from_south=200):
    """返回 (rows_win, cols) 预测掩膜：像元中心在 region 外=True（只看南边带）。"""
    r0 = H - rows_from_south
    rr, cc = np.mgrid[r0:H, 0:W]
    lon = t.c + (cc + 0.5) * t.a
    lat = t.f - (rr + 0.5) * abs(t.e)
    inside = np.ones(lat.shape, dtype=bool)
    inside &= (lon >= box['x0']) & (lon <= box['x1'])
    phi_s = np.array([[arc_lat(box['y0'], float(x), box['x0'], box['x1'])] for x in lon[0, :]]).ravel()
    phi_n = np.array([[arc_lat(box['y1'], float(x), box['x0'], box['x1'])] for x in lon[0, :]]).ravel()
    inside &= lat >= phi_s[None, :]
    inside &= lat <= phi_n[None, :]
    return r0, ~inside, lon, lat


def main():
    boxes = load_boxes()
    res = []
    for tile in ['T1206', 'T1306', 'T1205', 'T1610', 'T1802', 'T2102', 'T1304']:
        fp = os.path.join(TILES, '%s_10m.tif' % tile)
        if not os.path.exists(fp):
            continue
        b = boxes[tile]
        with rasterio.open(fp) as ds:
            W, H, t = ds.width, ds.height, ds.transform
            n = 260
            r0 = H - n
            act = ds.read(1, window=Window(0, r0, W, n))
        _, pred, lon, lat = predict_nodata(b, t, W, H, rows_from_south=n)
        act_nod = (act == 0)
        # 只在南边带内比较（预测掩膜在设计上只影响南边）
        inter = int((act_nod & pred).sum())
        union = int((act_nod | pred).sum())
        jac = inter / union if union else 1.0
        # 逐列实测“从南边数第一条有值行”与预测深度
        def depth_from_south(mask2d):
            # 每列：从底部往上连续 True 的个数（向量化）
            rev = mask2d[::-1, :]
            d = (~rev).argmax(axis=0).astype(np.int32)
            d[mask2d.all(axis=0)] = mask2d.shape[0]
            return d
        da = depth_from_south(act_nod)
        dp = depth_from_south(pred)
        dcol = da - dp
        rec = dict(tile=tile, prov=b['prov'], box=[b['x0'], b['y0'], b['x1'], b['y1']],
                   W=W, H=H, n_rows=n,
                   actual_nodata_px=int(act_nod.sum()), pred_nodata_px=int(pred.sum()),
                   jaccard=round(jac, 6),
                   depth_err_px=dict(mean=round(float(dcol.mean()), 3),
                                     max_abs=int(np.abs(dcol).max()),
                                     p95_abs=round(float(np.percentile(np.abs(dcol), 95)), 2),
                                     exact_frac=round(float((dcol == 0).mean()), 4)),
                   wedge_rows_measured=int((act_nod.sum(axis=1) > 0).sum()),
                   wedge_h_pred=round(arc_lat(b['y0'], (b['x0'] + b['x1']) / 2, b['x0'], b['x1']) - b['y0'], 8),
                   wedge_h_pred_rows=round((arc_lat(b['y0'], (b['x0'] + b['x1']) / 2, b['x0'], b['x1']) - b['y0']) / PXDEG, 2),
                   )
        res.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    with open(os.path.join(HERE, 'geodesic_test.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('\nWROTE geodesic_test.json')
    # 汇总表
    print('\n%-7s %-4s %-22s %8s %8s %8s %8s' % ('tile', 'prov', 'box', 'wedge行', '预测行', 'Jaccard', '深度maxerr'))
    for r in res:
        print('%-7s %-4s %-22s %8d %8.2f %8.4f %8d' % (
            r['tile'], r['prov'], str(r['box']), r['wedge_rows_measured'], r['wedge_h_pred_rows'],
            r['jaccard'], r['depth_err_px']['max_abs']))


if __name__ == '__main__':
    main()
