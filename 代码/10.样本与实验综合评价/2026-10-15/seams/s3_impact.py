# -*- coding: utf-8 -*-
"""s3_impact — 楔形（测地南边拱）在三个**省件**内的丢失像元/面积（省界内共径，与交付同口径）

* 输入：mosaic2/{四川,广东,宁夏}_2023_10m.tif（交付省件，nodata=0）+ plan/tiles_secondrun.csv box
* 规则源：楔形 = 每瓦南边测地大圆拱与瓦片南网格线之间的透镜区（s2_geodesic_test 已证）；
    省界 = p5common.prov_geom（DataV GCJ-02 raw）+ buffer(-0.005°)（与 c5 clip_prov/交付同口径），
    像元中心在缓冲后多边形内 → 省内
* 门槛：口径三档同时给，避免夸大
    - bbox 上界：省件 bbox 内 楔∩nodata（含省外 nodata，只是上界）
    - 省内（主口径）：楔∩nodata∩(省界内)  ← 可称"省内损失"
    - 省内期望：楔∩(省界内)（该区本应全部有值；AEF 有数据由样本池 521 点支持）
* 输出：impact_prov.json / impact_prov.csv（三档分列）
* 用法：python s3_impact.py
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
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
HERE = os.path.dirname(os.path.abspath(__file__))
MOS = r'F:/lc_work/prod2nd_2023/mosaic2'
PXDEG = 8.983152841195215e-05
M_DEG = 111320.0
BUF = -0.005


def arc_lat(lat0, lon, lon_a, lon_b):
    t0 = math.tan(math.radians(lat0))
    d2 = math.radians(abs(lon_b - lon_a)) / 2.0
    lm = (lon_a + lon_b) / 2.0
    return math.degrees(math.atan(t0 * math.cos(math.radians(lon - lm)) / math.cos(d2)))


def main():
    import p5common as PC
    from shapely import contains_xy, prepare
    from shapely.ops import unary_union
    boxes = {}
    for r in csv.DictReader(open(r'F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv', encoding='utf-8')):
        boxes[r['tile']] = dict(x0=float(r['x0']), y0=float(r['y0']),
                                x1=float(r['x1']), y1=float(r['y1']), prov=r['prov'])
    provbands = {}
    for t, b in boxes.items():
        provbands.setdefault(b['prov'], set()).add(b['y0'])
    out = []
    for prov, fname in [('川', '四川_2023_10m.tif'), ('粤', '广东_2023_10m.tif'), ('宁', '宁夏_2023_10m.tif')]:
        fp = os.path.join(MOS, fname)
        geom = PC.prov_geom(PC.PROV_ALIAS.get(prov, prov))
        geom = geom.buffer(BUF)
        geom = unary_union([g for g in (geom.geoms if hasattr(geom, 'geoms') else [geom])
                            if g.geom_type in ('Polygon', 'MultiPolygon')])
        prepare(geom)
        gminx, gminy, gmaxx, gmaxy = geom.bounds
        with rasterio.open(fp) as ds:
            H, W, t = ds.height, ds.width, ds.transform
            bl = ds.bounds.left
            tot = dict(bbox_px=0, bbox_m2=0.0, in_px=0, in_m2=0.0, exp_px=0, exp_m2=0.0)
            bands = []
            for y0 in sorted(provbands.get(prov, [])):
                lo = max(0, int(math.floor((t.f - (y0 + 0.006)) / PXDEG)))
                hi = min(H, int(math.ceil((t.f - y0) / PXDEG)) + 2)
                if hi <= lo:
                    continue
                blk = ds.read(1, window=Window(0, lo, W, hi - lo))
                rr, cc = np.mgrid[lo:hi, 0:W]
                lon = t.c + (cc + 0.5) * t.a
                lat = t.f - (rr + 0.5) * PXDEG
                lens = np.zeros(blk.shape, dtype=bool)
                for tb, b in boxes.items():
                    if b['prov'] != prov or abs(b['y0'] - y0) > 1e-9:
                        continue
                    inb = (lon >= b['x0']) & (lon <= b['x1'])
                    arc = np.array([arc_lat(b['y0'], float(x), b['x0'], b['x1']) for x in lon[0, :]])
                    lens |= (lat < arc[None, :]) & (lat >= b['y0'] - 1e-9) & inb
                # 省界内掩膜（仅在 bbox 交集列内计算，省时间）
                csel = (lon[0, :] >= gminx - 0.01) & (lon[0, :] <= gmaxx + 0.01)
                rsel = (lat[:, 0] >= gminy - 0.01) & (lat[:, 0] <= gmaxy + 0.01)
                inside = np.zeros(blk.shape, dtype=bool)
                if csel.any() and rsel.any():
                    sub = contains_xy(geom, lon[np.ix_(rsel, csel)], lat[np.ix_(rsel, csel)])
                    inside[np.ix_(rsel, csel)] = sub
                lat_mid = t.f - (lo + blk.shape[0] / 2.0) * PXDEG
                cell = (M_DEG * abs(t.a)) * (M_DEG * PXDEG * math.cos(math.radians(lat_mid)))
                nod = (blk == 0)
                b_bbox = int((lens & nod).sum())
                b_in = int((lens & nod & inside).sum())
                e_in = int((lens & inside).sum())
                tot['bbox_px'] += b_bbox; tot['bbox_m2'] += b_bbox * cell
                tot['in_px'] += b_in; tot['in_m2'] += b_in * cell
                tot['exp_px'] += e_in; tot['exp_m2'] += e_in * cell
                bands.append(dict(y0=y0, row_lo=int(lo), row_hi=int(hi),
                                  bbox_wedge_nodata_px=b_bbox, bbox_km2=round(b_bbox * cell / 1e6, 3),
                                  inprov_wedge_nodata_px=b_in, inprov_km2=round(b_in * cell / 1e6, 3),
                                  inprov_lens_expected_px=e_in))
        rec = dict(prov=prov, file=fname, W=W, H=H, buffer_deg=BUF, gcj='raw',
                   bbox_upper_bound=dict(px=int(tot['bbox_px']), km2=round(tot['bbox_m2'] / 1e6, 2)),
                   inprov=dict(px=int(tot['in_px']), km2=round(tot['in_m2'] / 1e6, 2),
                               lens_expected_px=int(tot['exp_px']),
                               fill_rate_pct=round(100.0 * (tot['exp_px'] - tot['in_px']) / tot['exp_px'], 2)
                               if tot['exp_px'] else None),
                   bands=bands)
        out.append(rec)
        print('%s: 省内损失 %d px = %.2f km²（期望楔区 %d px，缺填率 %.2f%%）；bbox 上界 %d px = %.2f km²' % (
            prov, tot['in_px'], tot['in_m2'] / 1e6, tot['exp_px'],
            100.0 * (tot['exp_px'] - tot['in_px']) / max(1, tot['exp_px']),
            tot['bbox_px'], tot['bbox_m2'] / 1e6))
    with open(os.path.join(HERE, 'impact_prov.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, 'impact_prov.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['prov', 'file', 'y0_band', 'row_lo', 'row_hi',
                    'bbox_wedge_nodata_px', 'bbox_km2', 'inprov_wedge_nodata_px', 'inprov_km2',
                    'inprov_lens_expected_px'])
        for r in out:
            for b in r['bands']:
                w.writerow([r['prov'], r['file'], b['y0'], b['row_lo'], b['row_hi'],
                            b['bbox_wedge_nodata_px'], b['bbox_km2'],
                            b['inprov_wedge_nodata_px'], b['inprov_km2'],
                            b['inprov_lens_expected_px']])
    print('WROTE impact_prov.json / impact_prov.csv')


if __name__ == '__main__':
    main()
