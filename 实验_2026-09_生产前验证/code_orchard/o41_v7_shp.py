# -*- coding: utf-8 -*-
"""o41_v7_shp.py — 乔灌园地 v7 逐年面板 → 8 套 WGS84 SHP 交付（含 CTAmap 省归属）

* 输入：v7_逐年面板/乔灌园地v7_点表.csv（160,931 点：uid/lon/lat/subtype/grade/srcset/rec）
* 规则源：
  ① 坐标制式（用户 2026-10-13 指令）：只用 WGS84，不做 GCJ-02；
  ② 区划用 CTAmap T2026_province（GCS_WGS_1984）：界内 contains 赋值，**界外点不丢**，
     按 STRtree 最近省赋值 + ct_km（geodesic 最近距离 km，界内=0）；
  ③ SHP 写出：pyshp，.cpg=UTF-8，.prj=OGC WKT1（WGS84）；DBF 字段名 ASCII ≤10 字节。
* 输出：v7_逐年面板/<年>/乔灌园地v7_<年>_WGS84.{shp,shx,dbf,prj,cpg} × 8
        + v7_shp_report.json（逐文件回读记录数/抽检坐标一致）
* 用法：python o41_v7_shp.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import numpy as np
import pandas as pd
import shapefile
from shapely import STRtree, contains_xy, shortest_line
from shapely.geometry import Point
from pyproj import Geod
from k23_shp_fix import write_sidecars, WGS84_WKT

DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v7_逐年面板'
PROV = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
GEOD = Geod(ellps='WGS84')
FIELDS = [('uid', 'N', 12, 0), ('lon', 'F', 12, 7), ('lat', 'F', 12, 7),
          ('subtype', 'C', 40), ('grade', 'C', 40), ('srcset', 'C', 20), ('rec', 'L'),
          ('ct_prov', 'C', 30), ('ct_code', 'C', 8), ('ct_km', 'F', 10, 2)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    import geopandas as gpd
    pts = pd.read_csv(os.path.join(DEST, '乔灌园地v7_点表.csv'), encoding='utf-8-sig')
    emit('点表 %d 行' % len(pts))
    prov = gpd.read_file(PROV)
    names = prov['省'].astype(str).to_numpy()
    codes = prov['省级码'].astype(str).to_numpy()
    geoms = prov.geometry.to_numpy()
    tree = STRtree(list(geoms))
    lon, lat = pts.lon.to_numpy(), pts.lat.to_numpy()
    inside_i = np.full(len(pts), -1)
    for i, geom in enumerate(geoms):
        m = contains_xy(geom, lon, lat)
        inside_i[m & (inside_i < 0)] = i
    ctp = np.where(inside_i >= 0, names[np.clip(inside_i, 0, None)], '')
    ctc = np.where(inside_i >= 0, codes[np.clip(inside_i, 0, None)], '')
    km = np.zeros(len(pts))
    na = np.where(ctp == '')[0]
    if len(na):
        for i in na:
            j = tree.nearest(Point(lon[i], lat[i]))
            ctp[i], ctc[i] = names[j], codes[j]
            ln = shortest_line(Point(lon[i], lat[i]), geoms[j])
            x1, y1 = ln.coords[0][:2]
            x2, y2 = ln.coords[-1][:2]
            km[i] = GEOD.inv(x1, y1, x2, y2)[2] / 1000.0
        emit('界外点 %d（%.2f%%）按最近省赋值，最远 %.1f km' % (len(na), 100 * len(na) / len(pts), km.max()))
    else:
        emit('全部点在省界内')
    rep = {'n_points': int(len(pts)), 'n_outside': int(len(na)),
           'km_max': round(float(km.max()), 2), 'files': {}}
    for y in range(2017, 2025):
        yd = os.path.join(DEST, str(y))
        os.makedirs(yd, exist_ok=True)
        fp = os.path.join(yd, '乔灌园地v7_%d_WGS84.shp' % y)
        w = shapefile.Writer(fp, shapeType=shapefile.POINT, encoding='utf-8', encodingErrors='strict')
        for f in FIELDS:
            w.field(*f)
        for i in range(len(pts)):
            w.point(float(lon[i]), float(lat[i]))
            w.record(int(pts.uid.iloc[i]), float(lon[i]), float(lat[i]),
                     str(pts.subtype.iloc[i]), str(pts.grade.iloc[i]), str(pts.srcset.iloc[i]),
                     bool(pts.rec.iloc[i]), str(ctp[i]), str(ctc[i]), round(float(km[i]), 2))
        w.close()
        write_sidecars(fp, WGS84_WKT, 'wgs84')
        # 回读校验（几何 float64 全精度；DBF 12.7 字段 ≈1.1 cm 舍入）
        r = shapefile.Reader(fp, encoding='utf-8')
        n_rd = len(r)
        recs = r.records()
        idx = np.random.default_rng(7).choice(len(pts), size=min(200, len(pts)), replace=False)
        ok_g = all(np.allclose(r.shape(int(i)).points[0], (lon[i], lat[i]), atol=1e-9) for i in idx)
        ok_d = all(abs(recs[i][1] - lon[i]) < 1e-6 and abs(recs[i][2] - lat[i]) < 1e-6 for i in idx)
        r.close()
        rep['files'][str(y)] = {'records': int(n_rd), 'geom_check': bool(ok_g), 'dbf_check': bool(ok_d),
                                'size_mb': round(os.path.getsize(fp) / 1e6, 2)}
        emit('%d：%d 记录，几何 %s / DBF %s（%.1f MB）' % (
            y, n_rd, 'PASS' if ok_g else 'FAIL', 'PASS' if ok_d else 'FAIL', os.path.getsize(fp) / 1e6))
    with open(os.path.join(DEST, 'v7_shp_report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    emit('→ v7_shp_report.json')


if __name__ == '__main__':
    main()
