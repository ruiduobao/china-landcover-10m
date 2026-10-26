# -*- coding: utf-8 -*-
"""f8_v8_shp.py — v8 样本集 SHP 导出（正样本 8 年逐年 + 负样本单层），WGS84 + CTAmap 归属

* 输入：v8_ForTy提纯扩量/乔灌园地v8_正样本_点表.csv、乔灌园地v8_负样本_点表.csv
* 规则源：仅 WGS84（无 GCJ-02）；.cpg=UTF-8、.prj=OGC WKT1；DBF 字段名 ASCII ≤10 字节；
  坐标 DBF 存 7 位小数（≈1.1 cm，几何 float64 全精度）；回读校验（计数 + 几何 1e-9 + DBF 1e-6）
* 输出：<v8>/2017..2024/乔灌园地v8_<年>_WGS84.{shp,shx,dbf,prj,cpg}（每年同一批 250,826 点，标签继承）
        <v8>/负样本/乔灌园地v8_负样本_WGS84.*（118,660 点）
        <v8>/v8_shp_report.json
* 用法：python f8_v8_shp.py
"""
import json
import os
import sys
import time

sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import shapefile
from k23_shp_fix import write_sidecars, WGS84_WKT

DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v8_ForTy提纯扩量'
POS = os.path.join(DEST, '乔灌园地v8_正样本_点表.csv')
NEG = os.path.join(DEST, '乔灌园地v8_负样本_点表.csv')
PFIELDS = [('uid', 'N', 10, 0), ('lon', 'F', 12, 7), ('lat', 'F', 12, 7),
           ('subtype', 'C', 24), ('grade', 'C', 24), ('srcset', 'C', 20),
           ('status', 'C', 20), ('src', 'C', 16), ('f_agree', 'N', 1, 0), ('p5', 'F', 8, 3),
           ('ct_prov', 'C', 24), ('ct_code', 'C', 8), ('ct_km', 'F', 10, 2),
           ('elev', 'F', 10, 1), ('slope', 'F', 8, 1), ('rec', 'L')]
NFIELDS = [('uid', 'N', 10, 0), ('lon', 'F', 12, 7), ('lat', 'F', 12, 7),
           ('neg_type', 'C', 8), ('dom', 'N', 2, 0), ('pur', 'F', 8, 3), ('p5', 'F', 8, 3),
           ('d_v7_km', 'F', 10, 3), ('d_pos_km', 'F', 10, 3),
           ('ct_prov', 'C', 24), ('ct_code', 'C', 8), ('ct_km', 'F', 10, 2),
           ('elev', 'F', 10, 1), ('slope', 'F', 8, 1), ('label', 'N', 1, 0)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def num(v, d=np.nan):
    try:
        return float(v) if pd.notna(v) else d
    except Exception:
        return d


def write_points(fp, rows, fields):
    w = shapefile.Writer(fp, shapeType=shapefile.POINT, encoding='utf-8', encodingErrors='replace')
    for f in fields:
        w.field(*f)
    for r in rows:
        w.point(float(r['lon']), float(r['lat']))
        w.record(*r['rec'])
    w.close()
    write_sidecars(fp, WGS84_WKT, 'wgs84')


def verify(fp, pts, tol_dbf=1e-6):
    r = shapefile.Reader(fp, encoding='utf-8')
    n = len(r)
    idx = np.random.default_rng(7).choice(len(pts), size=min(200, len(pts)), replace=False)
    ok_g = all(np.allclose(r.shape(int(i)).points[0], (pts.lon.iloc[i], pts.lat.iloc[i]), atol=1e-9) for i in idx)
    recs = r.records()
    ok_d = all(abs(recs[i][1] - pts.lon.iloc[i]) < tol_dbf and abs(recs[i][2] - pts.lat.iloc[i]) < tol_dbf for i in idx)
    r.close()
    return n, ok_g, ok_d


def main():
    pos = pd.read_csv(POS, encoding='utf-8-sig')
    neg = pd.read_csv(NEG, encoding='utf-8-sig')
    emit('正样本 %d / 负样本 %d' % (len(pos), len(neg)))
    pos_rows = []
    for r in pos.itertuples():
        pos_rows.append({'lon': r.lon, 'lat': r.lat, 'rec': [
            int(r.uid), float(r.lon), float(r.lat),
            str(r.subtype), str(r.grade), str(r.srcset), str(r.status), str(r.src),
            int(r.forty_agree), round(num(r.p5, 0), 3),
            str(r.ct_prov) if pd.notna(r.ct_prov) else '', str(r.ct_code) if pd.notna(r.ct_code) else '',
            round(num(r.ct_km, 0), 2), round(num(r.elev), 1) if pd.notna(r.elev) else 0.0,
            round(num(r.slope), 1) if pd.notna(r.slope) else 0.0, bool(r.rec) if pd.notna(r.rec) else True]})
    rep = {'pos_points': int(len(pos)), 'neg_points': int(len(neg)), 'files': {}}
    for y in range(2017, 2025):
        yd = os.path.join(DEST, str(y))
        os.makedirs(yd, exist_ok=True)
        fp = os.path.join(yd, '乔灌园地v8_%d_WGS84.shp' % y)
        write_points(fp, pos_rows, PFIELDS)
        n, og, od = verify(fp, pos)
        rep['files']['%d' % y] = {'records': n, 'geom': bool(og), 'dbf': bool(od)}
        emit('%d：%d 记录 几何 %s / DBF %s' % (y, n, 'PASS' if og else 'FAIL', 'PASS' if od else 'FAIL'))
    nd = os.path.join(DEST, '负样本')
    os.makedirs(nd, exist_ok=True)
    fn = os.path.join(nd, '乔灌园地v8_负样本_WGS84.shp')
    nrows = [{'lon': r.lon, 'lat': r.lat, 'rec': [
        int(r.uid), float(r.lon), float(r.lat), str(r.neg_type), int(r.dom), round(num(r.pur, 0), 3),
        round(num(r.p5, 0), 3), round(num(r.dist_v7_km, 0), 3), round(num(r.dist_pos_km, 0), 3),
        str(r.ct_prov) if pd.notna(r.ct_prov) else '', str(r.ct_code) if pd.notna(r.ct_code) else '',
        round(num(r.ct_km, 0), 2), round(num(r.elev), 1) if pd.notna(r.elev) else 0.0,
        round(num(r.slope), 1) if pd.notna(r.slope) else 0.0, 0]} for r in neg.itertuples()]
    write_points(fn, nrows, NFIELDS)
    n2, og2, od2 = verify(fn, neg)
    rep['files']['neg'] = {'records': n2, 'geom': bool(og2), 'dbf': bool(od2)}
    emit('负样本：%d 记录 几何 %s / DBF %s' % (n2, 'PASS' if og2 else 'FAIL', 'PASS' if od2 else 'FAIL'))
    json.dump(rep, open(os.path.join(DEST, 'v8_shp_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ v8_shp_report.json')


if __name__ == '__main__':
    main()
