# -*- coding: utf-8 -*-
"""s4_ctamap_nearest.py — 省界外点按"最近省份"补 CTAmap 归属 + 追加 ct_km 距离字段

* 背景（用户 2026-10-13 指令）：红树林/潮滩/海岸盐沼等常落在省界矢量之外，**这些点不能丢**，
  也要有省份归属 → 对省界外的点按最近省份（geodesic 最近）赋值；同时给所有点加 `ct_km`
  （到 CTAmap 省界的最近距离，km；界内=0），离岸程度一目了然。
* 规则源：Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp（GCS_WGS_1984）
* 做法：STRtree.nearest + shortest_line + pyproj.Geod 精确距离；重写 .shp/.shx/.dbf（含删除锁回退），逐文件校验。
* 输出：38 套 SHP 原地更新 + `_CTAmap_字段追加记录.csv`（含 界内/就近/最大距离 统计）
* 用法：python s4_ctamap_nearest.py
"""
import glob, os, sys, time
import numpy as np
import shapefile
import geopandas as gpd
import shapely
from shapely import STRtree, shortest_line, Point
from pyproj import Geod

sys.stdout.reconfigure(encoding='utf-8')
PROV = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
GEOD = Geod(ellps='WGS84')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    prov = gpd.read_file(PROV)
    names = prov['省'].astype(str).tolist()
    codes = prov['省级码'].astype(str).tolist()
    geoms = list(prov.geometry)
    tree = STRtree(geoms)
    emit('CTAmap 省 %d（%s）' % (len(prov), prov.crs))
    files = sorted(glob.glob(os.path.join(DEST, '*/[0-9][0-9][0-9][0-9]/*_WGS84.shp'))) + \
        sorted(glob.glob(os.path.join(DEST, '灌木/*合幅_WGS84.shp')))
    log = []
    for f in files:
        if '_母库点_' in f:
            continue
        r = shapefile.Reader(f, encoding='utf-8')
        flds = [(x[0], x[1], x[2]) for x in r.fields[1:]]
        fl = [x[0] for x in flds]
        if 'ct_km' in fl:
            r.close(); emit('已含 ct_km，跳过 %s' % os.path.relpath(f, DEST)); continue
        ip, ic = fl.index('ct_prov'), fl.index('ct_code')
        n = len(r)
        pts = [p.shape.points[0] for p in r.shapeRecords()]
        recs = [list(x) for x in r.iterRecords()]
        r.close()
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        ctp = np.array([rec[ip] for rec in recs], dtype=object)
        ctc = np.array([rec[ic] for rec in recs], dtype=object)
        km = np.zeros(n)
        na = (ctp == '')
        if na.any():
            for i in np.where(na)[0]:
                pt = Point(xs[i], ys[i])
                j = int(tree.nearest(pt))
                line = shortest_line(pt, geoms[j])
                ex, ey = line.coords[-1]
                _, _, d = GEOD.inv(xs[i], ys[i], ex, ey)
                ctp[i] = names[j]; ctc[i] = codes[j]; km[i] = d / 1000.0
        # 重写（新字段 ct_km 追加在末尾）
        base = f[:-4]
        w = shapefile.Writer(base + '_tmp', shapeType=shapefile.POINT, encoding='utf-8')
        for (nm, tp, sz) in flds:
            w.field(nm, tp, sz)
        w.field('ct_km', 'N', 9, 2)
        for i in range(n):
            w.point(float(xs[i]), float(ys[i]))
            recs[i][ip] = ctp[i]; recs[i][ic] = ctc[i]
            w.record(*(recs[i] + [round(float(km[i]), 2)]))
        w.close()
        rt = shapefile.Reader(base + '_tmp.shp', encoding='utf-8')
        ok = len(rt) == n and len(rt.records()) == n
        idx = np.random.RandomState(3).permutation(n)[:200]
        md = max(abs(rt.shape(i).points[0][0] - xs[i]) for i in idx)
        rt.close()
        if not (ok and md < 1e-6):
            raise SystemExit('校验失败 %s' % f)
        for ext in ('.shp', '.shx', '.dbf'):
            for att in range(4):
                try:
                    os.replace(base + '_tmp' + ext, base + ext); break
                except PermissionError:
                    if att == 3:
                        data = open(base + '_tmp' + ext, 'rb').read()
                        with open(base + ext, 'wb') as fh:
                            fh.write(data)
                        os.remove(base + '_tmp' + ext)
                        emit('  %s 命中删除锁 → 原地替换回退' % ext)
                    else:
                        time.sleep(2.0 * (att + 1))
        nn = int(na.sum())
        log.append(dict(shp=os.path.relpath(f, DEST), n=n, inside=n - nn, nearest=nn,
                        km_max=round(float(km.max()), 2) if nn else 0.0,
                        km_med=round(float(np.median(km[na])), 2) if nn else 0.0))
        emit('%-48s n=%-7d 界内 %-7d 就近 %-5d 最远 %.1f km' % (
            os.path.relpath(f, DEST), n, n - nn, nn, km.max() if nn else 0.0))
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(DEST, '_CTAmap_字段追加记录.csv'), index=False, encoding='utf-8-sig')
    emit('完成 %d 套；记录 → _CTAmap_字段追加记录.csv' % len(log))


if __name__ == '__main__':
    main()
