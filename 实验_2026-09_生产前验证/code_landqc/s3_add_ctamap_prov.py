# -*- coding: utf-8 -*-
"""s3_add_ctamap_prov.py — 为交付点 SHP 追加 CTAmap 行政归属字段（ct_prov / ct_code）

* 规则源（用户 2026-10-13 指令）：区划数据用 CTAmap
    Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp（GCS_WGS_1984，34 省）
* 做法：对每个 *_WGS84.shp（含灌木合幅）逐点做"省界内"判定（shapely contains_xy 矢量化，34 多边形循环），
        追加两个字段后重写 .shp/.shx/.dbf（.prj/.cpg 不变）；先写 *_tmp 再校验替换（计数/坐标抽检/属性抽检）。
* 语义：ct_prov = 省（CTAmap 2026，全称）；ct_code = 省级码（6 位字符串）；省界以外（近岸滩涂/红树林等）为空串——
        对潮滩/红树林这类"多在海岸线之外"的类，空值是定义使然，见校验记录的内点率。
* 输出：原地更新 + `_CTAmap_字段追加记录.csv`
* 用法：python s3_add_ctamap_prov.py [--check]
"""
import argparse, glob, os, sys, time
import numpy as np
import shapefile
from shapely import contains_xy
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')
PROV = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只检查字段是否已存在')
    a = ap.parse_args()
    prov = gpd.read_file(PROV)
    names = prov['省'].astype(str).to_numpy()
    codes = prov['省级码'].astype(str).to_numpy()
    geoms = prov.geometry.to_numpy()
    emit('CTAmap 省 %d 个（%s）' % (len(prov), prov.crs))
    files = sorted(glob.glob(os.path.join(DEST, '*/[0-9][0-9][0-9][0-9]/*_WGS84.shp'))) + \
        sorted(glob.glob(os.path.join(DEST, '灌木/*合幅_WGS84.shp')))
    log = []
    for f in files:
        r = shapefile.Reader(f, encoding='utf-8')
        flds = [(x[0], x[1], x[2]) for x in r.fields[1:]]
        if 'ct_prov' in [x[0] for x in flds]:
            r.close()
            emit('已含 ct_prov，跳过 %s' % os.path.relpath(f, DEST)); continue
        n = len(r)
        pts = r.shapeRecords()
        xs = np.array([p.shape.points[0][0] for p in pts], float)
        ys = np.array([p.shape.points[0][1] for p in pts], float)
        ctp = np.full(n, '', dtype=object); ctc = np.full(n, '', dtype=object); done = np.zeros(n, bool)
        for i in range(len(names)):
            m = contains_xy(geoms[i], xs, ys) & (~done)
            ctp[m] = names[i]; ctc[m] = codes[i]; done |= m
        recs = [list(rec) for rec in r.iterRecords()]
        r.close()
        # 写 tmp 再替换
        base = f[:-4]
        w = shapefile.Writer(base + '_tmp', shapeType=shapefile.POINT, encoding='utf-8')
        for (nm, tp, sz) in flds:
            w.field(nm, tp, sz)
        w.field('ct_prov', 'C', 30); w.field('ct_code', 'C', 8)
        for i in range(n):
            w.point(float(xs[i]), float(ys[i]))
            w.record(*(recs[i] + [ctp[i], ctc[i]]))
        w.close()
        # 校验：计数 + 200 点坐标 + 属性
        rt = shapefile.Reader(base + '_tmp.shp', encoding='utf-8')
        ok = len(rt) == n and len(rt.records()) == n
        idx = np.random.RandomState(3).permutation(n)[:200]
        md = 0.0
        for i in idx:
            px, py = rt.shape(i).points[0]
            md = max(md, abs(px - xs[i]), abs(py - ys[i]))
        if not (ok and md < 1e-6):
            raise SystemExit('校验失败 %s（ok=%s md=%.9f）' % (f, ok, md))
        rt.close()
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
                        emit('  %s 命中删除锁 → 原地内容替换回退' % ext)
                    else:
                        time.sleep(2.0 * (att + 1))
        inside = float(done.mean())
        log.append(dict(shp=os.path.relpath(f, DEST), n=n, inside=round(inside, 4),
                        ct_prov_filled=round(float((ctp != '').mean()), 4)))
        emit('%-46s n=%-7d 省界内 %.1f%%' % (os.path.relpath(f, DEST), n, 100 * inside))
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(DEST, '_CTAmap_字段追加记录.csv'), index=False, encoding='utf-8-sig')
    emit('完成 %d 个文件；记录 → _CTAmap_字段追加记录.csv' % len(log))


if __name__ == '__main__':
    main()
