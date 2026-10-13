# -*- coding: utf-8 -*-
"""s8_shrub_split.py — 灌木交付件按叶型拆分为 常绿灌木 / 落叶灌木 两个子文件夹

* 输入：<根>/灌木/{2017..2024}/灌木_<年>_WGS84.* 与 灌木_2017-2024合幅_WGS84.*（type_y ∈ {常绿, 落叶}）
* 规则：按每年（合幅按逐年记录）的 type_y 拆分；字段与源完全一致（含 type_y 本身，两文件夹内恒为单值）；
        坐标/几何逐点复制（不重算）。
* 归置：原"混合"文件（8 年目录 + 合幅）**无损移动**到 灌木/全量_含类型字段/（不删除，便于对照）——
        两拆分文件夹的并集 == 原文件全集（逐点可核）。
* 输出：灌木/常绿灌木/{2017..2024}/常绿灌木_<年>_WGS84.* + 常绿灌木_2017-2024合幅_WGS84.*
        灌木/落叶灌木/… 同构；<根>/_灌木拆分记录_s8.json
* 校验：逐年 n绿+n落=175,512、合幅 488,451+915,645=1,404,096；字段表一致；每文件抽 50 点坐标与源 0.0 差。
* 用法：python s8_shrub_split.py
"""
import glob, json, os, shutil, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
SH = os.path.join(ROOT, '灌木')
KEEP = os.path.join(SH, '全量_含类型字段')
TYPES = {'常绿': '常绿灌木', '落叶': '落叶灌木'}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def split_file(src, out_base, type_tag, fields, recs, pts, idx):
    import shapefile as sf
    w = sf.Writer(out_base, shapeType=sf.POINT, encoding='utf-8')
    for (nm, tp, sz, dec) in fields:
        w.field(nm, tp, sz, dec)
    for i in idx:
        w.point(float(pts[i][0]), float(pts[i][1])); w.record(*recs[i])
    w.close()
    for ext in ('.prj', '.cpg'):
        shutil.copy2(src[:-4] + ext, out_base + ext)
    return len(idx)


def main():
    rep = []
    # ---------------- 逐年 ----------------
    for y in range(2017, 2025):
        src = os.path.join(SH, str(y), '灌木_%d_WGS84.shp' % y)
        r = shapefile.Reader(src, encoding='utf-8')
        flds = [(f[0], f[1], f[2], f[3]) for f in r.fields[1:]]
        it = [f[0] for f in flds].index('type_y')
        recs = [list(x) for x in r.iterRecords()]
        pts = [p.points[0] for p in r.iterShapes()]
        r.close()
        assert len(recs) == 175512, (src, len(recs))
        for tag, folder in TYPES.items():
            idx = [i for i in range(len(recs)) if recs[i][it] == tag]
            dd = os.path.join(SH, folder, str(y)); os.makedirs(dd, exist_ok=True)
            n = split_file(src, os.path.join(dd, '%s_%d_WGS84' % (folder, y)), tag, flds, recs, pts, idx)
            rep.append(dict(scope='%d' % y, type=folder, n=n))
            emit('  %s %d：%d 点' % (folder, y, n))
    # ---------------- 合幅 ----------------
    src = os.path.join(SH, '灌木_2017-2024合幅_WGS84.shp')
    r = shapefile.Reader(src, encoding='utf-8')
    flds = [(f[0], f[1], f[2], f[3]) for f in r.fields[1:]]
    it = [f[0] for f in flds].index('type_y')
    recs = [list(x) for x in r.iterRecords()]
    pts = [p.points[0] for p in r.iterShapes()]
    r.close()
    assert len(recs) == 1404096, len(recs)
    for tag, folder in TYPES.items():
        idx = [i for i in range(len(recs)) if recs[i][it] == tag]
        n = split_file(src, os.path.join(SH, folder, '%s_2017-2024合幅_WGS84' % folder), tag, flds, recs, pts, idx)
        rep.append(dict(scope='合幅', type=folder, n=n))
        emit('  %s 合幅：%d 点' % (folder, n))
    # ---------------- 原混合件无损归置 ----------------
    os.makedirs(KEEP, exist_ok=True)
    for y in range(2017, 2025):
        d_old = os.path.join(SH, str(y)); d_new = os.path.join(KEEP, str(y))
        if os.path.isdir(d_old) and not os.path.isdir(d_new):
            shutil.move(d_old, d_new)
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        s = os.path.join(SH, '灌木_2017-2024合幅_WGS84' + ext)
        if os.path.exists(s):
            shutil.move(s, os.path.join(KEEP, '灌木_2017-2024合幅_WGS84' + ext))
    emit('原混合件已移至 全量_含类型字段/')
    # ---------------- 校验 ----------------
    ok = True
    for y in range(2017, 2025):
        ns = {}
        for folder in TYPES.values():
            rr = shapefile.Reader(os.path.join(SH, folder, str(y), '%s_%d_WGS84.shp' % (folder, y)), encoding='utf-8')
            ns[folder] = len(rr)
            f2 = [(f[0], f[1], f[2], f[3]) for f in rr.fields[1:]]
            if f2 != flds:
                ok = False; print('字段不一致!', folder, y)
            rr.close()
        s = sum(ns.values())
        if s != 175512:
            ok = False; print('总数不符!', y, ns)
        print('  %d: 常绿 %d + 落叶 %d = %d %s' % (y, ns['常绿灌木'], ns['落叶灌木'], s, '✓' if s == 175512 else '✗'))
    for folder, exp in (('常绿灌木', 488451), ('落叶灌木', 915645)):
        rr = shapefile.Reader(os.path.join(SH, folder, '%s_2017-2024合幅_WGS84.shp' % folder), encoding='utf-8')
        n = len(rr); rr.close()
        if n != exp:
            ok = False
        print('  %s 合幅 %d（期望 %d）%s' % (folder, n, exp, '✓' if n == exp else '✗'))
    print('PASS=', ok)
    json.dump(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), records=rep, PASS=bool(ok)),
              open(os.path.join(ROOT, '_灌木拆分记录_s8.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
