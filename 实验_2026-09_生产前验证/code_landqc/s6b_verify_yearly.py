# -*- coding: utf-8 -*-
"""s6b_verify_yearly.py — 逐年落盘的独立校验（对 s6 产物；另进程、另路径读取）

* 校验项（逐类）：
  V1 年份目录齐全（2017–2024 八目录，各含 .shp/.shx/.dbf/.prj/.cpg）
  V2 逐年字段与 2023 源完全一致（名/型/宽/小数位）
  V3 逐年点数 == 逐年质检统计.csv 的 n_in_folder；且 == 逐年池成员数.csv 的 n_in_pool（2023 除外）
  V4 逐年点坐标为源点集的子集（按 pid 对齐，随机 100 点坐标 0.0 差）
  V5 .prj 为 WGS84；统计 CSV 无 "不一致" 标记
* 用法：python s6b_verify_yearly.py --classes 草地,裸体...
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
YEARS = list(range(2017, 2025))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--out', default=os.path.join(DEST, '_逐年校验记录_s6b.json'))
    a = ap.parse_args()
    rep = {}
    for cls in [x.strip() for x in a.classes.split(',') if x.strip()]:
        root = os.path.join(DEST, cls)
        src = os.path.join(root, '2023', '%s_2023_WGS84.shp' % cls)
        r = shapefile.Reader(src, encoding='utf-8')
        sf = [(f[0], f[1], f[2], f[3]) for f in r.fields[1:]]
        fl = [f[0] for f in sf]
        ipid = fl.index('pid')
        src_pid, src_xy = {}, {}
        for rec, shp in zip(r.iterRecords(), r.iterShapes()):
            p = str(rec[ipid]); src_pid[p] = rec
            src_xy[p] = shp.points[0]
        r.close()
        st = pd.read_csv(os.path.join(root, '逐年质检统计.csv'), encoding='utf-8-sig')
        stn = {int(x.year): x for x in st.itertuples()}
        pool = pd.read_csv(os.path.join(root, '逐年池成员数.csv'), encoding='utf-8-sig')
        pn = {int(x.year): int(x.n_in_pool) for x in pool.itertuples()}
        res, ok = {}, True
        for y in YEARS:
            dd = os.path.join(root, str(y))
            files = ['%s_%d_WGS84%s' % (cls, y, e) for e in ('.shp', '.shx', '.dbf', '.prj', '.cpg')]
            miss = [f for f in files if not os.path.exists(os.path.join(dd, f))]
            if miss:
                res[y] = dict(err='缺文件 %s' % miss); ok = False; continue
            rr = shapefile.Reader(os.path.join(dd, files[0]), encoding='utf-8')
            n = len(rr)
            f2 = [(f[0], f[1], f[2], f[3]) for f in rr.fields[1:]]
            md = 0.0; nbad = 0
            idx = np.random.RandomState(11).permutation(n)[:min(100, n)]
            for i in idx:
                pid = str(rr.record(i)[ipid])
                if pid not in src_xy:
                    nbad += 1; continue
                px, py = rr.shape(i).points[0]
                md = max(md, abs(px - src_xy[pid][0]), abs(py - src_xy[pid][1]))
            prj = open(os.path.join(dd, files[3]), encoding='utf-8').read()
            rr.close()
            exp = int(stn[y].n_in_folder)
            exp_pool = pn.get(y)
            v = dict(n=n, stat=exp, pool=exp_pool, fields_ok=(f2 == sf), maxdiff=round(md, 9),
                     orphan=nbad, wgs84=('WGS' in prj or 'WGS_1984' in prj or 'WGS 84' in prj))
            v['ok'] = (n == exp) and (y == 2023 or n == exp_pool) and v['fields_ok'] and md < 1e-6 \
                and nbad == 0 and v['wgs84']
            ok &= v['ok']
            res[y] = v
        notes = st['note'].fillna('').astype(str)
        bad_note = notes[notes.str.contains('不一致')].tolist()
        rep[cls] = dict(n_src=len(src_pid), ok=bool(ok and not bad_note), years=res,
                        bad_notes=bad_note)
        print('%-6s %s' % (cls, 'PASS' if rep[cls]['ok'] else 'FAIL %s' % {k: v for k, v in res.items() if not (v.get('ok', False))}))
    old = json.load(open(a.out, encoding='utf-8')) if os.path.exists(a.out) else {}
    old.update(rep)
    json.dump(old, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('汇总：%d 类，PASS %d' % (len(rep), sum(1 for v in rep.values() if v['ok'])))


if __name__ == '__main__':
    main()
