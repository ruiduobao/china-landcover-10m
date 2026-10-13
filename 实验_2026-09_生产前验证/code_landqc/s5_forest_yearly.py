# -*- coding: utf-8 -*-
"""s5_forest_yearly.py — 四类林地"逐年质检结果"落地（2017–2024 × 4 类）；并集件归位

* 背景（用户 2026-10-13 排查）：w5_forest_assemble.py 只把"逐年影响"算成统计（per_year_counts.json），
  从未落逐年数据件；SHP 交付又把并集放进了 2023/ → 表现为"只有 2023"。
* 语义（与 w5 逐年统计完全一致）：某年的结果 = 该年训练子集（r7_train_<年>，30 类森林码 51/52/61/62/71/72/81/82）
  中、且已被并集质检的 4 类林点，取并集质检结论（layer=strict/clean；quarantine 不进数据件，另见剔除清单）。
* 做法：以现有交付 SHP（并集 clean 层，含 ct_prov/ct_code/ct_km）为记录源，按 row_id 逐年过滤重写 SHP；
  并集件从 <类>/2023/ 移到 <类>/ 根（命名 *_2017-2024并集_*）；逐年统计与剔除清单逐年成员一并落盘。
* 输出：林地_<类>/{2017..2024}/林地_<类>_<年>_WGS84.* + 根目录并集件 + 逐年质检统计.csv + 剔除清单_逐年成员.csv
* 校验：逐年四类计数 == per_year_counts.json（strict / quarantined / clean），逐点与并集坐标 0.0 差。
* 用法：python s5_forest_yearly.py
"""
import glob, json, os, shutil, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
ANN = r'F:/lc_work/年度子集_含稀有类'
FR = [51, 52, 61, 62, 71, 72, 81, 82]
C4 = {4: '常绿阔叶林', 5: '落叶阔叶林', 6: '常绿针叶林', 7: '落叶针叶林'}
FIELDS = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
          ('cls4', 'N', 2), ('clsname', 'C', 18), ('layer', 'C', 6), ('tier', 'C', 8),
          ('sug', 'N', 3), ('src', 'C', 24), ('ct_prov', 'C', 30), ('ct_code', 'C', 8), ('ct_km', 'N', 9, 2)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def write_pts(base, rows, xs, ys):
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding='utf-8')
    for f in FIELDS:
        w.field(f[0], f[1], f[2], f[3] if len(f) > 3 else 0)
    for (x, y, rec) in zip(xs, ys, rows):
        w.point(float(x), float(y)); w.record(*rec)
    w.close()
    return base


def sidecars(base, ref):
    for ext in ('.prj', '.cpg'):
        shutil.copy2(ref + ext, base + ext)


def main():
    # 1) 逐年成员（row_id）
    mem = {}
    for y in range(2017, 2025):
        s = pd.read_parquet(os.path.join(ANN, 'r7_train_%d.parquet' % y), columns=['row_id', 'class_new'])
        s = s[s.class_new.isin(FR)]
        mem[y] = s.row_id.to_numpy(np.int64)
        emit('%d 年森林成员 %d' % (y, len(mem[y])))
    pyd = json.load(open(r'F:/lc_work/v31_exp/data/forest_qc/out/per_year_counts.json', encoding='utf-8'))
    stats = []
    for c4, nm in C4.items():
        root = os.path.join(DEST, '林地_%s' % nm)
        # 2) 并集件归位：<类>/2023/<...>_2023_* → <类>/林地_<类>_2017-2024并集_*
        old = os.path.join(root, '2023', '林地_%s_2023_WGS84' % nm)
        new = os.path.join(root, '林地_%s_2017-2024并集_WGS84' % nm)
        if os.path.exists(old + '.shp') and not os.path.exists(new + '.shp'):
            for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                os.replace(old + ext, new + ext)
            emit('并集件归位 → %s' % os.path.relpath(new, DEST))
        ref = new if os.path.exists(new + '.shp') else old
        r = shapefile.Reader(ref + '.shp', encoding='utf-8')
        flds = [f[0] for f in r.fields[1:]]
        recs = list(r.iterRecords())
        pts = [p.shape.points[0] for p in r.shapeRecords()]
        pid = np.array([int(x[flds.index('pid')]) for x in recs], np.int64)
        r.close()
        i4, il, it, isu, isr = (flds.index('cls4'), flds.index('layer'), flds.index('tier'),
                                flds.index('sug'), flds.index('src'))
        ipr, ipv, ipc, ipk = (flds.index('prov'), flds.index('ct_prov'), flds.index('ct_code'),
                              flds.index('ct_km'))
        clsname = nm
        # 3) 逐年写出 + 统计
        for y in range(2017, 2025):
            m = np.isin(pid, mem[y])
            idx = np.where(m)[0]
            dd = os.path.join(root, str(y)); os.makedirs(dd, exist_ok=True)
            rows = [(str(pid[i]), pts[i][0], pts[i][1], recs[i][ipr], int(recs[i][i4]), clsname,
                     recs[i][il], recs[i][it], int(recs[i][isu]), recs[i][isr],
                     recs[i][ipv], recs[i][ipc], float(recs[i][ipk])) for i in idx]
            base = os.path.join(dd, '林地_%s_%d_WGS84' % (nm, y))
            write_pts(base, rows, [pts[i][0] for i in idx], [pts[i][1] for i in idx])
            sidecars(base, ref)
            # 回读校验
            rr = shapefile.Reader(base + '.shp', encoding='utf-8')
            assert len(rr) == len(rows)
            if len(idx):
                k = min(50, len(idx))
                for i in np.random.RandomState(5).permutation(len(idx))[:k]:
                    px, py = rr.shape(i).points[0]
                    assert abs(px - pts[idx[i]][0]) < 1e-6 and abs(py - pts[idx[i]][1]) < 1e-6
            rr.close()
            st = int((np.array([recs[i][il] for i in idx], dtype=object) == 'strict').sum()) if len(idx) else 0
            stats.append(dict(class4=c4, clsname=nm, year=y, n=len(idx), strict=st, clean=len(idx) - st))
            emit('  %s %d：%d 点（strict %d）' % (nm, y, len(idx), st))
    df = pd.DataFrame(stats).sort_values(['clsname', 'year'])
    df.to_csv(os.path.join(DEST, '林地_逐年质检统计.csv'), index=False, encoding='utf-8-sig')
    # 4) 与 per_year_counts.json 交叉核验（四类合计）
    ok = True
    for y in range(2017, 2025):
        g = df[df.year == y]
        exp_n = pyd[str(y)]['n_forest'] - pyd[str(y)]['quarantined']
        exp_st = pyd[str(y)]['strict']
        line = '  %d：交付 %d（期望 %d）｜strict %d（期望 %d）' % (y, g.n.sum(), exp_n, g.strict.sum(), exp_st)
        if g.n.sum() != exp_n or g.strict.sum() != exp_st:
            line += '  ✗'; ok = False
        else:
            line += '  ✓'
        print(line)
    # 5) 剔除清单逐年成员
    q = pd.read_csv(r'F:/lc_work/v31_exp/data/forest_qc/qc/quarantine.csv', encoding='utf-8-sig',
                    usecols=['point_id', 'class4', 'prov', 'src', 'tier'])
    q['row_id'] = q.point_id.astype(np.int64)
    for y in range(2017, 2025):
        q['in_%d' % y] = np.isin(q.row_id.to_numpy(np.int64), mem[y])
    q.drop(columns=['row_id']).to_csv(os.path.join(DEST, '林地_剔除清单_逐年成员.csv'),
                                      index=False, encoding='utf-8-sig')
    print('交叉核验：', '全 PASS' if ok else '存在不一致！')
    print('→ 林地_逐年质检统计.csv / 林地_剔除清单_逐年成员.csv')


if __name__ == '__main__':
    main()
