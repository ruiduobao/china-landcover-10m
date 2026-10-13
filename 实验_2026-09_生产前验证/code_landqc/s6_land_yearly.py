# -*- coding: utf-8 -*-
"""s6_land_yearly.py — 其余地类与沼泽的"逐年质检结果"落盘（2017–2024 × 16 类）

* 语义（与林地 s5_forest_yearly.py 完全一致）：`<年>/` 目录 = **质检后的交付点集 ∩ 该年训练池**
  （`r7_train_<年>` 的 row_id 集合）；**只筛选"该点在该年池中与否"，不改任何 QC 结论**
  （源文件必须是质检交付件 —— 脚本用字段指纹校验：地类 14 类需含 `tier`，沼泽需含 `flood_m/max300`，全部需含 `ct_km`）。
* 2023 目录 = 交付全量（现状保留，不重写）：14 类的交付集 100% 落在 r7_train_2023，故 2023 全量 = 2023 池成员；
  沼泽两类含 2023 口径的**新增扩展点**（不在任何历史年池），故其 2023 目录 = 全量（含新增），统计 CSV 用 note 注明。
* 输出：<类>/{2017..2022, 2024}/<类>_<年>_WGS84.*（2023 不动）；<类>/逐年质检统计.csv；
        <类>/<类>_剔除清单_逐年成员.csv（隔离点 + in_YYYY 成员列）
* 校验：逐年 n 与 `逐年池成员数.csv` 逐值一致（2023 除外，按上语义）；每年 50 点坐标回读 0.0 差。
* 用法：python s6_land_yearly.py --classes 草地,裸地,木本沼泽,...
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
ANN = r'F:/lc_work/年度子集_含稀有类'
YEARS = list(range(2017, 2025))
SWAMP = ('木本沼泽', '草本沼泽')
_LAND = {1: '草本旱地', 3: '灌溉耕地', 11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 22: '裸地',
         16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 23: '水体',
         21: '不透水面', 24: '永久冰雪'}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True, help='逗号分隔类名（须在 _LAND 或 SWAMP 中）')
    ap.add_argument('--years', default='2017,2018,2019,2020,2021,2022,2024',
                    help='要写出的年份（默认除 2023 外的 7 年；2023 目录=交付全量，不重写）')
    a = ap.parse_args()
    classes = [x.strip() for x in a.classes.split(',') if x.strip()]
    bad = [c for c in classes if c not in _LAND.values() and c not in SWAMP]
    if bad:
        raise SystemExit('未知类名: %s' % bad)
    # 逐年池成员（row_id 集合，只读）
    mem = {}
    for y in YEARS:
        s = pd.read_parquet(os.path.join(ANN, 'r7_train_%d.parquet' % y), columns=['row_id'])
        mem[y] = s.row_id.to_numpy(np.int64)
        emit('%d 池 %d 点' % (y, len(mem[y])))
    want_years = [int(x) for x in a.years.split(',')]
    rep = {}
    for cls in classes:
        root = os.path.join(DEST, cls)
        src = os.path.join(root, '2023', '%s_2023_WGS84.shp' % cls)
        r = shapefile.Reader(src, encoding='utf-8')
        flds = [(f[0], f[1], f[2], f[3]) for f in r.fields[1:]]
        fl = [f[0] for f in flds]
        # —— 质检件指纹校验（防用错源） ——
        need = ['ct_km', 'ct_prov', 'ct_code']
        if cls in SWAMP:
            need += ['flood_m', 'max300', 'clsw']
        else:
            need += ['tier', 'cls24']
        miss = [c for c in need if c not in fl]
        assert not miss, '%s 源缺质检字段 %s（疑似非质检交付件）' % (cls, miss)
        recs = [list(x) for x in r.iterRecords()]
        pts = [p.shape.points[0] for p in r.shapeRecords()]
        r.close()
        pid_s = pd.Series([str(x[fl.index('pid')]) for x in recs])
        pid_num = pd.to_numeric(pid_s, errors='coerce')
        ok_num = pid_num.notna().to_numpy()
        pid = pid_num.fillna(-1).astype(np.int64).to_numpy()   # 非数字 pid（扩展点）→ -1，永不命中池
        n_all = len(pid)
        tier_i = fl.index('tier') if 'tier' in fl else None
        pool_csv = pd.read_csv(os.path.join(root, '逐年池成员数.csv'), encoding='utf-8-sig')
        pool_n = {int(x.year): int(x.n_in_pool) for x in pool_csv.itertuples()}
        # 2023 自检：地类须 100% 在 2023 池；沼泽允许新增点
        m23 = ok_num & np.isin(pid, mem[2023])
        in23 = int(m23.sum())
        n_new = int((~ok_num).sum())
        if cls not in SWAMP:
            assert in23 == n_all, '%s 交付集并非全部在 2023 池（%d/%d）' % (cls, in23, n_all)
        stats = []
        for y in YEARS:
            m = ok_num & np.isin(pid, mem[y])
            if y == 2023:
                stats.append(dict(year=y, n_pool=pool_n.get(y, in23), n_in_folder=n_all,
                                  strong='', weak='',
                                  note='交付全量（r7_train_2023 池成员 %d / 非 2023 池点 %d%s）' % (
                                      in23, n_all - in23,
                                      '' if cls not in SWAMP else
                                      '：其中新增扩展点（DS-/RC-/OS- 非数字 id）%d' % n_new)))
                continue
            assert y in want_years
            idx = np.where(m)[0]
            dd = os.path.join(root, str(y)); os.makedirs(dd, exist_ok=True)
            base = os.path.join(dd, '%s_%d_WGS84' % (cls, y))
            w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding='utf-8')
            for (nm, tp, sz, dec) in flds:
                w.field(nm, tp, sz, dec)
            for i in idx:
                w.point(float(pts[i][0]), float(pts[i][1]))
                w.record(*recs[i])
            w.close()
            for ext in ('.prj', '.cpg'):
                shutil.copy2(src[:-4] + ext, base + ext)
            # 回读校验
            rr = shapefile.Reader(base + '.shp', encoding='utf-8')
            assert len(rr) == len(idx)
            for i in np.random.RandomState(9).permutation(len(idx))[:min(50, len(idx))]:
                px, py = rr.shape(i).points[0]
                assert abs(px - pts[idx[i]][0]) < 1e-6 and abs(py - pts[idx[i]][1]) < 1e-6
            rr.close()
            st = int(sum(1 for i in idx if tier_i is not None and recs[i][tier_i] == 'strong'))
            wk = int(sum(1 for i in idx if tier_i is not None and recs[i][tier_i] == 'weak'))
            stats.append(dict(year=y, n_pool=pool_n.get(y, -1), n_in_folder=len(idx),
                              strong=st if tier_i is not None else '',
                              weak=wk if tier_i is not None else '',
                              note='' if len(idx) == pool_n.get(y, -1) else '与池成员数不一致!'))
            emit('  %s %d：%d 点（池 %d）' % (cls, y, len(idx), pool_n.get(y, -1)))
        df = pd.DataFrame(stats)[['year', 'n_pool', 'n_in_folder', 'strong', 'weak', 'note']]
        df.to_csv(os.path.join(root, '逐年质检统计.csv'), index=False, encoding='utf-8-sig')
        # 剔除清单逐年成员
        qp = os.path.join(root, '%s_剔除清单.csv' % cls)
        if os.path.exists(qp):
            q = pd.read_csv(qp, encoding='utf-8-sig', low_memory=False)
            if 'point_id' in q.columns:
                qn = pd.to_numeric(q.point_id, errors='coerce')
                ok = qn.notna()
                qn_i = qn.fillna(-1).astype(np.int64).to_numpy()
                for y in YEARS:
                    q['in_%d' % y] = np.where(ok, np.isin(qn_i, mem[y]), False)
                q.to_csv(os.path.join(root, '%s_剔除清单_逐年成员.csv' % cls), index=False,
                         encoding='utf-8-sig')
        rep[cls] = dict(n_all=n_all, in23=in23, years={str(s['year']): s['n_in_folder'] for s in stats})
        emit('%s 完成（%d 点；已写 %d 年）' % (cls, n_all, len(want_years)))
    out = os.path.join(DEST, '_逐年落盘记录_s6.json')
    old = json.load(open(out, encoding='utf-8')) if os.path.exists(out) else {}
    old.update(rep)
    json.dump(old, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False)[:500])


if __name__ == '__main__':
    main()
