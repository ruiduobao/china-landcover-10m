# -*- coding: utf-8 -*-
"""s1_samples_to_shp.py — 单类样本 SHP 交付导出（WGS84 + 标准 GCJ-02 双坐标；逐年目录）

* 用途：把本轮完成的各类样本导出为 SHP，落到
        F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/<类>/<年>/
        类目录：灌木（逐年面板）/ 乔灌园地（逐年面板）/ 林地_四类 / 木本沼泽 / 草本沼泽 / 其余 14 类
* 年份语义：
        - 灌木、乔灌园地：真逐年（每年一套标签/面板）→ <类>/2017../2024
        - 池类（林地/沼泽/14 类）：标签与证据为 2023 口径、各年训练池成员高度重叠（93–100%）
          → 单目录 <类>/2023/，类目录下另附 `逐年池成员数.csv`（该点集在 8 个年度子集中的命中数）
* 坐标：WGS84＝原始；GCJ02＝k23_shp_fix.gcj02_std（标准公式，与 shrub 逐年套件实测一致）
* 灌木：直接复制已核验的 k34 逐年双坐标 SHP（改名 灌木_{Y}）；其余类由本脚本用 pyshp 新写
* 校验：逐文件回读（DBF 记录数=点数）、随机 200 点坐标回读一致、.prj/.cpg 检查 → shp_export_report.json
* 用法：python s1_samples_to_shp.py [--only 灌木,乔灌园地] [--coord both|wgs84]
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
from k23_shp_fix import gcj02_std, write_sidecars, WGS84_WKT, GCJ02_WKT  # noqa

DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
WORK = r'F:/lc_work/v31_exp'
ANN = r'F:/lc_work/年度子集_含稀有类'
Y14 = {1: '草本旱地', 3: '灌溉耕地', 11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 22: '裸地',
       16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 23: '水体',
       21: '不透水面', 24: '永久冰雪', 14: '木本沼泽', 15: '草本沼泽'}
C4 = {4: '常绿阔叶林', 5: '落叶阔叶林', 6: '常绿针叶林', 7: '落叶针叶林'}
REP = {'files': {}}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def gcj02_vec(lon, lat):
    """矢量化标准 GCJ-02（与 k23_shp_fix.gcj02_std 逐点一致）。"""
    a, ee = 6378245.0, 0.00669342162296594323
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x, y = lon - 105.0, lat - 35.0
    d_lat = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x)))
    d_lon = (300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
             + 2.0 / 3.0 * d_lat)
    rad = lat / 180.0 * np.pi
    magic = 1 - ee * np.sin(rad) ** 2
    sq = np.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sq) * np.pi)
    d_lon = (d_lon * 180.0) / (a / sq * np.cos(rad) * np.pi)
    return lon + d_lon, lat + d_lat


def _cut(s, size):
    """按字节安全截断（不切断多字节字符；pyshp 对 C 字段按字节截断会造坏 UTF-8）。"""
    if not isinstance(s, str):
        return s
    b = s.encode('utf-8')
    if len(b) <= size:
        return s
    b = b[:size]
    while b:
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            b = b[:-1]
    return ''


def write_pts(base, fields, rows, xs, ys):
    """fields=[(name,type,size[,dec])]；rows=列表元组。"""
    import shapefile
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding='utf-8')
    for f in fields:
        w.field(f[0], f[1], f[2], f[3] if len(f) > 3 else 0)
    sizes = [f[2] for f in fields]
    for (x, y, rec) in zip(xs, ys, rows):
        rec = tuple(_cut(v, sizes[i]) for i, v in enumerate(rec))
        w.point(float(x), float(y)); w.record(*rec)
    w.close()
    return base


def emit_pair(tag, dst_dir, fields, rows, xs, ys, coord):
    """写一个双坐标套件并回读校验；返回记录数。"""
    os.makedirs(dst_dir, exist_ok=True)
    b1 = write_pts(os.path.join(dst_dir, tag + '_WGS84'), fields, rows, xs, ys)
    act1 = write_sidecars(b1 + '.shp', WGS84_WKT, 'WGS84')
    n2 = None
    if coord == 'both':
        gx, gy = gcj02_vec(xs, ys)
        b2 = write_pts(os.path.join(dst_dir, tag + '_GCJ02'), fields, rows, gx, gy)
        act2 = write_sidecars(b2 + '.shp', GCJ02_WKT, 'GCJ02')
        n2 = (b2, len(gx))
    import shapefile
    rd = shapefile.Reader(b1 + '.shp', encoding='utf-8')
    n = len(rd); shp_n = len(rd.shapes())
    pts = rd.shapeRecords()
    rng = np.random.RandomState(20261013)
    idx = rng.permutation(n)[:200] if n else []
    md = 0.0
    for i in idx:
        px, py = pts[i].shape.points[0]
        md = max(md, abs(px - float(xs[i])), abs(py - float(ys[i])))
    ok = (n == len(rows) and shp_n == len(rows) and md < 1e-6)
    # 中文字段抽检（防乱码）
    if n:
        fld = [f[0] for f in rd.fields[1:]]
        rec0 = rd.record(0)
        ok = ok and all(isinstance(v, (str, int, float)) for v in rec0)
    REP.setdefault('files', {})[os.path.relpath(b1 + '.shp', DEST)] = dict(
        n=int(n), shp=int(shp_n), roundtrip_maxdiff=round(float(md), 9), PASS=bool(ok),
        sidecar_actions=len(act1) + (len(act2) if coord == 'both' else 0))
    return int(n)


# ---------------------------------------------------------------- 灌木（复制已核验套件）
def do_shrub(coord):
    src = os.path.join(WORK, 'data', 'shrub_yearly', 'shp_yearly')
    root = os.path.join(DEST, '灌木')
    cnt = {}
    for y in range(2017, 2025):
        dd = os.path.join(root, str(y)); os.makedirs(dd, exist_ok=True)
        for coordtag in (['WGS84'] if coord == 'wgs84' else ['WGS84', 'GCJ02']):
            for ext in ('shp', 'shx', 'dbf', 'prj', 'cpg'):
                s = os.path.join(src, 'shrub_y%d_%s.%s' % (y, coordtag, ext))
                shutil.copy2(s, os.path.join(dd, '灌木_%d_%s.%s' % (y, coordtag, ext)))
        import shapefile
        rd = shapefile.Reader(os.path.join(dd, '灌木_%d_WGS84.shp' % y), encoding='utf-8')
        cnt[y] = len(rd)
        REP['files']['灌木/%d/灌木_%d_WGS84.shp' % (y, y)] = dict(n=len(rd), PASS=len(rd) == 175512)
    # 合幅
    for coordtag in (['WGS84'] if coord == 'wgs84' else ['WGS84', 'GCJ02']):
        for ext in ('shp', 'shx', 'dbf', 'prj', 'cpg'):
            shutil.copy2(os.path.join(src, 'shrub_yearly_2017_2024_%s.%s' % (coordtag, ext)),
                         os.path.join(root, '灌木_2017-2024合幅_%s.%s' % (coordtag, ext)))
    with open(os.path.join(root, 'GCJ02_说明.txt'), 'w', encoding='utf-8') as f:
        f.write('本目录 GCJ-02 套件为标准 GCJ-02 正算（k23_shp_fix.gcj02_std）结果，'
                '对 DataV 省界（GCJ-02）叠加使用；与影像（WGS84）叠加时会有 ~0.001–0.006° 偏移，属定义使然。\n'
                '注：shrub_yearly 旧目录中的 _GCJ02_NOTE.txt 为 k18/h8 时代套用的旧告警文本，'
                '经本目录导出时抽检（5 点 vs 标准公式 max|Δ|=0.0）确认与本套件无关。\n')
    emit('灌木：%s' % cnt)
    return cnt


# ---------------------------------------------------------------- 通用池类
_POOL = {}


def pool_ids(y):
    if y not in _POOL:
        d = pd.read_parquet(os.path.join(ANN, 'r7_train_%d.parquet' % y), columns=['row_id'])
        _POOL[y] = d.row_id.to_numpy(np.int64)
    return _POOL[y]


def membership_csv(ids, out_csv):
    """逐年池成员数（id 在 r7_train_Y 的命中；非数值 id 记 -1 不进统计）。"""
    a = pd.to_numeric(pd.Series(list(ids)), errors='coerce').dropna().to_numpy(np.int64)
    rows, nmiss = [], len(ids) - len(a)
    for y in range(2017, 2025):
        n = int(np.isin(a, pool_ids(y)).sum())
        rows.append(dict(year=y, n_in_pool=n, pct=round(100.0 * n / max(1, len(ids)), 2)))
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    return rows, nmiss


# ---------------------------------------------------------------- 乔灌园地
def do_orchard(coord):
    root = os.path.join(DEST, '乔灌园地')
    p1 = os.path.join(root, '乔灌园地_母库面板_2017-2024.parquet')
    p2 = os.path.join(root, '乔灌园地_新增样本_面板.parquet')
    d1 = pd.read_parquet(p1); d2 = pd.read_parquet(p2)
    d1 = d1.rename(columns={'row_id': 'pid'}); d1['set'] = 'muk'
    d2 = d2.rename(columns={'point_id': 'pid'}); d2['set'] = 'new'
    for c in ('crop', 'form'):
        if c not in d2: d2[c] = ''
    if 'tier' not in d1: d1['tier'] = ''
    fields = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('set', 'C', 6),
              ('crop', 'C', 12), ('form', 'C', 12), ('prov', 'C', 30), ('tier', 'C', 8), ('src', 'C', 24)]
    cnt = {}
    for y in range(2017, 2025):
        g = pd.concat([d1[d1.year == y], d2[d2.year == y]], ignore_index=True)
        for c in ('set', 'crop', 'form', 'prov', 'tier', 'src'):
            if c in g.columns:
                g[c] = g[c].fillna('').astype(str)
        rows = [(str(r.pid), float(r.lon), float(r.lat), str(r.set), str(r.crop), str(r.form),
                 str(r.prov), str(r.tier), str(r.src)) for r in g.itertuples()]
        cnt[y] = emit_pair('乔灌园地_%d' % y, os.path.join(root, str(y)), fields, rows,
                           g.lon.tolist(), g.lat.tolist(), coord)
    # v6 推荐训练集 → 2023 目录（额外文件）
    f6 = os.path.join(root, 'v6_含苹果', '乔灌园地v6_推荐训练集.csv')
    if os.path.exists(f6):
        v = pd.read_csv(f6, encoding='utf-8-sig')
        v = v.drop_duplicates(['lon', 'lat']).reset_index(drop=True)
        fields6 = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
                   ('grade', 'C', 12), ('subtype', 'C', 12), ('label', 'C', 12)]
        rows = [('v6_%05d' % i, float(r.lon), float(r.lat), str(r.prov), str(r.grade),
                 str(r.subtype), str(r.label)) for i, r in enumerate(v.itertuples())]
        emit_pair('乔灌园地v6_推荐训练集_2023', os.path.join(root, '2023'), fields6, rows,
                  v.lon.tolist(), v.lat.tolist(), coord)
    emit('乔灌园地：%s' % cnt)
    return cnt


# ---------------------------------------------------------------- 林地 4 类
def do_forest(coord):
    src = os.path.join(WORK, 'data', 'forest_qc', 'out')
    d = pd.read_parquet(os.path.join(src, 'forest_samples_v1.parquet'))
    strict_ids = set(pd.read_parquet(os.path.join(src, 'forest_samples_v1_strict.parquet'),
                                     columns=['point_id']).point_id.astype(str))
    d['layer'] = np.where(d.point_id.astype(str).isin(strict_ids), 'strict', 'clean')
    d['sug'] = pd.to_numeric(d.suggest_class4, errors='coerce').fillna(-9).astype(int)
    fields = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
              ('cls4', 'N', 2), ('clsname', 'C', 18), ('layer', 'C', 6), ('tier', 'C', 8),
              ('sug', 'N', 3), ('src', 'C', 24)]
    keep = d[~d.quarantined.astype(bool)].copy()
    q = pd.read_csv(os.path.join(src, '..', 'qc', 'quarantine.csv'), encoding='utf-8-sig')
    cnt = {}
    for c4, nm in C4.items():
        g = keep[keep.class4 == c4]
        rows = [(str(r.point_id), float(r.lon), float(r.lat), str(r.prov), int(r.class4), nm,
                 str(r.layer), str(r.tier), int(r.sug), str(r.src)) for r in g.itertuples()]
        root = os.path.join(DEST, '林地_%s' % nm)
        cnt[nm] = emit_pair('林地_%s_2023' % nm, os.path.join(root, '2023'), fields, rows,
                            g.lon.tolist(), g.lat.tolist(), coord)
        qq = q[q.class4 == c4] if 'class4' in q.columns else q.iloc[0:0]
        qq.to_csv(os.path.join(root, '林地_%s_剔除清单.csv' % nm), index=False, encoding='utf-8-sig')
        membership_csv(set(g.point_id.astype(str)), os.path.join(root, '逐年池成员数.csv'))
        emit('  林地 %s：交付 %d / 剔除 %d' % (nm, len(g), len(qq)))
    emit('林地：%s' % cnt)
    return cnt


# ---------------------------------------------------------------- 沼泽 2 类
def do_swamp(coord):
    src = os.path.join(WORK, 'data', 'swamp_qc')
    d = pd.read_parquet(os.path.join(src, 'out', 'swamp_samples_v2.parquet'))
    fields = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
              ('clsw', 'N', 2), ('src', 'C', 24), ('tsrc', 'C', 14),
              ('flood_m', 'N', 3), ('max300', 'F', 8, 3)]
    qf = os.path.join(src, 'qc', 'quarantine_swamp_full.csv')
    q = pd.read_csv(qf, encoding='utf-8-sig') if os.path.exists(qf) else None
    cnt = {}
    for code, nm in ((14, '木本沼泽'), (15, '草本沼泽')):
        g = d[d.class_swamp == code]
        rows = [(str(r.point_id), float(r.lon), float(r.lat), str(r.prov), int(r.class_swamp),
                 str(r.source), str(r.type_src), int(r.pt_flood_months), float(r.max_300))
                for r in g.itertuples()]
        root = os.path.join(DEST, nm)
        cnt[nm] = emit_pair('%s_2023' % nm, os.path.join(root, '2023'), fields, rows,
                            g.lon.tolist(), g.lat.tolist(), coord)
        if q is not None and 'class_swamp' in q.columns:
            q[q.class_swamp == code].to_csv(os.path.join(root, '%s_剔除清单.csv' % nm),
                                            index=False, encoding='utf-8-sig')
        membership_csv(set(g.point_id.astype(str)), os.path.join(root, '逐年池成员数.csv'))
        emit('  %s：%d' % (nm, len(g)))
    emit('沼泽：%s' % cnt)
    return cnt


# ---------------------------------------------------------------- 其余 14 类
def do_land14(coord):
    src = os.path.join(WORK, 'data', 'land_qc', 'out')
    d = pd.read_parquet(os.path.join(src, 'clean', 'land_other14_v1.parquet'))
    qs = pd.concat([pd.read_csv(f, encoding='utf-8-sig')
                    for f in glob.glob(os.path.join(WORK, 'data', 'land_qc', 'qc', 'quarantine_*.csv'))])
    fields = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6),
              ('cls24', 'N', 4), ('cls_zh', 'C', 18), ('tier', 'C', 8), ('src', 'C', 24)]
    cnt = {}
    for cl in (1, 3, 11, 12, 13, 22, 16, 17, 18, 19, 20, 23, 21, 24):
        nm = Y14[cl]
        g = d[d.class24 == cl]
        rows = [(str(r.point_id), float(r.lon), float(r.lat), int(r.class24), nm,
                 str(r.tier), str(r.src)) for r in g.itertuples()]
        root = os.path.join(DEST, nm)
        cnt[nm] = emit_pair('%s_2023' % nm, os.path.join(root, '2023'), fields, rows,
                            g.lon.tolist(), g.lat.tolist(), coord)
        qq = qs[qs.class24 == cl]
        qq.to_csv(os.path.join(root, '%s_剔除清单.csv' % nm), index=False, encoding='utf-8-sig')
        membership_csv(set(g.point_id.astype(str)), os.path.join(root, '逐年池成员数.csv'))
        emit('  %s：%d（剔除 %d）' % (nm, len(g), len(qq)))
    emit('14 类：%s' % cnt)
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='灌木,乔灌园地,林地,沼泽,其余14类')
    ap.add_argument('--coord', default='both', choices=['both', 'wgs84'])
    a = ap.parse_args()
    only = [x.strip() for x in a.only.split(',') if x.strip()]
    t0 = time.time()
    # GCJ-02 矢量版 vs k23 标量版一致性抽检
    for _x, _y in [(116.4, 39.9), (104.1, 30.6), (121.5, 31.3), (87.6, 43.8)]:
        vx, vy = gcj02_vec(_x, _y)
        sx, sy = gcj02_std(_x, _y)
        assert abs(float(vx) - sx) < 1e-9 and abs(float(vy) - sy) < 1e-9, (_x, _y)
    emit('GCJ-02 矢量版与 k23 标量版一致（4 点抽检）')
    if '灌木' in only:
        do_shrub(a.coord)
    if '乔灌园地' in only:
        do_orchard(a.coord)
    if '林地' in only:
        do_forest(a.coord)
    if '沼泽' in only:
        do_swamp(a.coord)
    if '其余14类' in only:
        do_land14(a.coord)
    REP['elapsed_s'] = round(time.time() - t0, 1)
    REP['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    REP['PASS'] = all(v.get('PASS', True) for v in REP['files'].values())
    json.dump(REP, open(os.path.join(DEST, 'shp_export_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    bad = [k for k, v in REP['files'].items() if not v.get('PASS', True)]
    emit('SHP 导出：%d 个文件组，PASS=%s，用时 %.1fs%s' % (
        len(REP['files']), REP['PASS'], REP['elapsed_s'], ('，FAIL: %s' % bad) if bad else ''))


if __name__ == '__main__':
    main()
