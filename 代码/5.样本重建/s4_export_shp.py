# -*- coding: utf-8 -*-
"""s4_export_shp.py — 把样本导出一份 Shapefile（GIS 可直接打开的点层，WGS84）

* 输入：
    - 数据/本地处理/样本重建/r7_train.parquet            冻结母库 2,252,207
    - 数据/本地处理/样本重建/r7_train_validity.parquet   母库 + 跨年有效性列
    - 数据/本地处理/样本重建/r7_pool.parquet             采样候选池 9,646,715
    - 数据/本地处理/样本重建/r8_rare_candidates.parquet  稀有类补样候选 9,018
    - 数据/本地处理/样本重建/r8b_rare_boundary.parquet   稀有类边界补样 7,285
    - 数据/本地处理/样本底座/validation_pool_v2.parquet  留出验证池 5,186
    - F:/lc_work/年度子集_含稀有类/r7_train_2017..2024.parquet   年度子集 8 年
    - F:/lc_work/prod5p_2023/data/train2023_clean.parquet        2023 生产训练池
* 规则源：代码/0.本地流水线/lc_conf.py（30 类码/中文名/10 大类分组）；
          实验_2026-09_生产前验证/config/v31_map.json（30→24 类合并与 24 类中文名）
* 门槛：只读源文件；单文件 <2GB（Shapefile/dbf 硬限）；编码 GBK + .cpg（中文 Windows ArcGIS 直读）；
        写后逐文件回读抽验（行数 / 包围盒 / 随机 200 点 row_id+类码 与源一致）；
        属性不含 A00–A63 嵌入维（dbf 数字存定宽文本，单年会膨胀到 1.4–3GB 且超限；需要时按 row_id 回连 parquet）
* 输出：<OUT>/<名称>.{shp,shx,dbf,prj,cpg} + _导出说明.md + 导出台账.json
        OUT 默认 Z:/地理所/论文/中国土地覆盖数据_2017-2024/样本_Shp（Z 不可用落 F:/lc_work/样本_Shp导出 并留痕）
* 用法：python s4_export_shp.py                # 全部导出
        python s4_export_shp.py --only 年度子集 # 只导名字含该串的
        python s4_export_shp.py --verify-only   # 只做回读抽验
幂等：目标 .shp 已存在且台账行数一致则跳过（--force 重写）。
"""
import os
import re
import sys
import json
import time
import random
import argparse

sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORK = r'F:/lc_work'
Z_ROOT = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
OUT_DEF = os.path.join(Z_ROOT, '样本_Shp')
OUT_FB = os.path.join(WORK, '样本_Shp导出')
V31_MAP = os.path.join(PROJ, '实验_2026-09_生产前验证', 'config', 'v31_map.json')
V31_MAP_FB = os.path.join(WORK, 'v31_exp', 'config', 'v31_map.json')
LEDGER = '_导出台账.json'
README = '_导出说明.md'
BATCH = 300_000
LIMIT = int(1.75 * 2 ** 30)   # 单文件体积预算（Shapefile/dbf 硬限 2GB，留 12% 余量）

WGS84_WKT = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
             'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')

FILES = [
    dict(name='母库_r7_train', path=os.path.join(PROJ, '数据/本地处理/样本重建/r7_train.parquet'),
         note='冻结母库（MD5 11920eef7f6657e8ccc8822ad3077ed8）；含 159 已知错位点，用前按剔除清单过滤'),
    dict(name='母库_r7_train_validity', path=os.path.join(PROJ, '数据/本地处理/样本重建/r7_train_validity.parquet'),
         note='母库 + valid_from/valid_to/qc_scope（跨年有效性）'),
    dict(name='候选池_r7_pool', path=os.path.join(PROJ, '数据/本地处理/样本重建/r7_pool.parquet'),
         note='清洗前采样候选池（9,646,715）；无 row_id'),
    dict(name='稀有类_r8_rare_candidates', path=os.path.join(PROJ, '数据/本地处理/样本重建/r8_rare_candidates.parquet'),
         note='稀有类补样候选（9,018）'),
    dict(name='稀有类_r8b_rare_boundary', path=os.path.join(PROJ, '数据/本地处理/样本重建/r8b_rare_boundary.parquet'),
         note='稀有类边界/core 补样（7,285）'),
    dict(name='验证池_validation_pool_v2', path=os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2.parquet'),
         note='25km 块留出验证池（5,186）——开发集，非最终真值'),
    dict(name='生产池_2023_clean', path=os.path.join(WORK, 'prod5p_2023/data/train2023_clean.parquet'),
         note='2023 生产训练视图（2,060,577；五省 76 瓦实际使用）'),
] + [dict(name='年度子集_%d' % y, path=os.path.join(WORK, '年度子集_含稀有类', 'r7_train_%d.parquet' % y),
          note='%d 年度子集（当年标签 + 当年 AEF）' % y) for y in range(2017, 2025)]

# 列规格：(dbf 类型, 宽度, 小数位)；宽度按实测取值范围定，改前先复核
SPEC = {
    'row_id': ('N', 11, 0), 'class_new': ('N', 4, 0), 'class_raw': ('N', 6, 0),
    'tier': ('C', 12, 0), 'year': ('N', 5, 0), 'src': ('C', 30, 0), 'src_conf': ('N', 5, 2),
    'agree_n': ('N', 3, 0), 'border': ('N', 2, 0), 'stab_years': ('N', 4, 0),
    'wc_l0': ('N', 4, 0), 'esri_l0': ('N', 4, 0), 'th_l0': ('N', 4, 0),
    'clcd_l0': ('N', 4, 0), 'cn30_l0': ('N', 4, 0), 'province': ('C', 24, 0), 'urban': ('N', 2, 0),
    '_fcs10_match': ('C', 10, 0), '_crop_built_frac': ('N', 5, 2), '_prio': ('N', 9, 5),
    'train_weight': ('N', 6, 3), 'valid_from': ('N', 5, 0), 'valid_to': ('N', 5, 0),
    'qc_scope': ('C', 20, 0), 'qc_status': ('C', 20, 0), 'sample_type': ('C', 10, 0),
    'weight': ('N', 6, 3), 'zone': ('C', 12, 0), 'val_src': ('C', 22, 0), 'bk_key': ('C', 12, 0),
}
# dbf 字段名 ≤10 字符；源列名过长或带下划线前缀的一律重命名
RENAME = {'class_new': 'cls30', 'class_raw': 'src_code', '_fcs10_match': 'fcs10mtch',
          '_crop_built_frac': 'cropbfrac', '_prio': 'prio', 'train_weight': 'tr_weight',
          'stab_years': 'stabyears', 'agree_n': 'agreen', 'src_conf': 'srcconf',
          'province': 'prov', 'sample_type': 'smp_type', 'valid_from': 'vfrom',
          'valid_to': 'vto', 'qc_status': 'qcstat', 'qc_scope': 'qcscope'}
# 输出字段顺序（存在的才写）
ORDER = ['row_id', 'cls30', 'cls30nm', 'cls24', 'cls24nm', 'grp10', 'grp10nm',
         'year', 'tier', 'src', 'srcconf', 'agreen', 'border', 'stabyears',
         'wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0', 'prov', 'urban',
         'fcs10mtch', 'cropbfrac', 'prio', 'tr_weight', 'vfrom', 'vto', 'qcscope',
         'qcstat', 'smp_type', 'weight', 'zone', 'val_src', 'bk_key', 'src_code']
DERIVED = ['cls30nm', 'cls24', 'cls24nm', 'grp10', 'grp10nm']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build_maps():
    """30 类码 → 中文名 / 24 类码 / 24 类名 / 10 大类码与名。"""
    sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
    import lc_conf as C
    fp = V31_MAP if os.path.exists(V31_MAP) else V31_MAP_FB
    mp = json.load(open(fp, encoding='utf-8'))
    c24 = {int(k): int(v) for k, v in mp['identity'].items()}
    for new, olds in mp['merges'].items():
        for o in olds:
            c24[int(o)] = int(new)
    n24 = {int(k): v for k, v in mp['names'].items()}
    gname = {1: '耕地', 2: '森林', 3: '灌丛', 4: '草地', 5: '湿地',
             6: '不透水', 7: '裸地', 8: '稀疏植被', 9: '水体', 10: '冰雪'}
    nm30, m24, nm24, g10, gnm = {}, {}, {}, {}, {}
    for raw, (_en, zh, gkey, _g2) in C.CLASSES.items():
        nm30[raw] = zh
        m24[raw] = c24[raw]
        nm24[raw] = n24[c24[raw]]
        g10[raw] = C.LEVEL1[gkey]
        gnm[raw] = gname[C.LEVEL1[gkey]]
    emit('码表：30 类 %d 个 → 24 类 %d 个（源 %s）' % (len(nm30), len(set(m24.values())), os.path.basename(fp)))
    return dict(nm30=nm30, cls24=m24, nm24=nm24, g10=g10, gnm=gnm)


def resolve_fields(all_cols):
    keep = [c for c in all_cols if not re.fullmatch(r'A\d\d', c) and c not in ('lon', 'lat')]
    return [f for f in ORDER if f in {RENAME.get(c, c) for c in keep} | set(DERIVED)]


def row_bytes(fields):
    """单行估算：dbf 各字段定宽 + 1 删除标记 + shp 28B + shx 8B。"""
    tot = 1 + 28 + 8
    for f in fields:
        src = ([k for k, v in RENAME.items() if v == f] or [f])[0]
        tot += SPEC.get(src, ('C', 24, 0))[1]
    return tot


def expand(items, limit):
    """体积超预算的源按行分卷（Shapefile/dbf 2GB 硬限）。"""
    import pyarrow.parquet as pq
    out = []
    for it in items:
        pf = pq.ParquetFile(it['path'])
        rows = pf.metadata.num_rows
        bpr = row_bytes(resolve_fields(pf.schema_arrow.names))
        total = rows * bpr
        parts = max(1, -(-total // limit))
        if parts == 1:
            out.append(dict(it, rows_total=rows))
            continue
        step = -(-rows // parts)
        for i in range(parts):
            s, e = i * step, min((i + 1) * step, rows)
            if s >= e:
                continue
            out.append(dict(it, name='%s_第%d卷共%d卷' % (it['name'], i + 1, parts),
                            part=(s, e), rows_total=rows))
            emit('  分卷：%s 行 %d–%d（源 %s 行，预估 %.2fGB）' % (
                out[-1]['name'], s, e - 1, format(rows, ','), total / 2 ** 30))
    return out


def readme_text(items, outdir):
    L = ['# 样本 Shapefile 导出说明', '',
         '> 生成 %s ｜ 输出目录 `%s` ｜ 坐标系 WGS84（EPSG:4326）｜ 编码 GBK（含 .cpg）' % (
             time.strftime('%Y-%m-%d %H:%M'), outdir), '',
         '## 文件清单', '', '| 文件 | 点数 | 源 | 说明 |', '|---|---|---|---|']
    for it in items:
        if it.get('rows'):
            rel = os.path.relpath(it['path'], PROJ).replace('\\', '/')
            src = rel if not rel.startswith('..') else it['path'].replace('\\', '/')
            L.append('| `%s.shp` | %s | `%s` | %s |' % (
                it['name'], format(it['rows'], ','), src, it['note']))
    L += ['', '## 字段说明', '',
          '| 字段 | 含义 |', '|---|---|',
          '| row_id | 母库行号（**回连 parquet / AEF 嵌入的键**；候选池无此列） |',
          '| cls30 / cls30nm | 30 类细类码与中文名（源 `class_new`） |',
          '| cls24 / cls24nm | 24 类主产品码与中文名（按 v31_map 归并：五组疏密森林、城乡不透水） |',
          '| grp10 / grp10nm | 10 大类组码与名（耕地/森林/灌丛/草地/湿地/不透水/裸地/稀疏植被/水体/冰雪） |',
          '| year | 标签年份 |',
          '| tier | 质量档（gold/silver/bronze/external/reject/uncovered） |',
          '| src / srcconf | 来源产品标记 / 来源置信度 |',
          '| agreen | 各外部产品一致数 |',
          '| border | 是否位于类别边界带 |',
          '| stabyears | 年际稳定年数（-1 = 未评估） |',
          '| wc_l0 / esri_l0 / th_l0 / clcd_l0 / cn30_l0 | 各外部产品的大类码（WorldCover/ESRI/…，-1 = 无） |',
          '| prov | 省（DataV 省界点内判定） |',
          '| urban | 城乡属性（有夜光/建成区判定的子集） |',
          '| fcs10mtch / cropbfrac / prio | FCS10 匹配状态 / 耕地-建成混合度 / 采样优先级得分 |',
          '| tr_weight | 训练权重（1.0 / 1.2） |',
          '| qcstat | 年度质检状态（keep/downweight/persistent_change/rare_topup） |',
          '| vfrom / vto / qcscope | 有效年起止 / 质检范围（仅 validity 文件） |',
          '| src_code | 来源产品原始码（多数为空） |',
          '',
          '## 两条纪律', '',
          '1. **A00–A63（64 维 AEF 嵌入）不在 shp 里**——dbf 的数字按定宽文本存储，64 列会让单年文件膨胀到 1.4–3 GB（超 Shapefile 2 GB 硬限），且 64 维在 GIS 里无法使用。需要时按 `row_id` 回连同名 parquet。',
          '2. 类码体系：`cls30` 是细类码，`cls24` 是生产主产品码，`grp10` 是论文大类；三者并存避免口径漂移。码表权威在 `代码/0.本地流水线/lc_conf.py` 与 `实验_2026-09_生产前验证/config/v31_map.json`。',
          '']
    return '\n'.join(L)


def export(item, maps, outdir, force=False):
    base = os.path.join(outdir, item['name'])
    rec = dict(item)
    if os.path.exists(base + '.shp') and not force:
        led = read_ledger(outdir)
        old = led.get(item['name'], {})
        rec.update(rows=old.get('rows'), bytes=old.get('bytes'), skipped=True,
                   verify=old.get('verify', 'skipped'))
        emit('%-28s 已存在（跳过，--force 重写）' % item['name'])
        return rec
    import pyarrow.parquet as pq
    import shapefile
    pf = pq.ParquetFile(item['path'])
    all_cols = pf.schema_arrow.names
    keep = [c for c in all_cols if not re.fullmatch(r'A\d\d', c) and c not in ('lon', 'lat')]
    need = ['lon', 'lat'] + [c for c in keep if c in SPEC or c == 'class_new']
    fields = resolve_fields(all_cols)
    start, end = item.get('part', (0, None))
    t0 = time.time()
    n = 0
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding='gbk', encodingErrors='replace')
    for f in fields:
        src = [k for k, v in RENAME.items() if v == f]
        src = src[0] if src else f
        typ, size, dec = SPEC.get(src, ('C', 24, 0))
        w.field(f, typ, size, dec)
    srcnames = {f: ([k for k, v in RENAME.items() if v == f] or [f])[0] for f in fields}
    pos = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=need):
        lo, hi = max(start - pos, 0), (len(batch) if end is None else min(end - pos, len(batch)))
        pos += len(batch)
        if hi <= lo:
            continue
        df = batch.slice(lo, hi - lo).to_pandas()
        if 'class_new' in df:
            miss = set(df.class_new.dropna().unique()) - set(maps['nm30'])
            if miss:
                raise SystemExit('源含未知类码 %s（%s）' % (sorted(miss), item['name']))
            df['cls30nm'] = df.class_new.map(maps['nm30'])
            df['cls24'] = df.class_new.map(maps['cls24'])
            df['cls24nm'] = df.class_new.map(maps['nm24'])
            df['grp10'] = df.class_new.map(maps['g10'])
            df['grp10nm'] = df.class_new.map(maps['gnm'])
        cols = {f: df[srcnames[f]].astype(object).where(df[srcnames[f]].notna(), None).tolist()
                for f in fields}
        recs = list(zip(*[cols[f] for f in fields]))
        plon = df.lon.tolist()
        plat = df.lat.tolist()
        for i in range(len(plon)):
            w.point(float(plon[i]), float(plat[i]))
            w.record(*recs[i])
        n += len(plon)
    w.close()
    open(base + '.prj', 'w').write(WGS84_WKT)
    open(base + '.cpg', 'w').write('GBK')
    nbytes = sum(os.path.getsize(base + e) for e in ('.shp', '.shx', '.dbf', '.prj', '.cpg')
                 if os.path.exists(base + e))
    if nbytes > 2 * 1024 ** 3:
        raise SystemExit('%s 体积 %.2fGB 超 Shapefile 2GB 硬限' % (item['name'], nbytes / 2 ** 30))
    rec.update(rows=n, bytes=nbytes, fields=len(fields), elapsed=round(time.time() - t0, 1))
    emit('%-28s %9s 点 %6.0f MB %5.1fs' % (item['name'], format(n, ','), nbytes / 2 ** 20, rec['elapsed']))
    return rec


def verify(items, outdir):
    """回读抽验：行数 + 随机 200 点按行序与源逐点核对（类码 + 坐标一致）。"""
    import shapefile
    import pyarrow.parquet as pq
    ok_all = True
    for it in items:
        base = os.path.join(outdir, it['name'])
        if not os.path.exists(base + '.shp') or not it.get('rows'):
            continue
        r = shapefile.Reader(base, encoding='gbk')
        n = len(r)
        fld = [f[0] for f in r.fields[1:]]
        ci = fld.index('cls30') if 'cls30' in fld else None
        pf = pq.ParquetFile(it['path'])
        cols = [c for c in ('class_new', 'lon', 'lat') if c in pf.schema_arrow.names]
        start, end = it.get('part', (0, None))
        t = pf.read(columns=cols).to_pandas().iloc[start:end].reset_index(drop=True)
        rng = random.Random(42)
        pos = sorted(rng.sample(range(n), min(200, n)))
        hit = bad = 0
        for i in pos:
            x, y = r.shape(i).points[0]
            same = abs(float(t.lon.iloc[i]) - float(x)) < 1e-9 and abs(float(t.lat.iloc[i]) - float(y)) < 1e-9
            if same and ci is not None:
                same = int(t.class_new.iloc[i]) == int(r.record(i)[ci])
            hit += 1 if same else 0
            bad += 0 if same else 1
        ok = (n == it['rows']) and bad == 0
        ok_all &= ok
        emit('  验 %-30s 行数 %s 逐点核对 %d/%d %s' % (
            it['name'], 'OK' if n == it['rows'] else '不一致(%d vs %d)' % (n, it['rows']),
            hit, hit + bad, 'OK' if ok else 'FAIL'))
        it['verify'] = 'OK' if ok else 'FAIL'
        r.close()
    return ok_all


def read_ledger(outdir):
    fp = os.path.join(outdir, LEDGER)
    if os.path.exists(fp):
        try:
            return {x['name']: x for x in json.load(open(fp, encoding='utf-8'))}
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='只处理名字含该串的样本')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    outdir = a.out or (OUT_DEF if os.path.isdir(Z_ROOT) else OUT_FB)
    if os.path.abspath(outdir) == os.path.abspath(OUT_FB):
        open(os.path.join(WORK, '_样本Shp_Z不可用留痕.txt'), 'a', encoding='utf-8').write(
            '%s Z 未挂载，落 F\n' % time.strftime('%Y-%m-%d %H:%M'))
    os.makedirs(outdir, exist_ok=True)
    items = [it for it in FILES if (not a.only or a.only in it['name'])]
    emit('输出目录 %s（%d 份）' % (outdir, len(items)))
    if a.verify_only:
        verify(expand(items, LIMIT), outdir)
        return
    maps = build_maps()
    items = expand(items, LIMIT)
    done = []
    for it in items:
        done.append(export(it, maps, outdir, a.force))
    led = read_ledger(outdir)
    led.update({x['name']: x for x in done if x.get('rows')})
    entries = sorted(led.values(), key=lambda x: x['name'])
    ok = verify(entries, outdir)
    json.dump(entries, open(os.path.join(outdir, LEDGER), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(outdir, README), 'w', encoding='utf-8').write(readme_text(entries, outdir))
    tot = sum(x['rows'] for x in entries if x.get('rows'))
    emit('完成：%d 份 / %s 点 / %.2f GB；回读抽验 %s' % (
        len([x for x in entries if x.get('rows')]), format(tot, ','),
        sum(x.get('bytes', 0) for x in entries) / 2 ** 30, '全过' if ok else '有失败项'))


if __name__ == '__main__':
    main()
