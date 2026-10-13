# -*- coding: utf-8 -*-
"""s7_repair_dbf.py — 修复 s3/s4 引入的 DBF 小数位丢失（全 F 字段被截为整数）

* 根因：s3_add_ctamap_prov.py / s4_ctamap_nearest.py 重写 DBF 时用了 (name,type,size) 三元组，
  丢掉第 4 位 decimal → pyshp 按整数写 → 所有 F 字段（lon/lat、灌木 ndvi_amp/amp_rel/thr_y/dd、
  沼泽 max300 等）被截断。
* 修复：按族显式声明正确 schema（含小数位），逐文件重写 .shp/.shx/.dbf：
    - lon/lat ← 几何坐标（精确回填）
    - 灌木 ndvi_amp/amp_rel/thr_y/dd ← data/shrub_yearly/y<年>_shrub.parquet（合幅按 pid+year）
    - 沼泽 max300 ← data/swamp_qc/out/swamp_samples_v2.parquet
* 校验：点数一致；.shp 字节 MD5 与修复前一致（几何零改动）；字段 decimal 正确；
  抽 20 点 lon/lat == 几何；抽 20 点回补值 == 源（|Δ|<1e-6 / 四舍五入 3 位相等）。
* 用法：python s7_repair_dbf.py --group land|forest|orchard|shrub|swamp   （可逗号并列）
"""
import argparse, glob, hashlib, os, shutil, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
SH = r'F:/lc_work/v31_exp/data'
SENT_F = -999.0
LANDS = ['草本旱地', '灌溉耕地', '草地', '地衣苔藓', '稀疏植被', '裸地', '湖河滩地', '盐渍湿地',
         '红树林', '海岸盐沼', '潮滩', '水体', '不透水面', '永久冰雪']
CT = [('ct_prov', 'C', 30), ('ct_code', 'C', 8), ('ct_km', 'N', 9, 2)]
SCHEMA = {
    'land': [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('cls24', 'N', 4),
             ('cls_zh', 'C', 18), ('tier', 'C', 8), ('src', 'C', 24)] + CT,
    'forest': [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
               ('cls4', 'N', 2), ('clsname', 'C', 18), ('layer', 'C', 6), ('tier', 'C', 8),
               ('sug', 'N', 3), ('src', 'C', 24)] + CT,
    'swamp': [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
              ('clsw', 'N', 2), ('src', 'C', 24), ('tsrc', 'C', 14), ('flood_m', 'N', 3),
              ('max300', 'F', 8, 3)] + CT,
    'orchard': [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('set', 'C', 6),
                ('crop', 'C', 12), ('form', 'C', 12), ('prov', 'C', 30), ('tier', 'C', 8),
                ('src', 'C', 24)] + CT,
    'orchard_v6': [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('prov', 'C', 30),
                   ('grade', 'C', 12), ('subtype', 'C', 12), ('label', 'C', 12)] + CT,
    'shrub': [('point_id', 'C', 20), ('pcode', 'C', 3), ('prov', 'C', 30), ('tier', 'C', 8),
              ('src', 'C', 24), ('year', 'N', 4), ('type_y', 'C', 6), ('conf_y', 'C', 10),
              ('stage', 'C', 14), ('agr23', 'N', 2), ('c24', 'N', 4), ('c30', 'N', 4),
              ('ndvi_amp', 'F', 13, 3), ('amp_rel', 'F', 13, 3), ('thr_y', 'F', 13, 3),
              ('dd', 'F', 13, 3), ('aef_ok', 'N', 2), ('ndvims', 'N', 2), ('eco', 'C', 12)] + CT,
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def files_of(group):
    if group == 'land':
        return [s for c in LANDS for s in sorted(glob.glob(os.path.join(DEST, c, '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))]
    if group == 'forest':
        return sorted(glob.glob(os.path.join(DEST, '林地_*', '*并集_WGS84.shp')))
    if group == 'orchard':
        fs = sorted(glob.glob(os.path.join(DEST, '乔灌园地', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))
        return fs
    if group == 'shrub':
        return sorted(glob.glob(os.path.join(DEST, '灌木', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp'))) + \
            sorted(glob.glob(os.path.join(DEST, '灌木', '*合幅_WGS84.shp')))
    if group == 'swamp':
        return sorted(glob.glob(os.path.join(DEST, '木*沼泽', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp'))) + \
            sorted(glob.glob(os.path.join(DEST, '草*沼泽', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))
    raise SystemExit('未知组 %s' % group)


def family_of(f, group):
    if group == 'orchard':
        return 'orchard_v6' if 'v6' in os.path.basename(f) else 'orchard'
    return group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--classes', default='', help='land 组内仅处理这些类（逗号分隔）')
    ap.add_argument('--log', default=os.path.join(DEST, '_DBF小数位修复记录_s7.json'))
    a = ap.parse_args()
    only = [x.strip() for x in a.classes.split(',') if x.strip()]
    shrub_cache, swamp_max = {}, None
    log = []
    for junk in glob.glob(os.path.join(DEST, '**', '*_tmp.shp'), recursive=True):
        for ext in ('.shp', '.shx', '.dbf'):
            try:
                os.remove(junk[:-4] + ext)
            except OSError:
                pass
    for grp in a.group.split(','):
        grp = grp.strip()
        fam = family_of('', grp)
        fs = files_of(grp)
        if only and grp == 'land':
            fs = [f for f in fs if os.path.basename(os.path.dirname(os.path.dirname(f))) in only]
        emit('%s：%d 个文件' % (grp, len(fs)))
        for f in fs:
            fam = family_of(f, grp)
            schema = SCHEMA[fam]
            names = [x[0] for x in schema]
            r = shapefile.Reader(f, encoding='utf-8')
            cur = [x[0] for x in r.fields[1:]]
            if [c for c in cur] != names:
                emit('  ✗ 字段不匹配，跳过：%s\n    现 %s' % (os.path.relpath(f, DEST), cur))
                r.close(); continue
            dec_now = {x[0]: x[3] for x in r.fields[1:]}
            want_dec = {x[0]: (x[3] if len(x) > 3 else 0) for x in schema}
            if all(dec_now[k] == want_dec[k] for k in dec_now):
                emit('  已正确，跳过 %s' % os.path.relpath(f, DEST)); r.close(); continue
            recs = [list(x) for x in r.iterRecords()]
            pts = [p.points[0] for p in r.iterShapes()]
            r.close()
            n = len(recs)
            md5_before = hashlib.md5(open(f, 'rb').read()).hexdigest()
            # —— 回补值 ——
            if 'lon' in names:
                il, it = names.index('lon'), names.index('lat')
                for i in range(n):
                    recs[i][il] = round(float(pts[i][0]), 6)
                    recs[i][it] = round(float(pts[i][1]), 6)
            if fam == 'shrub':
                pid = np.array([str(x[0]) for x in recs], dtype=object)
                yrs = np.array([int(x[names.index('year')]) for x in recs], np.int32)
                for y in np.unique(yrs):
                    if y not in shrub_cache:
                        d = pd.read_parquet(os.path.join(SH, 'shrub_yearly', 'y%d_shrub.parquet' % y),
                                            columns=['point_id', 'ndvi_amp_y', 'amp_rel_y', 'thr_y', 'delta_d_y'])
                        d.index = d.point_id.astype(str)
                        shrub_cache[y] = d
                    d = shrub_cache[y]
                    m = yrs == y
                    j = [names.index(c) for c in ('ndvi_amp', 'amp_rel', 'thr_y', 'dd')]
                    for c_src, jj in zip(('ndvi_amp_y', 'amp_rel_y', 'thr_y', 'delta_d_y'), j):
                        vals = d[c_src].reindex(pid[m]).to_numpy(float)
                        vals = np.where(np.isfinite(vals), np.round(vals, 3), SENT_F)
                        for k, i in enumerate(np.where(m)[0]):
                            recs[i][jj] = float(vals[k])
            if fam == 'swamp':
                if swamp_max is None:
                    d = pd.read_parquet(os.path.join(SH, 'swamp_qc', 'out', 'swamp_samples_v2.parquet'),
                                        columns=['point_id', 'max_300'])
                    swamp_max = d.set_index(d.point_id.astype(str)).max_300
                j = names.index('max300')
                pid = [str(x[0]) for x in recs]
                vals = swamp_max.reindex(pid).to_numpy(float)
                for k in range(n):
                    recs[k][j] = float(round(vals[k], 3)) if np.isfinite(vals[k]) else SENT_F
            # —— 重写（tmp 校验后替换；删除锁回退） ——
            base = f[:-4]
            w = shapefile.Writer(base + '_tmp', shapeType=shapefile.POINT, encoding='utf-8')
            for (nm, tp, sz, *rest) in schema:
                w.field(nm, tp, sz, rest[0] if rest else 0)
            for i in range(n):
                w.point(float(pts[i][0]), float(pts[i][1])); w.record(*recs[i])
            w.close()
            md5_after = hashlib.md5(open(base + '_tmp.shp', 'rb').read()).hexdigest()
            rr = shapefile.Reader(base + '_tmp.shp', encoding='utf-8')
            dec_ok = all(x[3] == want_dec[x[0]] for x in rr.fields[1:])
            m = np.random.RandomState(7).permutation(n)[:20]
            geo_ok = True
            if 'lon' in names:
                il, it = names.index('lon'), names.index('lat')
                geo_ok = all(abs(rr.shape(i).points[0][0] - recs[i][il]) < 1e-6
                             and abs(rr.shape(i).points[0][1] - recs[i][it]) < 1e-6 for i in m)
            val_ok = True
            if fam == 'shrub':
                j = names.index('ndvi_amp')
                for i in m:
                    src = shrub_cache[int(recs[i][names.index('year')])]['ndvi_amp_y'].get(str(recs[i][0]), np.nan)
                    want = round(float(src), 3) if np.isfinite(src) else SENT_F
                    if abs(float(recs[i][j]) - want) > 1e-6:
                        val_ok = False
            if fam == 'swamp':
                j = names.index('max300')
                for i in m:
                    src = swamp_max.get(str(recs[i][0]), np.nan)
                    want = round(float(src), 3) if np.isfinite(src) else SENT_F
                    if abs(float(recs[i][j]) - want) > 1e-6:
                        val_ok = False
            ok = (len(rr) == n) and dec_ok and geo_ok and val_ok and (md5_after == md5_before)
            rr.close()
            if not ok:
                for ext in ('.shp', '.shx', '.dbf'):
                    try:
                        os.remove(base + '_tmp' + ext)
                    except OSError:
                        pass
                raise SystemExit('校验失败 %s（n=%d dec=%s geo=%s val=%s shpMD5同=%s）'
                                 % (f, n, dec_ok, geo_ok, val_ok, md5_after == md5_before))
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
                        else:
                            time.sleep(2.0 * (att + 1))
            log.append(dict(group=fam, shp=os.path.relpath(f, DEST), n=n,
                            decimals_ok=dec_ok, shp_md5_unchanged=(md5_after == md5_before)))
            emit('  ✓ %s（n=%d；小数位修复；shp 字节不变=%s）' % (os.path.relpath(f, DEST), n,
                                                             md5_after == md5_before))
    out = a.log
    old = []
    if os.path.exists(out):
        import json
        old = json.load(open(out, encoding='utf-8'))
    import json
    json.dump(old + log, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('完成 %d 个文件' % len(log))


if __name__ == '__main__':
    main()
