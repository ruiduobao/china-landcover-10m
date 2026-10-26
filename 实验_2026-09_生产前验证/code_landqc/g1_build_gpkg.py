# -*- coding: utf-8 -*-
"""g1_build_gpkg.py — 交付 SHP → GPKG（内嵌 AEF64，float32 列，关空间索引）并替换原件

* AEF 源（全部本地，按行键精确连接；抽检同源 200/200）：
    land/forest/swamp 逐年 : F:/lc_work/年度子集_含稀有类/r7_train_<年>.parquet（row_id ↔ pid；年匹配 AEF64）
    swamp 2023 全量         : 数字 pid → r7_train_2023；非数字 pid（DS-/RC-/OS-）→ data/swamp_qc/feats/aef/*.parquet
    forest 并集             : data/forest_qc/out/forest_samples_v1.parquet（自带 AEF64，与 r7_2023 同源）
    orchard v7 逐年         : 乔灌园地/乔灌园地v7_逐年面板_2017-2024.parquet（uid ↔ pid；年匹配 AEF64）
    shrub 逐年              : data/shrub_yearly/y<年>_shrub.parquet（point_id；年匹配 AEF64）
    shrub 合幅              : 同上 8 年按 (point_id, year) 连接
* 写法：geopandas+pyogrio，AEF 存 64 列 float32（GPKG 声明 FLOAT），SPATIAL_INDEX=NO；坐标 EPSG:4326
* 校验：回读 GPKG（行数 / AEF 抽 30 点与源零差 / 几何抽 30 点零差）→ 通过后删除该 SHP 五件套
* 台账：<交付根>/_SHP转GPKG记录.json（逐文件 n / shp字节 / gpkg字节 / AEF缺失 / 键缺失 / 秒）
* 用法：python g1_build_gpkg.py --group land --classes 草地,裸地   ｜ --group shrub --subdir 常绿灌木
        ｜ --group orchard ｜ --group forest ｜ --group swamp
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
ANN = r'F:/lc_work/年度子集_含稀有类'
AEFC = ['A%02d' % k for k in range(64)]
AEF_INFO = os.path.join(ROOT, '_SHP转GPKG记录.json')
_yearly_cache = {}
_shrub_cache = {}
_swamp_cache = None
_orchard_cache = {}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def aef_from_subset(year, ids):
    """r7_train_<year> 的 AEF64（按 row_id 字符串键）；缓存最近 2 年。"""
    if year not in _yearly_cache:
        d = pd.read_parquet(os.path.join(ANN, 'r7_train_%d.parquet' % year), columns=['row_id'] + AEFC)
        d['k'] = d.row_id.astype(str)
        for c in AEFC:
            d[c] = d[c].astype('float32')
        _yearly_cache.clear() if len(_yearly_cache) >= 2 else None
        _yearly_cache[year] = d.set_index('k')[AEFC]
    return _yearly_cache[year].reindex(ids)


def aef_shrub(year):
    if year not in _shrub_cache:
        d = pd.read_parquet(os.path.join(WORK, 'data', 'shrub_yearly', 'y%d_shrub.parquet' % year),
                            columns=['point_id'] + AEFC)
        d['k'] = d.point_id.astype(str)
        for c in AEFC:
            d[c] = d[c].astype('float32')
        _shrub_cache[year] = d.set_index('k')[AEFC]
    return _shrub_cache[year]


def aef_orchard(year):
    if year not in _orchard_cache:
        d = pd.read_parquet(os.path.join(ROOT, '乔灌园地', '乔灌园地v7_逐年面板_2017-2024.parquet'),
                            columns=['uid', 'year'] + AEFC)
        d = d[d.year == year]
        d['k'] = d.uid.astype(str)
        for c in AEFC:
            d[c] = d[c].astype('float32')
        _orchard_cache[year] = d.set_index('k')[AEFC]
    return _orchard_cache[year]


def aef_swamp_blocks():
    global _swamp_cache
    if _swamp_cache is None:
        fs = sorted(glob.glob(os.path.join(WORK, 'data', 'swamp_qc', 'feats', 'aef', '*.parquet')))
        d = pd.concat([pd.read_parquet(f, columns=['point_id'] + AEFC) for f in fs], ignore_index=True)
        d['k'] = d.point_id.astype(str)
        for c in AEFC:
            d[c] = d[c].astype('float32')
        _swamp_cache = d.set_index('k')[AEFC]
    return _swamp_cache


def files_of(group, subdir, classes):
    if group == 'land':
        return [f for c in classes for f in sorted(glob.glob(os.path.join(ROOT, c, '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))]
    if group == 'forest':
        return sorted(glob.glob(os.path.join(ROOT, '林地_*', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp'))) + \
            sorted(glob.glob(os.path.join(ROOT, '林地_*', '*并集_WGS84.shp')))
    if group == 'swamp':
        return sorted(glob.glob(os.path.join(ROOT, '*沼泽', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))
    if group == 'orchard':
        return sorted(glob.glob(os.path.join(ROOT, '乔灌园地', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp')))
    if group == 'mangrove':
        return sorted(glob.glob(os.path.join(ROOT, '红树林', '[0-9][0-9][0-9][0-9]', '*_WGS84.shp'))) +             [os.path.join(ROOT, '红树林', '扩充_v1', '红树林_全量_10718_2023_WGS84.shp'),
             os.path.join(ROOT, '红树林', '扩充_v1', '红树林_新增_10066_2023_WGS84.shp')]
    if group == 'shrub':
        base = os.path.join(ROOT, subdir)   # 2026-10-13 晚：常绿灌木/落叶灌木 已上提交付根目录
        return sorted(glob.glob(os.path.join(base, '[0-9][0-9][0-9][0-9]', '*_WGS84.shp'))) + \
            sorted(glob.glob(os.path.join(base, '*合幅_WGS84.shp')))
    raise SystemExit('未知组')


def attach_aef(group, path, df):
    """返回 (df, aef_missing, key_missing)。"""
    y = None
    for part in path.replace('/', os.sep).split(os.sep):
        if len(part) == 4 and part.isdigit():
            y = int(part)
    if group == 'land' or (group == 'forest' and y is not None):
        pid = df['pid'].astype(str)
        oknum = pd.to_numeric(pid, errors='coerce').notna()
        a = aef_from_subset(y if y else 2023, pid)
        r = a.reset_index(drop=True)
        key_missing = int(r[AEFC[0]].isna().sum() - 0)
        for c in AEFC:
            df[c] = r[c].to_numpy()
        return df, int(df[AEFC[0]].isna().sum()), key_missing
    if group == 'forest':   # 并集（自带 AEF）
        src = pd.read_parquet(os.path.join(WORK, 'data', 'forest_qc', 'out', 'forest_samples_v1.parquet'),
                              columns=['point_id'] + AEFC)
        src['k'] = src.point_id.astype(str)
        for c in AEFC:
            src[c] = src[c].astype('float32')
        r = src.set_index('k')[AEFC].reindex(df['pid'].astype(str)).reset_index(drop=True)
        for c in AEFC:
            df[c] = r[c].to_numpy()
        return df, int(df[AEFC[0]].isna().sum()), 0
    if group == 'swamp':    # 数字 pid → 该年年度子集；非数字（DS-/RC-/OS-）→ 沼泽 AEF 块
        pid = df['pid'].astype(str)
        num = pd.to_numeric(pid, errors='coerce')
        ok = num.notna()
        for c in AEFC:
            df[c] = np.nan
        if ok.any():
            a = aef_from_subset(y if y else 2023, pid[ok]).reset_index(drop=True)
            need = a[AEFC[0]].isna().to_numpy()      # 年度子集未命中 → 回退沼泽 AEF 块（如 2023 的 1,188+5,946 点）
            if need.any():
                bidx = np.where(ok)[0][need]
                bb = aef_swamp_blocks().reindex(pid.iloc[bidx]).reset_index(drop=True)
                a.loc[need, AEFC] = bb[AEFC].to_numpy()
            for c in AEFC:
                col = df[c].to_numpy(); col[np.where(ok)[0]] = a[c].to_numpy(); df[c] = col
        if (~ok).any():
            b = aef_swamp_blocks().reindex(pid[~ok]).reset_index(drop=True)
            for c in AEFC:
                col = df[c].to_numpy(); col[np.where(~ok)[0]] = b[c].to_numpy(); df[c] = col
        return df, int(df[AEFC[0]].isna().sum()), 0
    if group == 'mangrove':
        idcol = 'cid' if 'cid' in df.columns else 'pid'
        yr = y if idcol == 'pid' else 2023
        cid = df[idcol].astype(str)
        for c in AEFC:
            df[c] = np.nan
        num = pd.to_numeric(cid, errors='coerce')
        ok = num.notna()
        if ok.any():
            a = aef_from_subset(yr if yr else 2023, cid[ok]).reset_index(drop=True)
            for c in AEFC:
                col = df[c].to_numpy(); col[np.where(ok)[0]] = a[c].to_numpy(); df[c] = col
        if (~ok).any():
            fs = sorted(glob.glob(os.path.join(WORK, 'data', 'mangrove_aef', 'aef_p*.parquet')))
            blk = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
            if len(blk):
                blk['k'] = blk.point_id.astype(str)
                b = blk.set_index('k')[AEFC].reindex(cid[~ok]).reset_index(drop=True)
                for c in AEFC:
                    col = df[c].to_numpy(); col[np.where(~ok)[0]] = b[c].to_numpy(); df[c] = col
        return df, int(df[AEFC[0]].isna().sum()), 0
    if group == 'orchard':
        r = aef_orchard(y).reindex(df['uid'].astype(str)).reset_index(drop=True)
        for c in AEFC:
            df[c] = r[c].to_numpy()
        return df, int(df[AEFC[0]].isna().sum()), 0
    if group == 'shrub':
        pid = df['point_id'].astype(str)
        if '合幅' in os.path.basename(path):
            key = pid + '|' + df['year'].astype(str)
            frames = []
            for yy in range(2017, 2025):
                a = aef_shrub(yy).copy()
                a['k'] = a.index + '|%d' % yy
                frames.append(a.set_index('k'))
            allc = pd.concat(frames)
            r = allc.reindex(key).reset_index(drop=True)
        else:
            r = aef_shrub(y).reindex(pid).reset_index(drop=True)
        for c in AEFC:
            df[c] = r[c].to_numpy()
        return df, int(df[AEFC[0]].isna().sum()), 0
    raise SystemExit('未知组')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--classes', default='')
    ap.add_argument('--subdir', default='')
    ap.add_argument('--log', default=AEF_INFO)
    a = ap.parse_args()
    classes = [x.strip() for x in a.classes.split(',') if x.strip()]
    fs = files_of(a.group, a.subdir, classes)
    emit('%s%s：%d 个 SHP' % (a.group, ('/' + a.subdir) if a.subdir else '', len(fs)))
    log = []
    for path in fs:
        stem = path[:-4]
        gpkg = stem.replace('_WGS84', '_WGS84') + '.gpkg' if not stem.endswith('.gpkg') else stem
        gpkg = stem + '.gpkg'
        if os.path.exists(gpkg) and not os.path.exists(path):
            continue
        t0 = time.time()
        r = shapefile.Reader(path, encoding='utf-8')
        fl = [x[0] for x in r.fields[1:]]
        recs = list(r.iterRecords())
        pts = list(r.iterShapes())
        r.close()
        df = pd.DataFrame(recs, columns=fl)
        shp_bytes = sum(os.path.getsize(stem + e) for e in ('.shp', '.shx', '.dbf', '.prj', '.cpg')
                        if os.path.exists(stem + e))
        df, miss, kex = attach_aef(a.group, path, df)
        geom = gpd.points_from_xy([p.points[0][0] for p in pts], [p.points[0][1] for p in pts])
        g = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326')
        if os.path.exists(gpkg):
            os.remove(gpkg)
        g.to_file(gpkg, driver='GPKG', layer=os.path.basename(stem), engine='pyogrio', SPATIAL_INDEX='NO')
        # 回读校验
        b = gpd.read_file(gpkg, engine='pyogrio')
        assert len(b) == len(df), (len(b), len(df))
        rng = np.random.RandomState(3)
        idx = rng.permutation(len(df))[:30]
        md_geo = max(abs(b.geometry.iloc[i].x - g.geometry.iloc[i].x) +
                     abs(b.geometry.iloc[i].y - g.geometry.iloc[i].y) for i in idx)
        md_aef = float(np.nanmax(np.abs(b['A00'].to_numpy(np.float32)[idx] - g['A00'].to_numpy(np.float32)[idx])))
        assert md_geo < 1e-9 and (md_aef == 0 or np.isnan(md_aef)), (md_geo, md_aef)
        gpkg_bytes = os.path.getsize(gpkg)
        for e in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
            if os.path.exists(stem + e):
                os.remove(stem + e)
        rec = dict(group=a.group, subdir=a.subdir, shp=os.path.relpath(path, ROOT), n=int(len(df)),
                   aef_missing=miss, aef_key_missing=kex, shp_bytes=int(shp_bytes),
                   gpkg_bytes=int(gpkg_bytes), secs=round(time.time() - t0, 1))
        log.append(rec)
        emit('  ✓ %-52s n=%-7d AEF缺 %d ｜ %.1f→%.1f MB ｜ %.0fs' % (
            os.path.basename(path), len(df), miss, shp_bytes / 2**20, gpkg_bytes / 2**20, rec['secs']))
    old = json.load(open(a.log, encoding='utf-8')) if os.path.exists(a.log) else []
    json.dump(old + log, open(a.log, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    tot_old = sum(x['shp_bytes'] for x in log) / 2**30
    tot_new = sum(x['gpkg_bytes'] for x in log) / 2**30
    miss_tot = sum(x['aef_missing'] for x in log)
    emit('完成 %d 文件：%d 点 ｜ SHP %.2f GB → GPKG %.2f GB ｜ AEF 缺失合计 %d' % (
        len(log), sum(x['n'] for x in log), tot_old, tot_new, miss_tot))


if __name__ == '__main__':
    main()
