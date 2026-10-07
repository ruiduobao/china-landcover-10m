# -*- coding: utf-8 -*-
"""
s7_validation.py — r1 步骤6：独立验证池 → r1_validation.parquet（不入训练）
* 修复旧链 validation_pools 两处问题：
    1) mountains_lc 未做国界裁剪（全球点集）
    2) 未映射码 -1（yrd 非不透水参考）/ 2（hdlv 森林组码）混入 class_new
* 约定: class_new 不可映射 30 类时置 -1，另存 class_group（level0 组码）供二元评估；
  README 中明确 -1 只参与"组级/二元"验证。
* 来源:
    mountains_lc  CCI 码 CCI_MAP 映射（Conf2020>=2），国界裁剪
    yrd           name=1 不透水 → GUB 拆 190/200；name=0 → class_new=-1(group=6 不透水组对侧)
    hdlv_xj       1耕地10/2森林组4草地130/5水202/7裸地201；2→class_new=-1,group=2
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

# CCI 码 → 30 类（亚类歧义处 conf 降）——同 n10 权威表
CCI_MAP = {10: 10, 20: 10, 50: 51, 60: 61, 70: 71, 80: 81, 90: 91,
           120: 121, 130: 130, 140: 140, 150: 150, 180: 181,
           190: 190, 200: 201, 210: 202, 220: 220}
CCI_CONF = {50: .5, 60: .5, 70: .5, 80: .5, 90: .5, 180: .6}
HDLV_MAP = {1: (10, -1), 2: (-1, 2), 4: (130, -1), 5: (202, -1), 7: (201, -1)}

def main():
    t0 = time.time()
    pools = []

    # 1) mountains（CCI 码，全球集 → 国界裁剪）
    mf = glob.glob(os.path.join(C.EXT, 'mountains_lc', '**', '*.shp'), recursive=True)
    if mf:
        import geopandas as gpd
        g = gpd.read_file(mf[0])
        if g.crs and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
        m = g[(g['Conf2020'] >= 2) & (g['LandCover2'] > 0)].copy()
        m['class_new'] = m['LandCover2'].astype(int).map(CCI_MAP).fillna(-1).astype(int)
        m = m[m.class_new > 0]
        lon = m['lon'].to_numpy(); lat = m['lat'].to_numpy()
        b = C.BBOX
        mm = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
        m = m[mm]
        inside = G.china_contains(m['lon'].to_numpy(), m['lat'].to_numpy())
        m = m[inside]
        m['class_group'] = -1
        m['src'] = 'mountains_lc_validation'
        m['src_conf'] = m['LandCover2'].astype(int).map(lambda c: CCI_CONF.get(c, 0.75))
        m['year'] = 2020
        pools.append(m[['lon', 'lat', 'class_new', 'class_group', 'src', 'src_conf', 'year']])
        print(f'mountains: 国界内 {len(m):,}')

    # 2) YRD 长三角不透水
    yf = glob.glob(os.path.join(C.EXT, 'yrd_impervious', '*.shp'))
    if yf:
        import geopandas as gpd
        g = gpd.read_file(yf[0])
        g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
        imp = g[g['name'] == 1].copy()
        if len(imp):
            in_gub = G.gub_contains(imp['lon'].to_numpy(), imp['lat'].to_numpy())
            imp['class_new'] = np.where(in_gub, 190, 200).astype(int)
            imp['class_group'] = 6
            imp['src'] = 'yrd_imperv_1985_2020'; imp['src_conf'] = 0.85; imp['year'] = 2020
            pools.append(imp[['lon', 'lat', 'class_new', 'class_group', 'src', 'src_conf', 'year']])
        print(f'yrd 不透水: {len(imp):,}（城镇 {int((imp.class_new==190).sum()):,}）')

    # 3) HDLV-XJ（2=森林组码不可映射 → class_new=-1, class_group=2）
    hf = os.path.join(C.EXT, 'hdlv_xj_validation.parquet')
    if os.path.exists(hf):
        h = pd.read_parquet(hf)
        raw_v = h['class_new'].map(lambda c: {10: 1, 2: 2, 130: 4, 202: 5, 201: 7}.get(c, -1))
        h['class_new'] = raw_v.map(lambda v: HDLV_MAP.get(v, (-1, -1))[0]).astype(int)
        h['class_group'] = raw_v.map(lambda v: HDLV_MAP.get(v, (-1, -1))[1]).astype(int)
        h['src'] = 'hdlv_xj_2020'; h['src_conf'] = 0.8; h['year'] = 2020
        pools.append(h[['lon', 'lat', 'class_new', 'class_group', 'src', 'src_conf', 'year']])
        print(f'hdlv_xj: {len(h):,}（组级 {int((h.class_new==-1).sum()):,}）')

    df = pd.concat(pools, ignore_index=True)
    df['tier'] = 'validation'
    df['province'] = G.province_of(df['lon'].to_numpy(), df['lat'].to_numpy())
    df.to_parquet(C.R1_VALID, index=False)
    print('\n验证池合计:', len(df))
    print(df['src'].value_counts().to_string())

    summary = {'total': int(len(df)),
               'by_src': {k: int(v) for k, v in df['src'].value_counts().items()},
               'group_only': int((df.class_new == -1).sum()),
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(C.R1_VALID.replace('.parquet', '_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('输出:', C.R1_VALID, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
