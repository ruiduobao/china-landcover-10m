# -*- coding: utf-8 -*-
"""m7_mangrove_export.py — 红树林扩充 第7步：交付打包（SHP/CSV/证据表/备份）

* 输入：out/qc/keep_v1.parquet（新增点）、cand/candidates.parquet、out/feats/{ev,aef}、
        原 652（2023/红树林_2023_WGS84.shp + feats/orig_{ev,aef}）
* 输出（到 交付根/红树林/）：
    扩充_v1/红树林_新增_<n>_2023_WGS84.*（字段见下）+ CSV
    扩充_v1/红树林_全量_<N>_2023_WGS84.*（set=orig/new）+ CSV
    扩充_v1/证据表_新增.parquet（含 AEF64）
    原始样本_备份/（原 652 拷贝 + MD5 + 原证据）
* 字段：cid, lon(F13,6), lat(F13,6), set(orig/new), src(mask 来源 wc95|giri|both), tier(qc 结果),
        s2_ndvi(N7,1), trees(F7,3), max_300(F7,3), dem(N4), ct_prov(C30), ct_code(C8), ct_km(N9,2)
* CTAmap 归属：省界内取本省；界外按最近省份（同 s4 口径），ct_km 为到省界的最近距离
* 用法：python m7_mangrove_export.py
"""
import glob, json, os, shutil, sys
import numpy as np
import pandas as pd
import shapefile
import geopandas as gpd
from shapely import STRtree, shortest_line, Point
from pyproj import Geod

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/红树林'
EXP = os.path.join(DEST, '扩充_v1')
CODE_FILES = ['m1_mangrove_scope.py', 'm2_mangrove_sample.py', 'm3_mangrove_assemble.py',
              'm4_mangrove_fetch.py', 'm5_mangrove_qc.py', 'm6_mangrove_validate.py',
              'm7_mangrove_export.py']
AEF = ['A%02d' % i for i in range(64)]
GEOD = Geod(ellps='WGS84')
FIELDS = [('cid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('set', 'C', 6),
          ('src', 'C', 8), ('tier', 'C', 8), ('s2_ndvi', 'N', 7, 1), ('trees', 'F', 7, 3),
          ('max_300', 'F', 7, 3), ('dem', 'N', 4), ('ct_prov', 'C', 30), ('ct_code', 'C', 8),
          ('ct_km', 'N', 9, 2)]


def load(stage):
    fs = sorted(glob.glob(os.path.join(WORK, 'out', 'feats', stage, '*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d['cid'] = d.cid.astype(str)
    return d.drop_duplicates('cid')


def write_pts(base, rows, xs, ys):
    w = shapefile.Writer(base, shapeType=shapefile.POINT, encoding='utf-8')
    for spec in FIELDS:
        nm, tp, sz = spec[:3]
        w.field(nm, tp, sz, spec[3] if len(spec) > 3 else 0)
    for x, y, rec in zip(xs, ys, rows):
        w.point(float(x), float(y)); w.record(*rec)
    w.close()


def ctamap_assign(d):
    prov = gpd.read_file(r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp')
    names = prov['省'].astype(str).tolist(); codes = prov['省级码'].astype(str).tolist()
    geoms = list(prov.geometry); tree = STRtree(geoms)
    ctp = np.full(len(d), '', dtype=object); ctc = np.full(len(d), '', dtype=object); km = np.zeros(len(d))
    inside = gpd.GeoSeries(gpd.points_from_xy(d.lon, d.lat), crs='EPSG:4326').within(
        prov.geometry.union_all()).to_numpy()
    for i in np.where(~inside)[0]:
        pt = Point(float(d.lon.iloc[i]), float(d.lat.iloc[i]))
        j = int(tree.nearest(pt))
        line = shortest_line(pt, geoms[j]); ex, ey = line.coords[-1]
        _, _, dist = GEOD.inv(pt.x, pt.y, ex, ey)
        ctp[i] = names[j]; ctc[i] = codes[j]; km[i] = dist / 1000.0
    for i in np.where(inside)[0]:
        j = int(tree.nearest(Point(float(d.lon.iloc[i]), float(d.lat.iloc[i]))))
        ctp[i] = names[j]; ctc[i] = codes[j]; km[i] = 0.0
    d = d.copy(); d['ct_prov'] = ctp; d['ct_code'] = ctc; d['ct_km'] = km
    return d


def main():
    os.makedirs(EXP, exist_ok=True)
    os.makedirs(os.path.join(DEST, '原始样本_备份'), exist_ok=True)
    keep = pd.read_parquet(os.path.join(WORK, 'out', 'qc', 'keep_v1.parquet'))
    ev = load('ev'); aef = load('aef')
    k = keep.merge(ev, on='cid', how='left', suffixes=('', '_e')).merge(aef, on='cid', how='left')
    k['src'] = np.where((k.wc95 == 1) & (k.giri == 1), 'both', np.where(k.wc95 == 1, 'wc95', 'giri'))
    k['tier'] = 'strong'   # 通过 K1–K6
    k = k.rename(columns={'s2_ndvi_xia': 's2_ndvi'})
    k['set'] = 'new'
    k = ctamap_assign(k)
    # 原 652
    r = shapefile.Reader(os.path.join(DEST, '2023', '红树林_2023_WGS84.shp'), encoding='utf-8')
    fl = [x[0] for x in r.fields[1:]]
    orows = []
    for rec, shp in zip(r.iterRecords(), r.iterShapes()):
        orows.append(dict(cid=str(rec[fl.index('pid')]), lon=shp.points[0][0], lat=shp.points[0][1],
                          tier=rec[fl.index('tier')], src='origpool', set='orig'))
    r.close()
    o = pd.DataFrame(orows)
    o = ctamap_assign(o)
    # SHP/CSV
    def to_rows(x):
        return [(str(c_), float(x_), float(y_), s_, sr_, t_,
                 (float(n_) if pd.notna(n_) else -999.0), float(tr_), float(m_), int(dm_),
                 p_, co_, float(kk_))
                for c_, x_, y_, s_, sr_, t_, n_, tr_, m_, dm_, p_, co_, kk_ in zip(
                    x.cid, x.lon, x.lat, x['set'], x.src, x.tier, x.s2_ndvi, x.trees, x.max_300,
                    x.dem, x.ct_prov, x.ct_code, x.ct_km)]
    nb = os.path.join(EXP, '红树林_新增_%d_2023_WGS84' % len(k))
    write_pts(nb, to_rows(k), k.lon, k.lat)
    for ext in ('.prj', '.cpg'):
        shutil.copy2(os.path.join(DEST, '2023', '红树林_2023_WGS84' + ext), nb + ext)
    allx = pd.concat([o[['cid', 'lon', 'lat', 'set', 'tier']].assign(src='origpool',
                     s2_ndvi=np.nan, trees=np.nan, max_300=np.nan, dem=-9, ct_prov=o.ct_prov,
                     ct_code=o.ct_code, ct_km=o.ct_km),
                      k[['cid', 'lon', 'lat', 'set', 'tier', 'src', 's2_ndvi', 'trees', 'max_300',
                         'dem', 'ct_prov', 'ct_code', 'ct_km']]], ignore_index=True)
    ab = os.path.join(EXP, '红树林_全量_%d_2023_WGS84' % len(allx))
    write_pts(ab, to_rows(allx), allx.lon, allx.lat)
    for ext in ('.prj', '.cpg'):
        shutil.copy2(os.path.join(DEST, '2023', '红树林_2023_WGS84' + ext), ab + ext)
    k.drop(columns=['geometry'], errors='ignore').to_csv(os.path.join(EXP, '红树林_新增_%d_2023.csv' % len(k)),
                                                         index=False, encoding='utf-8-sig')
    allx.to_csv(os.path.join(EXP, '红树林_全量_%d_2023.csv' % len(allx)), index=False, encoding='utf-8-sig')
    k.to_parquet(os.path.join(EXP, '证据表_新增.parquet'), index=False)
    # 备份
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        shutil.copy2(os.path.join(DEST, '2023', '红树林_2023_WGS84' + ext),
                     os.path.join(DEST, '原始样本_备份', '红树林_2023_原始652_WGS84' + ext))
    oev = load('orig_ev'); oaef = load('orig_aef')
    oev.merge(oaef, on='cid', how='left').to_parquet(
        os.path.join(DEST, '原始样本_备份', '原652_证据.parquet'), index=False)
    print('新增 %d ｜ 全量 %d ｜ 省分布(新) %s' % (
        len(k), len(allx), k.ct_prov.value_counts().head(6).to_dict()))
    print('→', EXP)


if __name__ == '__main__':
    main()
