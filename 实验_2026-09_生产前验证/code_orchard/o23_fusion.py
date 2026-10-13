# -*- coding: utf-8 -*-
"""o23_fusion.py — 多源证据融合 → 高精度样本分层（核心装配）

* 输入：data/orchard/scored_gem.parquet（big pool + gem 610，p_treecrop）
        data/orchard/aef/{tea,rubber}_p*.parquet（茶园/橡胶候选 2023 AEF）
        GEM 训练集（重训模型配方同 o20）
        本地栅格：TeaMap tif、Rubber_SEA china_tiles（采样值）
        矢量：OSM 果园/葡萄园 shp、SDPT Tree crops shp（点在多边形）
* 规则源（**预注册分层**）：
  证据源 = {gem2: GEM 图=2, pnew: 新模型≥0.5, rub1: 橡胶图=1, tea1: 茶园图=1, osm, sdpt}
  A_多源高置信 = pnew≥0.5 且 非 pnew 证据源 ≥2
  B_双源        = pnew≥0.5 且 非 pnew 证据源 =1；或 pnew≥0.7 且 非 pnew 证据源 ≥1
  C_单源强     = pnew≥0.7（无其他源）或（1 源 且 pnew≥0.3）
* 输出：data/orchard/fusion_all.parquet + fusion_tiers.csv + results/orchard/fusion.json
* 用法：python o23_fusion.py
"""
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')
GEM_TRAIN = r'F:/lc_work/外部样本/GEM_Forest/training_with_AEF.csv'
TEA_TIF = r'F:/lc_work/外部样本/rare_class_dl/TeaMap_2022_China/CAS.TEA-MAP.CHN.tif'
RUB_DIR = r'F:/lc_work/外部样本/rare_class_dl/Rubber_SEA_2020/china_tiles'
OSM_SHP = r'F:/lc_work/外部样本/OSM/lu/gis_osm_landuse_a_free_1.shp'
SDPT_SHP = r'E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def train_model():
    d = pd.read_csv(GEM_TRAIN, encoding='utf-8-sig')
    d['y'] = d['forest'].isin([2, 3]).astype(int)
    X = d[VC.FEATS].to_numpy('float64')
    ok = ~np.isnan(X).any(axis=1)
    y = d.y.to_numpy()[ok]
    X = X[ok]
    rng = np.random.default_rng(7)
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=2 * len(pos), replace=False)
    sub = np.concatenate([pos, neg])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7)
    clf.fit(X[sub], y[sub])
    return clf


def raster_sample(paths_or_dir, lon, lat):
    """从多个/目录下栅格采样值（返回 float 数组；未覆盖为 NaN）。"""
    import rasterio, rasterio.windows as rw
    vals = np.full(len(lon), np.nan)
    if os.path.isdir(paths_or_dir):
        fps = sorted(glob.glob(os.path.join(paths_or_dir, '*.tif')))
    else:
        fps = [paths_or_dir]
    for fp in fps:
        with rasterio.open(fp) as ds:
            b = ds.bounds
            m = (lon >= b.left) & (lon <= b.right) & (lat >= b.bottom) & (lat <= b.top) & np.isnan(vals)
            idx = np.where(m)[0]
            for i in idx:
                r, c = ds.index(float(lon[i]), float(lat[i]))
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    v = ds.read(1, window=rw.Window(c, r, 1, 1))[0, 0]
                    vals[i] = float(v)
    return vals


def poly_flag(shp, lon, lat, where=None, tag=''):
    """点在多边形（返回 bool）。"""
    import geopandas as gpd
    from shapely.strtree import STRtree
    from shapely.geometry import Point
    kw = {}
    if where:
        g = gpd.read_file(shp, where=where, columns=['fclass'] if 'OSM' in shp else ['iso', 'plant_ag'])
    else:
        g = gpd.read_file(shp, columns=['fclass'] if 'OSM' in shp else ['iso', 'plant_ag'])
    if 'fclass' in g.columns:
        g = g[g.fclass.isin(['orchard', 'vineyard'])]
    if 'plant_ag' in g.columns:
        g = g[g.plant_ag == 'Tree crops']
    flags = np.zeros(len(lon), dtype=bool)
    if len(g) == 0:
        return flags
    tree = STRtree(g.geometry.values)
    pts = [Point(x, y) for x, y in zip(lon, lat)]
    res = tree.query(pts)
    if res.size:
        for col in range(res.shape[1]):
            r, c = int(res[0, col]), int(res[1, col])
            if not flags[r] and tree.geometries[c].contains(pts[r]):
                flags[r] = True
    return flags


def main():
    t0 = time.time()
    clf = train_model()
    emit('模型就绪')
    # ---- 1. 组装候选（p≥0.3）----
    sg = pd.read_parquet(os.path.join(ORD, 'scored_gem.parquet'))
    pool = sg[(sg.src == 'pool')].copy()
    pool['src'] = 'pool_' + pool.frame.fillna('').astype(str)
    gem = sg[(sg.src == 'gem')].copy()
    gem['src'] = 'gem_map'
    parts = [pool[pool.p_treecrop >= 0.3], gem]
    for s, fp in [('tea', os.path.join(ORD, 'aef')), ('rubber', os.path.join(ORD, 'aef'))]:
        fs = sorted(glob.glob(os.path.join(fp, '%s_p*.parquet' % s)))
        if not fs:
            emit('%s 特征未就绪，跳过' % s)
            continue
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        d = d[d.aef_ok]
        d['p_treecrop'] = clf.predict_proba(d[VC.FEATS].to_numpy('float32'))[:, 1]
        d['src'] = s + '_map'
        d['frame'] = s
        emit('%s: %d 点，p 中位 %.3f，p>=0.5 占比 %.3f，p>=0.7 占比 %.3f' % (
            s, len(d), d.p_treecrop.median(), (d.p_treecrop >= 0.5).mean(), (d.p_treecrop >= 0.7).mean()))
        parts.append(d[d.p_treecrop >= 0.3][['lon', 'lat', 'frame', 'p_treecrop', 'src']])
        # 全量分布存一份（对照）
        d[['lon', 'lat', 'p_treecrop']].to_parquet(os.path.join(ORD, 'scored_%s.parquet' % s), index=False)
    allc = pd.concat(parts, ignore_index=True)
    allc['k'] = allc.lon.round(6).astype(str) + '_' + allc.lat.round(6).astype(str)
    allc = allc.drop_duplicates(subset=['k']).reset_index(drop=True)
    emit('融合候选 %d 点（p>=0.3）' % len(allc))
    lon = allc.lon.to_numpy(); lat = allc.lat.to_numpy()
    # ---- 2. 多源标志 ----
    # GEM 图（GEE，仅 pool/tea/rubber 需要）
    need_gem = allc.src != 'gem_map'
    import ee
    VC.ensure_ctx('zixen8v8')
    gemimg = ee.ImageCollection('projects/danielp-cuni/assets/GEM-Forest/GEM-Forest_2020').mosaic().select('b1')
    gemv = np.full(len(allc), -1.0)
    sub = allc[need_gem].reset_index().rename(columns={'index': 'orig'})
    for i in range(0, len(sub), 5000):
        ch = sub.iloc[i:i + 5000]
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'i': int(r.orig)})
                 for r in ch.itertuples()]
        got = gemimg.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                                   tileScale=4).getInfo()['features']
        for g in got:
            gemv[g['properties']['i']] = float(g['properties'].get('b1', -1) or -1)
    allc['gem2'] = gemv == 2
    allc.loc[allc.src == 'gem_map', 'gem2'] = True
    # 橡胶图 / 茶园图（本地）
    allc['rub1'] = raster_sample(RUB_DIR, lon, lat) == 1
    allc['tea1'] = raster_sample(TEA_TIF, lon, lat) == 1
    allc.loc[allc.src == 'rubber_map', 'rub1'] = True
    allc.loc[allc.src == 'tea_map', 'tea1'] = True
    emit('栅格标志完成（%.0fs）：rub1 %d / tea1 %d / gem2 %d' % (time.time() - t0, int(allc.rub1.sum()),
                                                           int(allc.tea1.sum()), int(allc.gem2.sum())))
    # OSM / SDPT（矢量）
    try:
        allc['osm'] = poly_flag(OSM_SHP, lon, lat)
        emit('OSM 命中 %d' % int(allc.osm.sum()))
    except Exception as e:
        allc['osm'] = False
        emit('OSM 失败: %s' % str(e)[:80])
    try:
        allc['sdpt'] = poly_flag(SDPT_SHP, lon, lat, where="iso='CHN'")
        emit('SDPT 命中 %d' % int(allc.sdpt.sum()))
    except Exception as e:
        allc['sdpt'] = False
        emit('SDPT 失败: %s' % str(e)[:80])
    # ---- 3. 分层 ----
    srcs = allc[['gem2', 'rub1', 'tea1', 'osm', 'sdpt']].sum(axis=1)
    allc['n_src'] = srcs
    pn = allc.p_treecrop
    tier = np.full(len(allc), 'C_单源强', dtype=object)
    tier[(pn >= 0.7) & (srcs >= 1)] = 'B_双源'
    tier[(pn >= 0.5) & (srcs >= 2)] = 'A_多源高置信'
    tier[(pn >= 0.5) & (srcs == 1)] = 'B_双源'
    allc['tier'] = tier
    allc = allc.drop(columns=['k'])
    allc.to_parquet(os.path.join(ORD, 'fusion_all.parquet'), index=False)
    allc.drop(columns=VC.FEATS, errors='ignore').to_csv(os.path.join(ORD, 'fusion_tiers.csv'),
                                                        index=False, encoding='utf-8-sig')
    stats = {'n_cand': int(len(allc)), 'tier': allc.tier.value_counts().to_dict(),
             'by_src': allc.src.value_counts().to_dict(),
             'cross': allc.groupby(['src', 'tier']).size().to_dict()}
    emit('分层: %s' % stats['tier'])
    print(allc.groupby(['tier']).size())
    VC.jsave({k: (v if not isinstance(v, dict) else {str(kk): int(vv) for kk, vv in v.items()})
              for k, v in stats.items()}, os.path.join(RES, 'fusion.json'))
    emit('→ fusion_all.parquet / fusion_tiers.csv（%.0f min）' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
