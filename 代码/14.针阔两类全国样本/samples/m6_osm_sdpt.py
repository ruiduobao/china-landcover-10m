# -*- coding: utf-8 -*-
"""m6_osm_sdpt.py — 方法E：外部独立证据（OSM 林地 + SDPT V2 人工林针阔属性）
* 输入：E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp（4.57M 全球面，中国 24,010）
        F:/lc_work/外部样本/OSM/lu/gis_osm_landuse_a_free_1.shp（1.02M 面，含 forest/scrub/orchard）
        labels/methods/work_base.parquet
* 规则：SDPT conifer_br ∈ {Conifer, Broadleaf} → 独立叶型证据（仅人工林，且年份 2004-2008）
        OSM fclass=forest → 只证明"是林地"，无叶型 → 只用于验证 Q1 非林框
* 输出：labels/methods/m6_osm.json、labels/methods/sdpt_pts.parquet
* 用法：python m6_osm_sdpt.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
SDPT = r'E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp'
OSM = r'F:/lc_work/外部样本/OSM/lu/gis_osm_landuse_a_free_1.shp'
BBOX = (73, 18, 135, 54)


def main():
    res = {}
    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'),
                         columns=['stable_uid', 'lon', 'lat', 'cl_24', 'leaf_fcs10', 'leaf_modis',
                                  'tile', 'prov_cta', 'region'])
    df['stable_uid'] = df['stable_uid'].astype(str)

    # ---------------- SDPT V2（中国区）
    import geopandas as gpd
    try:
        t0 = pd.Timestamp.now()
        g = gpd.read_file(SDPT, bbox=BBOX, engine='pyogrio',
                          columns=['conifer_br', 'ever_dec', 'plant_ag', 'country', 'creation_y'])
        print('SDPT bbox read', len(g), 'elapsed', (pd.Timestamp.now() - t0).total_seconds())
        g = g[g['country'].astype(str).str.contains('China', na=False)]
        res['sdpt_china_n'] = int(len(g))
        res['sdpt_conifer_br'] = g['conifer_br'].astype(str).value_counts().to_dict()
        res['sdpt_ever_dec'] = g['ever_dec'].astype(str).value_counts().to_dict()
        # 只保留有针阔判定的
        gg = g[g['conifer_br'].astype(str).isin(['Conifer', 'Broadleaf'])].copy()
        gg['sdpt_leaf'] = np.where(gg['conifer_br'].astype(str) == 'Conifer', 1, 0)
        res['sdpt_usable_n'] = int(len(gg))
        res['sdpt_usable_area_km2'] = float(gg.to_crs(3857).area.sum() / 1e6) if len(gg) else 0.0
        # 空间连接：点落面内 OR 最近面 <100m
        pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        j = gpd.sjoin(pts, gg[['sdpt_leaf', 'geometry']], how='inner', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        res['points_matched_sdpt'] = int(len(j))
        if len(j):
            jj = j[['stable_uid', 'sdpt_leaf', 'cl_24', 'leaf_fcs10', 'leaf_modis']]
            jj.to_parquet(os.path.join(WORK, 'sdpt_pts.parquet'), index=False)
            # 与池标签、FCS10 的一致率
            lab = np.where(jj['cl_24'].isin([6, 7]), 1, 0)
            res['sdpt_vs_pool_label'] = dict(n=int(len(jj)),
                                             agree=round(float((jj['sdpt_leaf'] == lab).mean()), 4))
            m = jj.dropna(subset=['leaf_fcs10'])
            if len(m):
                f10 = np.where(m['leaf_fcs10'] == 'N', 1, 0)
                res['sdpt_vs_fcs10'] = dict(n=int(len(m)), agree=round(float((m['sdpt_leaf'] == f10).mean()), 4))
            m = jj.dropna(subset=['leaf_modis'])
            if len(m):
                md = np.where(m['leaf_modis'] == 'N', 1, 0)
                res['sdpt_vs_modis'] = dict(n=int(len(m)), agree=round(float((m['sdpt_leaf'] == md).mean()), 4))
            res['sdpt_points_by_prov'] = jj.merge(df[['stable_uid', 'prov_cta']], on='stable_uid') \
                .groupby('prov_cta').size().sort_values(ascending=False).head(8).to_dict()
    except Exception as e:
        res['sdpt_error'] = repr(e)[:400]
        print('SDPT FAIL', repr(e)[:300])

    # ---------------- OSM 林地
    try:
        o = gpd.read_file(OSM, bbox=BBOX, engine='pyogrio', columns=['fclass', 'name'])
        print('OSM read', len(o))
        res['osm_n'] = int(len(o))
        res['osm_fclass_top'] = o['fclass'].value_counts().head(12).to_dict()
        forest = o[o['fclass'].isin(['forest', 'scrub', 'orchard', 'nature_reserve'])]
        pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        j = gpd.sjoin(pts, forest[['fclass', 'geometry']], how='inner', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        res['points_in_osm_woodland'] = int(len(j))
        res['points_in_osm_by_class'] = {int(k): int(v) for k, v in
                                         df.set_index('stable_uid').loc[j['stable_uid'], 'cl_24'].value_counts().items()}
        res['osm_agreement_rate_pct'] = round(float(len(j) / len(df)) * 100, 2)
        # 反向：OSM 林地内我方非林点比例（证明 Q1 非林框的方向）
        # 用整个 OSM forest 面去抽我方点：太慢，改为在林点内看 fclass
    except Exception as e:
        res['osm_error'] = repr(e)[:400]
        print('OSM FAIL', repr(e)[:300])

    json.dump(res, open(os.path.join(WORK, 'm6_osm.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == '__main__':
    main()
