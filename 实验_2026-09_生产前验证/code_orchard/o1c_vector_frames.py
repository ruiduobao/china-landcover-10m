# -*- coding: utf-8 -*-
"""o1c_vector_frames.py — 乔灌园地：矢量/点类候选框采样（OSM 果园 + SDPT 经济林 + GBIF + 人工林负样本）

* 输入：
  - F:/lc_work/外部样本/OSM/lu/gis_osm_landuse_a_free_1.shp（fclass=orchard/vineyard → 代表点）
  - E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp（iso=CHN 且 plant_ag=Tree crops → 代表点）
  - Z:/.../数据/外部样本/GBIF/园地_Orchard.parquet（中国区内、坐标不确定度<10km、year≥2000）
  - E:/地理所/论文/博士论文/数据/数据/全球人工林产品数据/global_plantedForest_30m_E*.tif（值>0 → 人工林负样本框，仅华南 5 省带）
* 规则源：各源仅作**选点框**（真值一律由后续影像判读/判别器给出）；一源一 frame 名
* 门槛：OSM/SDPT 每多边形取代表点；GBIF 上限 8000；人工林负样本上限 6000
* 输出：data/orchard/cand_vec.parquet（lon,lat,frame,src_id）
* 用法：python o1c_vector_frames.py
幂等：输出存在即跳过（--force 重写）。
"""
import glob
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/v31_exp'
OUT = os.path.join(WORK, 'data', 'orchard', 'cand_vec.parquet')
OSM = r'F:/lc_work/外部样本/OSM/lu/gis_osm_landuse_a_free_1.shp'
SDPT = r'E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp'
GBIF = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/外部样本/GBIF/园地_Orchard.parquet'
PF_TILES = [r'E:/地理所/论文/博士论文/数据/数据/全球人工林产品数据/global_plantedForest_30m_E%dN%d.tif'
            % (e, n) for e in (100, 105, 110, 115) for n in (20, 25, 30)]
PF_CAP = 6000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def poly_points(gdf, tag):
    """多边形 → 代表点（每条 1 点）。"""
    pts = gdf.copy()
    pts['geometry'] = pts.geometry.representative_point()
    return pd.DataFrame({'lon': pts.geometry.x, 'lat': pts.geometry.y,
                         'frame': tag, 'src_id': pts.index.astype(str)})


def main():
    rng = np.random.default_rng(20261012)
    if os.path.exists(OUT) and '--force' not in sys.argv:
        emit('已存在 %s（--force 重写）' % OUT)
        return
    parts = []
    # 1) OSM 果园/葡萄园
    try:
        import geopandas as gpd
        g = gpd.read_file(OSM, columns=['fclass'])
        g = g[g.fclass.isin(['orchard', 'vineyard'])].reset_index(drop=True)
        p = poly_points(g, 'osm_orchard')
        p['frame'] = np.where(g.fclass == 'vineyard', 'osm_vineyard', 'osm_orchard').tolist()
        emit('OSM: orchard %d / vineyard %d' % ((g.fclass == 'orchard').sum(), (g.fclass == 'vineyard').sum()))
        parts.append(p)
    except Exception as e:
        emit('OSM 失败: %s' % str(e)[:120])
    # 2) SDPT Tree crops（中国）—— 属性过滤 iso=CHN（避免几何全读 2.3M 图斑，速度关键）
    try:
        import geopandas as gpd
        g = gpd.read_file(SDPT, where="iso='CHN'", columns=['iso', 'plant_ag', 'timber_ag'])
        g = g[g.iso == 'CHN']
        tc = g[g.plant_ag == 'Tree crops'].reset_index(drop=True)
        pf = g[g.plant_ag == 'Planted forest'].reset_index(drop=True)
        if len(tc):
            parts.append(poly_points(tc, 'sdpt_treecrop'))
        if len(pf):
            q = poly_points(pf, 'sdpt_plantedforest_neg')
            q = q.sample(n=min(3000, len(q)), random_state=20261012)
            parts.append(q)
        emit('SDPT CHN: Tree crops %d / Planted forest %d' % (len(tc), len(pf)))
    except Exception as e:
        emit('SDPT 失败: %s' % str(e)[:120])
    # 3) GBIF 园地物种点
    try:
        d = pd.read_parquet(GBIF)
        d = d[(d.decimalLongitude >= 73) & (d.decimalLongitude <= 136) &
              (d.decimalLatitude >= 18) & (d.decimalLatitude <= 54)]
        u = pd.to_numeric(d.coordinateUncertaintyInMeters, errors='coerce')
        d = d[(u.isna()) | (u < 10000)]
        d = d[(pd.to_numeric(d.year, errors='coerce').fillna(0) >= 2000) | d.year.isna()]
        d = d.sample(n=min(8000, len(d)), random_state=20261012)
        parts.append(pd.DataFrame({'lon': d.decimalLongitude.to_numpy(), 'lat': d.decimalLatitude.to_numpy(),
                                   'frame': 'gbif_orchard_species',
                                   'src_id': d.species.fillna('').to_numpy()}))
        emit('GBIF: %d 点' % len(d))
    except Exception as e:
        emit('GBIF 失败: %s' % str(e)[:120])
    # 4) 人工林负样本（华南带值>0 抽稀）
    try:
        import rasterio
        lons, lats = [], []
        for f in PF_TILES:
            if not os.path.exists(f):
                continue
            with rasterio.open(f) as ds:
                a = ds.read(1, out_shape=(ds.height // 10, ds.width // 10))
                rr, cc = np.nonzero(a > 0)
                if len(rr) == 0:
                    continue
                if len(rr) > 4000:
                    sel = rng.choice(len(rr), 4000, replace=False)
                    rr, cc = rr[sel], cc[sel]
                xs, ys = rasterio.transform.xy(ds.transform, rr * 10 + 5, cc * 10 + 5, offset='center')
                lons += list(np.atleast_1d(xs)); lats += list(np.atleast_1d(ys))
        if lons:
            n = min(PF_CAP, len(lons))
            idx = rng.permutation(len(lons))[:n]
            parts.append(pd.DataFrame({'lon': np.asarray(lons)[idx], 'lat': np.asarray(lats)[idx],
                                       'frame': 'plantedforest_neg', 'src_id': ''}))
            emit('人工林负样本: 池 %d → 取 %d' % (len(lons), n))
    except Exception as e:
        emit('人工林失败: %s' % str(e)[:120])
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.lon >= 73) & (df.lon <= 136) & (df.lat >= 18) & (df.lat <= 54)].reset_index(drop=True)
    df['src_id'] = df['src_id'].astype(str)
    df.to_parquet(OUT, index=False)
    emit('合计 %d → %s' % (len(df), OUT))
    emit(df.frame.value_counts().to_dict())


if __name__ == '__main__':
    main()
