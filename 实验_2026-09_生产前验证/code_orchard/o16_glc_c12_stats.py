# -*- coding: utf-8 -*-
"""o16_glc_c12_stats.py — GLC class-12（乔灌园地）抽样结构调查：分布/年份/间距/省内归属

* 输入：data/orchard/glc_samples_china_c12.csv（o15 产出）、data/orchard/glc_samples_c12_global.parquet
        数据/边界/china_100000_full.json（DataV 省界，GCJ-02）
* 规则源：1° 格 = floor(lon),floor(lat)；省归属用 shapely 点在多边形（GCJ-02 界 vs WGS84 点，
          偏移约 0.5–0.9 km，仅影响界线上点，省/国级归属可用）
* 门槛：最近邻距离用等距近似（x=lon·111320·cosφ, y=lat·110574），局部误差 <0.1%
* 输出：results/orchard/c12_stats.json（全部数字）+ 控制台摘要
* 用法：python o16_glc_c12_stats.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RESD = os.path.join(WORK, 'results', 'orchard')
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
SEED = 7


def to_xy(lon, lat):
    """等距近似（局部可用）：x 向东、y 向北，单位 m"""
    lat_r = np.radians(lat)
    return lon * 111320.0 * np.cos(lat_r), lat * 110574.0


def nn_stats(lon, lat):
    x, y = to_xy(lon, lat)
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]                      # 最近邻距离（不含自身）
    return {
        'n': int(len(nn)),
        'nn_min_m': round(float(nn.min()), 1),
        'nn_p25_m': round(float(np.percentile(nn, 25)), 1),
        'nn_median_m': round(float(np.median(nn)), 1),
        'nn_p75_m': round(float(np.percentile(nn, 75)), 1),
        'nn_p95_m': round(float(np.percentile(nn, 95)), 1),
        'nn_max_m': round(float(nn.max()), 1),
        'frac_nn_lt_100m': round(float((nn < 100).mean()), 4),
        'frac_nn_lt_300m': round(float((nn < 300).mean()), 4),
        'frac_nn_gt_1000m': round(float((nn > 1000).mean()), 4),
    }


def main():
    out = {}
    df = pd.read_csv(os.path.join(ORD, 'glc_samples_china_c12.csv'))
    glob = pd.read_parquet(os.path.join(ORD, 'glc_samples_c12_global.parquet'))
    ext = json.load(open(os.path.join(RESD, 'c12_extract.json'), encoding='utf-8'))

    # ---------- 1. 基本量 ----------
    out['n_china_bbox'] = int(len(df))
    out['n_global'] = int(len(glob))
    out['bbox_share_of_global_pct'] = round(100 * len(df) / max(len(glob), 1), 3)
    out['n_cells_bbox'] = int(df.groupby([np.floor(df.lon).astype(int), np.floor(df.lat).astype(int)]).ngroups)
    out['columns'] = list(df.columns)
    out['year_dist'] = {str(k): int(v) for k, v in df.year.value_counts().items()}
    out['class_dist'] = {str(k): int(v) for k, v in df['class'].value_counts().items()}
    out['n_dup_coords'] = int(df.duplicated(['lon', 'lat']).sum())
    out['lon_range'] = [float(df.lon.min()), float(df.lon.max())]
    out['lat_range'] = [float(df.lat.min()), float(df.lat.max())]

    # ---------- 2. 1° 格 Top20 ----------
    cell = df.groupby([np.floor(df.lon).astype(int).rename('clon'), np.floor(df.lat).astype(int).rename('clat')]).size()
    cell = cell.sort_values(ascending=False)
    out['top20_cells'] = [{'clon': int(a), 'clat': int(b), 'n': int(v),
                           'cell': f'Lon{a}_Lat{b}'} for (a, b), v in cell.head(20).items()]
    out['cell_count_dist'] = {str(k): int(v) for k, v in
                              cell.value_counts().sort_index().items()}

    # ---------- 3. 省级归属（DataV 省界） ----------
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union
        g = gpd.read_file(BND)
        g = g[g.level == 'province'] if 'level' in g.columns else g
        g['geometry'] = g.geometry.make_valid()      # DataV 面有自相交，先修复
        union = unary_union(list(g.geometry))

        # 3a. 联表有效性对照（已知城市 + 已知境外城市）
        ctrl = [('北京', 116.407, 39.904), ('上海', 121.474, 31.230), ('广州', 113.264, 23.129),
                ('乌鲁木齐', 87.617, 43.793), ('拉萨', 91.132, 29.660), ('哈尔滨', 126.635, 45.756),
                ('成都', 104.066, 30.572), ('三亚', 109.508, 18.247),
                ('新德里', 77.209, 28.614), ('达卡', 90.412, 23.810),
                ('加德满都', 85.324, 27.717), ('河内', 105.834, 21.028)]
        cs = gpd.GeoDataFrame({'nm': [c[0] for c in ctrl]},
                              geometry=[Point(c[1], c[2]) for c in ctrl], crs='EPSG:4326')
        cj = gpd.sjoin(cs, g[['name', 'geometry']], how='left', predicate='within')
        out['join_control'] = {r.nm: (r.name if isinstance(r.name, str) else '境外')
                               for r in cj.itertuples()}

        pts = gpd.GeoDataFrame(df, geometry=[Point(a, b) for a, b in zip(df.lon, df.lat)],
                               crs='EPSG:4326')
        join = gpd.sjoin(pts, g[['name', 'geometry']], how='left', predicate='within')
        join = join[~join.index.duplicated(keep='first')]
        vc = join['name'].fillna('（境外/未命中）').value_counts()
        out['province_dist'] = {str(k): int(v) for k, v in vc.items()}
        out['n_inside_china'] = int((join['name'].notna()).sum())
        out['n_outside_china'] = int(join['name'].isna().sum())
        # 3b. 到中国国界的距离（度→km，按点纬度折算），判断"境外"是真境外还是界线抖动
        d_deg = np.array([geom.distance(union) for geom in pts.geometry])
        d_km = d_deg * 111.32 * np.cos(np.radians(df.lat.values))
        out['dist_to_china_km'] = {
            'within_1km': int((d_km <= 1).sum()),
            'within_5km': int((d_km <= 5).sum()),
            'within_25km': int((d_km <= 25).sum()),
            'gt_25km': int((d_km > 25).sum()),
            'gt_100km': int((d_km > 100).sum()),
            'median_km': round(float(np.median(d_km)), 1),
        }
        # 中国境内点的 Top 格
        inb = join[join['name'].notna()]
        c2 = inb.groupby([np.floor(inb.lon).astype(int), np.floor(inb.lat).astype(int)]).size().sort_values(
            ascending=False)
        out['top20_cells_inside_china'] = [{'clon': int(a), 'clat': int(b), 'n': int(v)}
                                           for (a, b), v in c2.head(20).items()]
        df_in = inb
    except Exception as e:
        out['province_error'] = f'{type(e).__name__}: {e}'
        df_in = df

    # ---------- 4. 最近邻距离 ----------
    out['nn_all_bbox'] = nn_stats(df.lon.values, df.lat.values)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
    out['nn_200_random'] = nn_stats(df.lon.values[idx], df.lat.values[idx])
    if len(df_in) and 'name' in getattr(df_in, 'columns', []):
        out['nn_all_inside_china'] = nn_stats(df_in.lon.values, df_in.lat.values)

    # ---------- 5. 全球语境：class12 全球 Top20 格 ----------
    gc = glob.groupby([np.floor(glob.lon).astype(int), np.floor(glob.lat).astype(int)]).size().sort_values(
        ascending=False)
    out['global_top20_cells'] = [{'clon': int(a), 'clat': int(b), 'n': int(v)}
                                 for (a, b), v in gc.head(20).items()]
    out['global_n_cells'] = int(gc.shape[0])
    out['global_cell_count_dist'] = {str(k): int(v) for k, v in
                                     gc.value_counts().sort_index().items()}

    # ---------- 6. 与中国格总类直方图的对照 ----------
    h = ext['china_all_class_hist']
    tot = sum(h.values())
    out['china_cells_total_rows'] = int(tot)
    out['china_class_hist_top15'] = [{'class': int(k), 'n': int(v),
                                      'pct': round(100 * v / tot, 4)}
                                     for k, v in list(h.items())[:15]]
    out['china_class12_pct_of_china_rows'] = round(100 * h.get('12', 0) / tot, 4)

    # ---------- 7. 5 个随机点 ----------
    r5 = np.random.default_rng(2026).choice(len(df), size=min(5, len(df)), replace=False)
    out['random5_points'] = [{'lon': round(float(df.lon.iloc[i]), 6),
                              'lat': round(float(df.lat.iloc[i]), 6),
                              'year': int(df.year.iloc[i]),
                              'tile_id': str(df.tile_id.iloc[i])} for i in r5]

    os.makedirs(RESD, exist_ok=True)
    with open(os.path.join(RESD, 'c12_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---------- 控制台摘要 ----------
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('china_class_hist_top15', 'cell_count_dist',
                                   'global_cell_count_dist')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
