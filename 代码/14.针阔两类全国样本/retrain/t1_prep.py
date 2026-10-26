# -*- coding: utf-8 -*-
"""t1_prep.py — 针阔两类重训：构建两臂样本表（FCS10 标签 + AEF64 + 30m 坡向 + 省归属）
* 输入（只读）：
    AEF64 + v2 池标签：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet
    GLC_FCS10-2023：   F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/{tile}_fcs10.tif（原生 10m）
    NASADEM 30m：      E:/DEM/NASA_DEM/NASADEM+UD20250222/NASADEM_30m_RD2020_UD20250221+COG.tif
    省级边界：         F:/BaiduSyncdisk/中国人口密度2000-2026/3.图表/Figure_人口居住掩膜/data/2026/省级/T2026年初省级.shp
    规范：             技术文档/76_全国针阔两类样本生产实施规范.md
    码表：             exp/two_class_retrain/needle_broad_map.json（只读，冻结）
* 规则源：
    叶型只按 FCS10 官方图例原义合并（51/52=EBF, 61/62=DBF, 71/72=ENF, 81/82=DNF, 91/92=MIX）；
    禁止 FCS30D 奇偶交叉映射用于 FCS10。
    类码一律"取值"（nearest 索引），不重采样、不插值。
    坡向用 terrain_lib（ESRI 约定）：阴=[315,360)∪[0,45]，阳=[135,225]，slope>5°。
* 门槛：
    两臂同点集（ArmA / ArmB 标签不同、行相同）
    ArmA = 旧池 24 类按 from_ours24 合并（4/5→32, 6/7→31, 8→33）
    ArmB = 旧池非森林类原样；旧池森林点的**森林去向**（31/32/33 或非森林）全部按 FCS10
    ArmB 无法定标（FCS10 缺失/未映射）的旧森林点 → 标 -1，训练与评估均剔除并计数
* 输出：exp/two_class_retrain/points_76.parquet（全 76 瓦池内点）
        exp/two_class_retrain/t1_prep_stats.json
* 用法：python t1_prep.py [--force]
"""
import os, sys, json, time, glob, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
NB = r'F:/lc_work/nb_sep_2026'
sys.path.insert(0, os.path.join(NB, 'aspect'))
import terrain_lib as TL

POOL = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
EXT = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10'
DEM = r'E:/DEM/NASA_DEM/NASADEM+UD20250222/NASADEM_30m_RD2020_UD20250221+COG.tif'
PROV_SHP = r'F:/BaiduSyncdisk/中国人口密度2000-2026/3.图表/Figure_人口居住掩膜/data/2026/省级/T2026年初省级.shp'
MAP = os.path.join(HERE, 'needle_broad_map.json')

AEF = [f'A{i:02d}' for i in range(64)]
# 9 个优先瓦片（任务指定）
PRIORITY = {
    'T1505': ('川南', '西南', '四川省', 103, 28, 105, 30),
    'T1406': ('川西', '西南', '四川省', 101, 30, 103, 32),
    'T1606': ('川东', '西南', '四川省', 105, 30, 107, 32),
    'T2002': ('粤中', '南方', '广东省', 113, 22, 115, 24),
    'T1902': ('粤北', '南方', '广东省', 111, 22, 113, 24),
    'T2813': ('黑小兴安岭', '北方', '黑龙江省', 129, 44, 131, 46),
    'T2712': ('黑吉长白', '北方', '黑龙江省', 127, 42, 129, 44),
    'T1609': ('宁贺兰南', '西北', '宁夏回族自治区', 105, 36, 107, 38),
    'T2204': ('闽北', '南方', '福建省', 117, 26, 119, 28),
}


def load_map():
    with open(MAP, encoding='utf-8') as f:
        return json.load(f)


def fcs10_lut(m):
    """FCS10 原始码 → 新码（0=未映射/无效）"""
    lut = np.zeros(256, dtype=np.int16)
    for new, codes in m['from_fcs10_raw']['forest'].items():
        for c in codes:
            lut[int(c)] = int(new)
    for k, v in m['from_fcs10_raw']['nonforest'].items():
        lut[int(k)] = int(v)
    return lut


def ours24_lut(m):
    lut = np.full(32, -1, dtype=np.int16)
    for old, new in m['from_ours24'].items():
        if old.startswith('_') or old == 'identity':
            continue
        lut[int(old)] = int(new)
    for c in m['from_ours24']['identity']:
        lut[int(c)] = int(c)
    return lut


def tile_of(lon, lat):
    ix = (np.floor(lon - 73.0).astype(np.int64)) // 2
    iy = (np.floor(lat - 18.0).astype(np.int64)) // 2
    return np.array(['T%02d%02d' % (a, b) for a, b in zip(ix, iy)])


def sample_native(fp, lons, lats):
    """在原生像元网格上按索引取值（nearest，不重采样）"""
    out = np.full(len(lons), -1, dtype=np.int32)
    with rasterio.open(fp) as s:
        rows, cols = rasterio.transform.rowcol(s.transform, lons, lats)
        rows = np.asarray(rows); cols = np.asarray(cols)
        ok = (rows >= 0) & (cols >= 0) & (rows < s.height) & (cols < s.width)
        if ok.any():
            r0, r1 = int(rows[ok].min()), int(rows[ok].max())
            c0, c1 = int(cols[ok].min()), int(cols[ok].max())
            sub = s.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            out[ok] = sub[rows[ok] - r0, cols[ok] - c0]
    return out


def terrain_block(lon0, lat0, lon1, lat1, pad=0.02, agg=2):
    """读 NASADEM 30m 窗口（2×2 均值聚合到 60m，与 doc75 地形单元同口径）
       → (aspect, slope, xs, ys)，配 terrain_lib（ESRI 约定）"""
    with rasterio.open(DEM) as s:
        w = s.window(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
        w = w.round_offsets().round_lengths()
        dem = s.read(1, window=w).astype(np.float32)
        res = abs(s.transform[0])
        row_off, col_off = int(w.row_off), int(w.col_off)
        H, W = dem.shape
        xs = s.transform[2] + (col_off + np.arange(W) + 0.5) * s.transform[0]
        ys = s.transform[5] + (row_off + np.arange(H) + 0.5) * s.transform[4]
    dem = np.where(dem <= -32000, np.nan, dem)
    if np.isnan(dem).any():
        # 3x3 邻域均值填充（向量化；generic_filter 在 7200² 上逐像元回调会卡死）
        from scipy.ndimage import uniform_filter
        msk = np.isnan(dem)
        v = np.where(msk, 0.0, dem).astype(np.float32)
        cnt = uniform_filter((~msk).astype(np.float32), size=3, mode='nearest')
        sm = uniform_filter(v, size=3, mode='nearest')
        dem[msk] = (sm / np.maximum(cnt, 1e-6))[msk]
    if agg > 1:
        H2, W2 = (H // agg) * agg, (W // agg) * agg
        dem = dem[:H2, :W2].reshape(H2 // agg, agg, W2 // agg, agg).mean(axis=(1, 3))
        res = res * agg
        xs = xs[:W2].reshape(-1, agg).mean(axis=1)
        ys = ys[:H2].reshape(-1, agg).mean(axis=1)
    lat_c = float((lat0 + lat1) / 2)
    asp, slp = TL.aspect_slope(dem, res, lat_c)
    return asp, slp, xs[1:-1], ys[1:-1]


def assign_province(df):
    """CTAmap 2026 省界空间连接（点在面内）。输出 province 列（缺省 None）"""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        print('  !! geopandas 不可用，省归属留空:', e)
        df['province'] = None
        return df
    g = gpd.read_file(PROV_SHP)[['省', 'geometry']].to_crs(4326)
    pts = gpd.GeoDataFrame(df[['lon', 'lat']].copy(),
                           geometry=[Point(x, y) for x, y in zip(df.lon, df.lat)],
                           crs=4326)
    j = gpd.sjoin(pts, g, how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    df['province'] = j['省'].reindex(df.index).values
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    out = os.path.join(HERE, 'points_76.parquet')
    if os.path.exists(out) and not a.force:
        print('已存在', out, '（--force 重算）'); return

    m = load_map()
    lut_f = fcs10_lut(m)
    lut_o = ours24_lut(m)
    t0 = time.time()

    tiles = [os.path.basename(p).replace('_fcs10.tif', '')
             for p in glob.glob(os.path.join(EXT, '*_fcs10.tif'))]
    tiles = sorted(tiles)
    print('FCS10 瓦片', len(tiles))

    df = pd.read_parquet(POOL, columns=['point_id', 'lon', 'lat', 'cl', 'src'] + AEF)
    df['tile'] = tile_of(df.lon.values, df.lat.values)
    keep = df.tile.isin(tiles).values
    df = df[keep].reset_index(drop=True)
    print('池内落入 76 瓦的点:', len(df))

    # ---- 逐瓦：FCS10 取样 + 地形 ----
    asp = np.full(len(df), np.nan, dtype=np.float32)
    slp = np.full(len(df), np.nan, dtype=np.float32)
    f10 = np.full(len(df), 0, dtype=np.int32)
    for ti, tile in enumerate(sorted(df.tile.unique())):
        sel = (df.tile == tile).values
        idx = np.where(sel)[0]
        lon = df.lon.values[idx]; lat = df.lat.values[idx]
        f10[idx] = sample_native(os.path.join(EXT, tile + '_fcs10.tif'), lon, lat)
        # 瓦片 box（用于 NASADEM 窗口）
        ix = int(tile[1:3]); iy = int(tile[3:5])
        lon0, lat0 = 73 + 2 * ix, 18 + 2 * iy
        lon1, lat1 = lon0 + 2, lat0 + 2
        A, S, xs, ys = terrain_block(lon0, lat0, lon1, lat1)
        ri = np.abs(lat[:, None] - ys[None, :]).argmin(axis=1)
        ci = np.abs(lon[:, None] - xs[None, :]).argmin(axis=1)
        asp[idx] = A[ri, ci]
        slp[idx] = S[ri, ci]
        if ti % 10 == 0:
            print('  地形/FCS10 %d/%d %s (%.0fs)' % (ti + 1, len(df.tile.unique()), tile,
                                                     time.time() - t0), flush=True)
    df['fcs10_raw'] = f10
    df['aspect'] = asp
    df['slope'] = slp
    df['asp_cls'] = TL.aspect_class(asp)

    # ---- 标签 ----
    cl = df.cl.values.astype(np.int64)
    df['cl_newA'] = lut_o[np.clip(cl, 0, 31)]
    fc = lut_f[np.clip(df.fcs10_raw.values, 0, 255)].copy()
    fc[df.fcs10_raw.values < 0] = -1
    df['cl_fcs10'] = fc
    is_old_forest = np.isin(cl, [4, 5, 6, 7, 8])
    # ArmB：旧森林点全部按 FCS10 定；FCS10 缺失/未映射 → -1
    clb = df.cl_newA.values.copy()
    clb[is_old_forest] = fc[is_old_forest]
    df['cl_newB'] = clb
    # 森林细分（只作属性，不参与分类）：needle/broad/mix 参照
    df['ref_nb'] = np.where(np.isin(fc, [31]), 1,
                            np.where(np.isin(fc, [32]), 2,
                                     np.where(fc == 33, 3, 0))).astype(np.int8)
    df['ours_leaf'] = np.where(np.isin(cl, [6, 7]), 1,
                               np.where(np.isin(cl, [4, 5]), 2,
                                        np.where(cl == 8, 3, 0))).astype(np.int8)
    df['region'] = df.tile.map({t: v[1] for t, v in PRIORITY.items()}).fillna('其它')
    df['tile_name'] = df.tile.map({t: v[0] for t, v in PRIORITY.items()}).fillna('')
    df['is_priority'] = df.tile.isin(list(PRIORITY)).astype(np.int8)

    df = assign_province(df)

    df.to_parquet(out, index=False)
    # ---- 统计 ----
    st = {
        'n': int(len(df)), 'tiles': int(df.tile.nunique()),
        'tiles_with_points': sorted(df.tile.unique().tolist()),
        'fcs10_missing_pct': round(float((df.fcs10_raw < 0).mean() * 100), 3),
        'fcs10_unmapped_pct': round(float(((df.fcs10_raw > 0) & (df.cl_fcs10 < 0)).mean() * 100), 3),
        'armA_dist': {str(k): int(v) for k, v in df.cl_newA.value_counts().sort_index().items()},
        'armB_dist': {str(k): int(v) for k, v in df.cl_newB.value_counts().sort_index().items() if k > 0},
        'armB_unlabeled': int((df.cl_newB < 0).sum()),
        'old_forest_n': int(is_old_forest.sum()),
        'old_forest_fcs10_nonforest': int((is_old_forest & (fc > 0) & ~np.isin(fc, [31, 32, 33])).sum()),
        'old_forest_fcs10_mixed': int((is_old_forest & (fc == 33)).sum()),
        'old_forest_fcs10_missing': int((is_old_forest & (fc < 0)).sum()),
        'cross_tab_ours_vs_fcs10_forest': {
            k: int(v) for k, v in pd.crosstab(
                pd.Series(cl[is_old_forest], name='ours'),
                pd.Series(fc[is_old_forest], name='fcs10')).stack().items()},
        'province_missing_pct': round(float(df.province.isna().mean() * 100), 3),
        'secs': round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, 't1_prep_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != 'cross_tab_ours_vs_fcs10_forest'},
                     ensure_ascii=False, indent=1))
    print('->', out)


if __name__ == '__main__':
    main()
