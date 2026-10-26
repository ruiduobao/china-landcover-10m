# -*- coding: utf-8 -*-
"""s0_common.py — 针阔两类样本生产的共享工具（路径 / 常量 / DEM 地形 / 省归属 / 排除索引）
* 输入：无（纯函数与常量）
* 规则源：技术文档/76（G1–G10）；技术文档/75（坡向伪影、FCS10 纠偏）
* 门槛：无（不含判据执行）
* 输出：被 s1/s2/s3 import 的常量与函数
* 用法：from s0_common import ...
"""
import json
import os

import numpy as np

WORK = r'F:/lc_work/nb_national_2026'
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
LAB = os.path.join(WORK, 'labels')
CODE = os.path.join(LAB, 'code')

# ---------- 输入 ----------
FCS10_TILES = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10'
FCS30D_TILES = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs30d'
YEARLY = r'F:/lc_work/年度子集_含稀有类'
DEM1K = r'F:/lc_work/v31_exp/data/eco_gate/_dem/srtm1000_chn.tif'
DEM60_DIR = r'F:/lc_work/v31_exp/data/eco_gate/_dem'
PROV_GEOJSON = os.path.join(PROJ, '数据/边界/china_100000_full.json')
VAL_POOL = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2.parquet')
VAL_HOLDOUT_JSON = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2_holdout_blocks.json')
MISALIGNED = os.path.join(PROJ, '数据/本地处理/全国清洗训练/错位样本剔除清单.parquet')
EVAL_YEARLY_DIR = r'F:/lc_work/v31_exp/data/yearly'

# ---------- 常量 ----------
TILE_DEG = 2.0
GRID_X0, GRID_Y0 = 73.0, 18.0      # 与 代码/8.全国生产 prod_conf 一致的 2° 网格原点
DECLUTTER_DEG = 0.01               # 1 km 去聚团格
NEEDLE = (71, 72, 81, 82)
BROAD = (51, 52, 61, 62)
MIXED = (91, 92)
FOREST10 = NEEDLE + BROAD + MIXED
YEARS = list(range(2017, 2025))

# 24 类（v31_map）中代表"森林"的码：4 常绿阔叶 / 5 落叶阔叶 / 6 常绿针叶 / 7 落叶针叶 / 8 混交
V31_FOREST = (4, 5, 6, 7, 8)

# FCS10 原始码 → 组（非森林按 FCS10 官方语义归并，仅作诊断与统计）
FCS10_GROUPS = {}
for c in NEEDLE:
    FCS10_GROUPS[c] = 'needle'
for c in BROAD:
    FCS10_GROUPS[c] = 'broad'
for c in MIXED:
    FCS10_GROUPS[c] = 'mixed'
for c in (10, 11, 12, 20):
    FCS10_GROUPS[c] = 'crop'
for c in (120, 121, 122):
    FCS10_GROUPS[c] = 'shrub'
FCS10_GROUPS[130] = 'grass'
FCS10_GROUPS[140] = 'lichen_moss'
FCS10_GROUPS[150] = 'sparse'
for c in (181, 182, 183, 184, 185, 186, 187):
    FCS10_GROUPS[c] = 'wetland'
for c in (190, 191, 192):
    FCS10_GROUPS[c] = 'impervious'
for c in (200, 201, 202):
    FCS10_GROUPS[c] = 'bare'
for c in (210, 215):
    FCS10_GROUPS[c] = 'water'
for c in (220,):
    FCS10_GROUPS[c] = 'snow'

# FCS10 原始码 → 中文名（官方图例原义）
FCS10_NAMES = {
    10: '旱地(雨养)', 11: '草本作物', 12: '乔灌园地', 20: '灌溉耕地',
    51: '常绿阔叶林(郁闭)', 52: '常绿阔叶林(疏)', 61: '落叶阔叶林(郁闭)', 62: '落叶阔叶林(疏)',
    71: '常绿针叶林(郁闭)', 72: '常绿针叶林(疏)', 81: '落叶针叶林(郁闭)', 82: '落叶针叶林(疏)',
    91: '针阔混交林(郁闭)', 92: '针阔混交林(疏)',
    120: '灌丛', 121: '常绿灌丛', 122: '落叶灌丛',
    130: '草地', 140: '地衣苔藓', 150: '稀疏植被',
    181: '木本沼泽', 182: '草本沼泽', 183: '洪泛滩地', 184: '盐渍地', 185: '红树林', 186: '盐沼', 187: '潮滩',
    190: '不透水面', 191: '不透水面(类)', 192: '不透水面(类2)',
    200: '裸地', 201: '裸地(类)', 202: '裸地(类2)',
    210: '水体', 215: '水体(类)', 220: '永久冰雪',
}

# FCS30D 原始码 → 叶型轴（仅用于"与旧标签源不一致率"统计；不用于 FCS10 映射）
def fcs30d_leaf(c):
    if c in (51, 52, 61, 62):
        return 'broad'
    if c in (71, 72, 81, 82):
        return 'needle'
    if c in (91, 92):
        return 'mixed'
    return None


def tile_of(lon, lat):
    ix = np.floor((np.asarray(lon) - GRID_X0) / TILE_DEG).astype(int)
    iy = np.floor((np.asarray(lat) - GRID_Y0) / TILE_DEG).astype(int)
    return np.array(['T%02d%02d' % (a, b) for a, b in zip(ix, iy)])


def tile_bounds(tile):
    ix = int(tile[1:3]); iy = int(tile[3:5])
    return GRID_X0 + ix * TILE_DEG, GRID_Y0 + iy * TILE_DEG, \
           GRID_X0 + (ix + 1) * TILE_DEG, GRID_Y0 + (iy + 1) * TILE_DEG


def class2_of(code):
    if code in NEEDLE:
        return 31
    if code in BROAD:
        return 32
    if code in MIXED:
        return 33
    return 0


def leaf_of(code):
    if code in NEEDLE:
        return 'needle'
    if code in BROAD:
        return 'broad'
    if code in MIXED:
        return 'mixed'
    return None


# ---------- 地形（1 km SRTM，correct ESRI branch） ----------
_DEM_CACHE = {}


def dem_arrays(kind='1k'):
    """返回 (slope_deg, aspect_esri_deg, elev, transform, nodata_mask)。1k 结果缓存到 npy。"""
    if kind in _DEM_CACHE:
        return _DEM_CACHE[kind]
    if kind == '1k':
        slope_fp = os.path.join(LAB, '_dem1k_slope.npy')
        asp_fp = os.path.join(LAB, '_dem1k_aspect.npy')
        elev_fp = os.path.join(LAB, '_dem1k_elev.npy')
        if os.path.isfile(slope_fp) and os.path.isfile(asp_fp):
            import rasterio
            with rasterio.open(DEM1K) as s:
                tfm, crs, w, h = s.transform, s.crs, s.width, s.height
            slope = np.load(slope_fp)
            asp = np.load(asp_fp)
            elev = np.load(elev_fp) if os.path.isfile(elev_fp) else None
            _DEM_CACHE[kind] = (slope, asp, elev, tfm, crs, w, h)
            return _DEM_CACHE[kind]
        import rasterio
        with rasterio.open(DEM1K) as s:
            z = s.read(1).astype('float32')
            tfm, crs, w, h = s.transform, s.crs, s.width, s.height
        z[z <= -30000] = np.nan
        slope, asp = _terrain(z, tfm)
        np.save(slope_fp, slope); np.save(asp_fp, asp); np.save(elev_fp, z)
        _DEM_CACHE[kind] = (slope, asp, z, tfm, crs, w, h)
        return _DEM_CACHE[kind]
    raise ValueError(kind)


def _terrain(z, tfm):
    """坡度（度）与坡向（ESRI 约定 0=北顺时针）。z 已置 nodata 为 NaN。

    坡向 ESRI 分支（见 代码/13.针阔可分性攻关/topo_corr_lib.py）：
      math_asp = arctan2(dz_dy, -dz_dx)（0=东逆时针）
      esri = where(math>90, 450-math, 90-math) % 360
    """
    h, w = z.shape
    dlat = abs(tfm.e); dlon = abs(tfm.a)
    lat = tfm.f - dlat * (np.arange(h) + 0.5)
    dy = np.full((h, 1), dlat * 110574.0, 'float32')
    dx = (dlon * 111320.0 * np.cos(np.deg2rad(lat))).astype('float32')[:, None]
    zf = np.where(np.isfinite(z), z, np.nan)
    # np.gradient 不支持 NaN，先以 0 填充再在边缘置 NaN
    zz = np.nan_to_num(zf, nan=0.0)
    gy, gx = np.gradient(zz)
    gy = -gy / dy            # 向北为正
    gx = gx / dx
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    math_asp = np.degrees(np.arctan2(gy, -gx)) % 360.0
    asp = np.where(math_asp > 90.0, 450.0 - math_asp, 90.0 - math_asp) % 360.0
    bad = ~np.isfinite(zf)
    slope[bad] = np.nan
    asp[bad] = np.nan
    asp[slope < 1e-6] = np.nan
    return slope.astype('float32'), asp.astype('float32')


def sample_raster(arr, tfm, lon, lat, w, h):
    """最近邻取值（原生索引，不插值）。返回 (vals, ok)。"""
    lon = np.asarray(lon, dtype='float64'); lat = np.asarray(lat, dtype='float64')
    a = tfm.a; e = tfm.e; c = tfm.c; f = tfm.f
    col = np.floor((lon - c) / a).astype('int64')
    row = np.floor((lat - f) / e).astype('int64')
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.full(len(lon), np.nan, dtype='float32')
    if ok.any():
        out[ok] = arr[row[ok], col[ok]]
    return out, ok


def aspect_band(asp, slope, min_slope=5.0):
    """阳坡=135–225°，阴坡=315–360∪0–45°；slope<=min_slope 记 'flat'。"""
    out = np.full(len(asp), 'flat', dtype=object)
    ok = np.isfinite(asp) & np.isfinite(slope) & (slope > min_slope)
    sunny = ok & (asp >= 135) & (asp <= 225)
    shady = ok & ((asp >= 315) | (asp <= 45))
    out[shady] = 'shady'
    out[sunny] = 'sunny'
    out[ok & ~sunny & ~shady] = 'other'
    return out


def aspect_or(df, cls_col='leaf', asp_col='aspect_band', sunny='sunny', shady='shady'):
    """needle vs broad 的阴阳坡 OR（needle 为 case）。返回 dict(OR, table)。"""
    d = df[df[asp_col].isin([sunny, shady]) & df[cls_col].isin(['needle', 'broad'])]
    tab = np.zeros((2, 2), dtype='int64')
    for i, c in enumerate(['needle', 'broad']):
        for j, a in enumerate([sunny, shady]):
            tab[i, j] = int(((d[cls_col] == c) & (d[asp_col] == a)).sum())
    n_sunny_needle, n_shady_needle = tab[0, 0], tab[0, 1]
    n_sunny_broad, n_shady_broad = tab[1, 0], tab[1, 1]
    if min(n_sunny_needle, n_shady_needle, n_sunny_broad, n_shady_broad) == 0:
        orv = float('inf') if (n_sunny_broad * n_shady_needle) > 0 else float('nan')
    else:
        orv = (n_sunny_needle * n_shady_broad) / (n_shady_needle * n_sunny_broad)
    return {
        'n': int(len(d)), 'or': float(orv),
        'needle_sunny': n_sunny_needle, 'needle_shady': n_shady_needle,
        'broad_sunny': n_sunny_broad, 'broad_shady': n_shady_broad,
        'needle_shady_frac': (n_shady_needle / (n_sunny_needle + n_shady_needle)) if (n_sunny_needle + n_shady_needle) else None,
        'broad_shady_frac': (n_shady_broad / (n_sunny_broad + n_shady_broad)) if (n_sunny_broad + n_shady_broad) else None,
    }


def mantel_haenszel_or(df, strata_cols, cls_col='leaf', asp_col='aspect_band'):
    """分层（Mantel–Haenszel）OR。doc75 §3.5：不分层会因地理混杂抵消真实对立。"""
    d = df[df[asp_col].isin(['sunny', 'shady']) & df[cls_col].isin(['needle', 'broad'])].copy()
    d['_sunny'] = (d[asp_col] == 'sunny').astype(int)
    d['_needle'] = (d[cls_col] == 'needle').astype(int)
    num = den = 0.0
    used = 0
    for key, g in d.groupby(strata_cols):
        a = int(((g._needle == 1) & (g._sunny == 1)).sum())
        b = int(((g._needle == 1) & (g._sunny == 0)).sum())
        c_ = int(((g._needle == 0) & (g._sunny == 1)).sum())
        dd = int(((g._needle == 0) & (g._sunny == 0)).sum())
        n = a + b + c_ + dd
        if n == 0:
            continue
        used += 1
        num += a * dd / n
        den += b * c_ / n
    return {'mh_or': (num / den) if den > 0 else float('nan'), 'strata_used': used, 'n': int(len(d))}


# ---------- 排除索引 ----------
def build_exclusion_positions():
    """位置键排除集合：validation_pool_v2 + 五窗留出(全部年份) + 错位清单。返回 (set, dict)。"""
    import glob
    import pandas as pd
    pos = set()
    counts = {}
    v = pd.read_parquet(VAL_POOL, columns=['lon', 'lat'])
    s = set(zip(np.round(v.lon.values, 6), np.round(v.lat.values, 6)))
    pos |= s; counts['validation_pool_v2'] = len(s)
    w = set()
    for fp in glob.glob(os.path.join(EVAL_YEARLY_DIR, 'eval_*_w*.parquet')):
        d = pd.read_parquet(fp, columns=['lon', 'lat'])
        w |= set(zip(np.round(d.lon.values, 6), np.round(d.lat.values, 6)))
    pos |= w; counts['five_windows'] = len(w)
    m = pd.read_parquet(MISALIGNED, columns=['lon', 'lat'])
    s2 = set(zip(np.round(m.lon.values, 6), np.round(m.lat.values, 6)))
    pos |= s2; counts['misaligned'] = len(s2)
    counts['union'] = len(pos)
    return pos, counts


def pos_keys(lon, lat):
    return list(zip(np.round(np.asarray(lon, dtype='float64'), 6),
                    np.round(np.asarray(lat, dtype='float64'), 6)))


# ---------- 省归属 ----------
_PROV = {}


def prov_index():
    if 'feats' in _PROV:
        return _PROV['feats']
    from shapely import contains_xy
    from shapely.geometry import shape
    d = json.load(open(PROV_GEOJSON, encoding='utf-8'))
    feats = [(f['properties'].get('name', ''), shape(f['geometry'])) for f in d['features']]
    _PROV['feats'] = feats
    return feats


def province_of(lon, lat):
    from shapely import contains_xy
    lon = np.asarray(lon, dtype='float64'); lat = np.asarray(lat, dtype='float64')
    out = np.full(len(lon), '', dtype=object)
    todo = np.ones(len(lon), dtype=bool)
    for name, geom in prov_index():
        if not name:
            continue
        m = contains_xy(geom, lon, lat)
        hit = m & todo
        out[hit] = name
        todo &= ~m
        if not todo.any():
            break
    return out
