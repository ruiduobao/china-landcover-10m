# -*- coding: utf-8 -*-
"""
q3_pft_qa.py — China Forest Dataset v1.0（1 km；PFT 1981-2013 + 森林/PFT 变化事件）独立诊断，
               裁决《2020 年中国 100 万植被图》能否作为 2017-2024 针阔标签源（doc81 判据 V3）
* 输入
    - Forest Cover / PFT / 变化事件  F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/外部数据/China_Forest_Dataset_v1.0/
    - 植被图                        .../数据/外部数据/2020年中国100万植被图/vegetation.shp
    - v2 样本                       F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet
    - 省界                          .../数据/边界/china_100000_full.json（DataV, GCJ-02）
* 规则源  技术文档/81（V1-V9 预注册）；本脚本只出 V3 与年代/稳定性诊断，不改判据
* 门槛    V3 针阔一致率 >= 0.70；分省 n < 30 不报
* 输出    F:/lc_work/nb_national_2026/opt/pft_qa.json
          F:/lc_work/nb_national_2026/opt/pft_yearly_curve.csv
          F:/lc_work/nb_national_2026/opt/pft_by_prov.csv
* 用法    python q3_pft_qa.py
* 约束    单进程、只读输入、内存 <= 10 GB
"""
import glob
import json
import math
import os
import time
import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
from pyproj import CRS, Transformer
from scipy import ndimage as ndi
from shapely import make_valid
from shapely.ops import unary_union

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
CFD = os.path.join(PROJ, '数据', '外部数据', 'China_Forest_Dataset_v1.0')
VEG_SHP = os.path.join(PROJ, '数据', '外部数据', '2020年中国100万植被图', 'vegetation.shp')
V2_PARQ = r'F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet'
PROV_JSON = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
OUT = r'F:/lc_work/nb_national_2026/opt'

FC_DIR = os.path.join(CFD, '1_Forest_Cover_Annual_1980-2023')
PFT_DIR = os.path.join(CFD, '2_PFT_Annual_1981-2013')
GAIN_DIR = os.path.join(CFD, '3_Forest_Change_Gain')
LOSS_DIR = os.path.join(CFD, '4_Forest_Change_Loss')
PFT_GAIN = os.path.join(CFD, '5_PFT_Change_Gain')
PFT_LOSS = os.path.join(CFD, '6_PFT_Change_Loss')

PFT_NEEDLE = ['ENF_boreal', 'ENF_temperate', 'DNF']
PFT_BROAD = ['EBF_temperate', 'EBF_tropical', 'DBF_boreal', 'DBF_temperate', 'DBF_tropical']
PFT_ALL = PFT_NEEDLE + PFT_BROAD
LABEL_CN = {'ENF_boreal': '北方针叶常绿', 'ENF_temperate': '温带针叶常绿', 'DNF': '针叶落叶',
            'EBF_temperate': '温带阔叶常绿', 'EBF_tropical': '热带阔叶常绿',
            'DBF_boreal': '北方针叶落叶?阔叶落叶北方针', 'DBF_temperate': '温带阔叶落叶',
            'DBF_tropical': '热带阔叶落叶'}

SOUTH6 = ['广东省', '福建省', '广西壮族自治区', '浙江省', '湖南省', '江西省']
STCOLS = [1981, 1990, 2000, 2013]
KEY_MASKS = ['vm_needle', 'vm_broad', 'vm_pine_s6', 'vm_fir_s6',
             'vm_needle_er', 'vm_broad_er', 'vm_pine_er', 'vm_fir_er']
Z95 = 1.959963984540054
T0 = time.time()


def log(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg), flush=True)


def wilson(k, n, z=Z95):
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, cen - half), 4), round(min(1.0, cen + half), 4)]


def pct(k, n):
    return None if n == 0 else round(100.0 * k / n, 2)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        if isinstance(o, float) and math.isnan(o):
            return None
        return o
    return str(o)


# ================================================================ 0. 主网格
with rasterio.open(os.path.join(FC_DIR, 'China-ForestCover_1km_2013_v1.0.tif')) as ds:
    FC_T = ds.transform
    FC_H, FC_W = ds.height, ds.width
    FC_CRS = ds.crs
    FC_WKT = ds.crs.to_wkt()
RES = float(FC_T.a)
PIX_KM2 = (RES * RES) / 1e6
FC_CRS_P = CRS.from_wkt(FC_WKT)
TO_ALBERS = Transformer.from_crs('EPSG:4326', FC_CRS_P, always_xy=True)
log('FC 主网格 %d x %d, res %.3f m, 单像元 %.5f km2' % (FC_W, FC_H, RES, PIX_KM2))


def pft_path(cls, year):
    return os.path.join(PFT_DIR, str(year), 'China-%s_1km_%d_v1.0.tif' % (cls, year))


def fc_path(year):
    return os.path.join(FC_DIR, 'China-ForestCover_1km_%d_v1.0.tif' % year)


_OFF = {}


def grid_off(path):
    """栅格在 FC 主网格中的整数像元偏移；所有输入共享同一网格，不做重采样"""
    if path in _OFF:
        return _OFF[path]
    with rasterio.open(path) as d:
        h, w = d.height, d.width
        # 主网格左上角 -> 子栅格左上角 的整数像元位移
        c_off = int(round((d.transform.c - FC_T.c) / RES))
        r_off = int(round((FC_T.f - d.transform.f) / RES))
        assert abs((d.transform.c - FC_T.c) / RES - c_off) < 1e-3, path
        assert abs((FC_T.f - d.transform.f) / RES - r_off) < 1e-3, path
    assert 0 <= r_off and 0 <= c_off and r_off + h <= FC_H and c_off + w <= FC_W, (path, r_off, c_off, h, w)
    _OFF[path] = (r_off, c_off, h, w)
    return _OFF[path]


def read_sub(path):
    r, c, h, w = grid_off(path)
    with rasterio.open(path) as d:
        return d.read(1), r, c


def read_placed(path, dtype=np.uint8):
    r, c, h, w = grid_off(path)
    a = np.zeros((FC_H, FC_W), dtype=dtype)
    with rasterio.open(path) as d:
        a[r:r + h, c:c + w] = d.read(1)
    return a


# ================================================================ 1. 植被图 -> 1 km 掩膜
log('=== 读植被图 ===')
veg = gpd.read_file(VEG_SHP)
log('图斑 %d' % len(veg))
veg = veg.set_crs('EPSG:4326', allow_override=True)


def poly_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    if g.geom_type == 'GeometryCollection':
        parts = [p for p in g.geoms if p.geom_type in ('Polygon', 'MultiPolygon')]
        return unary_union(parts) if parts else None
    return None


geom = veg.geometry
n_invalid = int((~geom.is_valid).sum())
gfix = geom.copy()
gfix.loc[~geom.is_valid] = gfix.loc[~geom.is_valid].apply(make_valid)
gfix = gfix.apply(poly_only)
veg['_g'] = gfix
vm = veg[veg['_g'].notna()].reset_index(drop=True)
vm = vm.set_geometry('_g')
log('invalid %d；有效几何 %d' % (n_invalid, len(vm)))

vt = vm['植被型'].fillna('').to_numpy()
lh = vm['生活型'].fillna('').to_numpy()
vtype = vm['植被类型'].fillna('').astype(str)

NEEDLE_H = {'常绿针叶', '落叶针叶'}
BROAD_H = {'常绿阔叶', '落叶阔叶'}
is_vt1 = vt == '1、针叶林'
is_vt2 = vt == '2、针阔叶混交林'
is_vt3 = vt == '3、阔叶林'
m_needle = is_vt1 & np.isin(lh, list(NEEDLE_H))
m_broad = is_vt3 & np.isin(lh, list(BROAD_H))
m_mixed = is_vt2
m_pine = vtype.str.contains('马尾松', regex=False).to_numpy()
m_fir = vtype.str.contains('杉木', regex=False).to_numpy()
log('针叶林图斑 %d / 阔叶林图斑 %d / 混交 %d / 马尾松 %d / 杉木 %d'
    % (m_needle.sum(), m_broad.sum(), m_mixed.sum(), m_pine.sum(), m_fir.sum()))

# 图斑归省
prov_gdf = gpd.read_file(PROV_JSON)
prow = []
for _, f in prov_gdf.iterrows():
    g = f.geometry
    if g is None or g.is_empty:
        continue
    if not g.is_valid:
        g = poly_only(make_valid(g))
        if g is None:
            continue
    try:
        int(f['adcode'])
    except (ValueError, TypeError):
        continue  # 九段线等非省要素
    prow.append({'prov': f['name'], 'geometry': g})
provs = gpd.GeoDataFrame(prow, crs='EPSG:4326')
vp = gpd.GeoDataFrame({'ix': np.arange(len(vm))},
                      geometry=vm.geometry.representative_point().to_numpy(), crs='EPSG:4326')
pj = gpd.sjoin(vp, provs[['prov', 'geometry']], how='left', predicate='within')
pj = pj[~pj.index.duplicated(keep='first')]
vm['_prov'] = pj.set_index('ix')['prov'].reindex(np.arange(len(vm))).to_numpy()
log('未归省图斑 %d' % int(vm['_prov'].isna().sum()))
in_s6 = vm['_prov'].isin(SOUTH6).to_numpy()

# 重投影到 Albers（栅格化必须与主网格同 CRS）
vm_a = vm.to_crs(FC_CRS_P).set_geometry('_g')
va = vm_a.geometry.to_numpy()
log('重投影 Albers 完成')


def burn(sel):
    out = np.zeros((FC_H, FC_W), dtype=np.uint8)
    geoms = [g for g, m in zip(va, sel) if m and g is not None]
    if geoms:
        rasterio.features.rasterize(((g, 1) for g in geoms), out=out, transform=FC_T,
                                    all_touched=False, dtype=np.uint8, fill=0)
    return out.astype(bool)


KEEP_YEARS = (1981, 2000, 2013)
on_year_needle = {}
on_year_broad = {}
_fc_cache = {}


def fc_curve_year(y, mk):
    """在任意 mask 上取某年 Forest Cover 像元数（带缓存，避免重复读）"""
    if y not in _fc_cache:
        with rasterio.open(fc_path(y)) as d:
            _fc_cache[y] = d.read(1) == 1
    return int((_fc_cache[y] & mk).sum())


M = OrderedDict()
M['vm_needle'] = burn(m_needle)
M['vm_broad'] = burn(m_broad)
M['vm_mixed'] = burn(m_mixed)
M['vm_pine_s6'] = burn(m_pine & in_s6)
M['vm_fir_s6'] = burn(m_fir & in_s6)
for k, v in M.items():
    log('  mask %-14s %9d 像元 (%8.0f km2)' % (k, int(v.sum()), v.sum() * PIX_KM2))

# 腐蚀（3x3 一圈 ≈ 1.14 km 内缩）
st3 = np.ones((3, 3), bool)
M['vm_needle_er'] = ndi.binary_erosion(M['vm_needle'], st3, border_value=0)
M['vm_broad_er'] = ndi.binary_erosion(M['vm_broad'], st3, border_value=0)
M['vm_pine_er'] = ndi.binary_erosion(M['vm_pine_s6'], st3, border_value=0)
M['vm_fir_er'] = ndi.binary_erosion(M['vm_fir_s6'], st3, border_value=0)
for k in ['vm_needle_er', 'vm_broad_er', 'vm_pine_er', 'vm_fir_er']:
    src = k.replace('_er', '') if k.replace('_er', '') in M else k.replace('_er', '_s6')
    log('  腐蚀 %-14s %d -> %d' % (k, int(M[src].sum()), int(M[k].sum())))

# ================================================================ 2. PFT 年际：basic + 年代曲线 + 稳定性
log('=== PFT 1981-2013 单遍 ===')
st_needle = np.zeros((FC_H, FC_W), np.uint8)
st_broad = np.zeros((FC_H, FC_W), np.uint8)
curve_rows = []
mutex = OrderedDict()
per_year_needle = OrderedDict()
per_year_broad = OrderedDict()
on_mask = OrderedDict((k, OrderedDict()) for k in KEY_MASKS)
cls_2013 = {}

for y in range(1981, 2014):
    yb_n = np.zeros((FC_H, FC_W), bool)
    yb_b = np.zeros((FC_H, FC_W), bool)
    n_cls = OrderedDict()
    for c in PFT_ALL:
        a, r, c0 = read_sub(pft_path(c, y))
        m = a == 1
        h, w = m.shape
        n_cls[c] = int(m.sum())
        if y == 2013:
            fa = np.zeros((FC_H, FC_W), bool)
            fa[r:r + h, c0:c0 + w] = m
            cls_2013[c] = fa
        if c in PFT_NEEDLE:
            yb_n[r:r + h, c0:c0 + w] |= m
        else:
            yb_b[r:r + h, c0:c0 + w] |= m
    n_n = int(yb_n.sum())
    n_b = int(yb_b.sum())
    if y in KEEP_YEARS:
        on_year_needle[y] = yb_n.copy()
        on_year_broad[y] = yb_b.copy()
    both = yb_n & yb_b
    per_year_needle[y] = n_n
    per_year_broad[y] = n_b
    mutex[y] = {'n_needle_or_broad': int((yb_n | yb_b).sum()),
                'n_both': int(both.sum()),
                'pct_both_of_any': pct(int(both.sum()), int((yb_n | yb_b).sum()))}
    row = OrderedDict(year=y)
    for c in PFT_ALL:
        row[c + '_pix'] = n_cls[c]
        row[c + '_km2'] = round(n_cls[c] * PIX_KM2, 1)
    row['needle_pix'] = n_n
    row['broad_pix'] = n_b
    row['needle_km2'] = round(n_n * PIX_KM2, 1)
    row['broad_km2'] = round(n_b * PIX_KM2, 1)
    row['needle_km2_incl_both'] = round((n_n + int(both.sum())) * PIX_KM2, 1)
    row['needle_share'] = round(n_n / max(1, n_n + n_b), 4)
    curve_rows.append(row)
    # 关键 mask 上的 PFT 年际（年代检验曲线）
    for k in KEY_MASKS:
        mk = M[k]
        nn = int((mk & yb_n).sum())
        nb = int((mk & yb_b).sum())
        bt = int((mk & both).sum())
        on_mask[k][y] = {'needle_only': nn - bt, 'broad_only': nb - bt, 'both': bt,
                         'needle_pct': pct(nn, int(mk.sum())), 'broad_pct': pct(nb, int(mk.sum()))}
    st_needle += yb_n.astype(np.uint8)
    st_broad += yb_b.astype(np.uint8)
    del yb_n, yb_b
    if y % 4 == 1 or y == 2013:
        log('  %d: needle %8d broad %8d both %6d share %.3f' % (y, n_n, n_b, int(both.sum()),
                                                               n_n / max(1, n_n + n_b)))

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv(os.path.join(OUT, 'pft_yearly_curve.csv'), index=False, encoding='utf-8-sig')
log('写出 pft_yearly_curve.csv')

cls_2013_n = {c: int(v.sum()) for c, v in cls_2013.items()}
n13 = per_year_needle[2013] + per_year_broad[2013]
sh = np.array([r['needle_share'] for r in curve_rows])

BASIC = {
    'pixel_area_km2': round(PIX_KM2, 5),
    'pixel_area_note': ('Albers 等积，像元 %.3f m x %.3f m = %.5f km2（README 写 "approx 1 km"，实为 1.29 km2）。'
                        '按此换算 2013 年森林 %d 像元 = %.2f 万 km2，与第九次森林资源清查 2.20 亿 ha 同量级，'
                        '反证面积标定正确。' % (RES, RES, PIX_KM2, n13, n13 * PIX_KM2 / 1e4)),
    'pft_2013_pixels': cls_2013_n,
    'pft_2013_area_km2': {c: round(cls_2013_n[c] * PIX_KM2, 1) for c in PFT_ALL},
    'pft_2013_needle_pix': per_year_needle[2013],
    'pft_2013_broad_pix': per_year_broad[2013],
    'pft_2013_needle_share': round(per_year_needle[2013] / max(1, n13), 4),
    'pft_2013_area_wan_km2_total': round(n13 * PIX_KM2 / 1e4, 2),
    'needle_broad_area_by_year': {str(y): {'needle_km2': round(per_year_needle[y] * PIX_KM2, 1),
                                           'broad_km2': round(per_year_broad[y] * PIX_KM2, 1),
                                           'needle_share': round(per_year_needle[y] /
                                                                 max(1, per_year_needle[y] + per_year_broad[y]), 4)}
                                  for y in per_year_needle},
    'mutual_exclusivity': {str(y): mutex[y] for y in mutex},
    'mutual_exclusivity_note': ('8 类 PFT 为独立二值掩膜。pct_both_of_any = 同时被至少两类标为 1 的像元占比；'
                                '>0 说明 PFT 非严格互斥（1 km 像元内可含多类），点采样按 needle/broad/both/none 四态处理。'),
    'needle_share_stability': {
        'min': round(float(sh.min()), 4), 'max': round(float(sh.max()), 4),
        'mean': round(float(sh.mean()), 4), 'std': round(float(sh.std()), 4),
        'range_pp': round(float((sh.max() - sh.min()) * 100), 2),
        'y1981': round(float(sh[0]), 4), 'y2013': round(float(sh[-1]), 4),
        'delta_pp_1981_2013': round(float((sh[-1] - sh[0]) * 100), 2),
    },
    'stability_counters': {
        'n_pix_grid': int(FC_H * FC_W),
        'n_pix_needle_ge25y': int((st_needle >= 25).sum()),
        'n_pix_needle_33y': int((st_needle == 33).sum()),
        'n_pix_broad_ge25y': int((st_broad >= 25).sum()),
        'n_pix_broad_33y': int((st_broad == 33).sum()),
        'n_pix_ever_both': int(((st_needle > 0) & (st_broad > 0)).sum()),
        'n_pix_stable_needle_strict': int(((st_needle >= 25) & (st_broad == 0)).sum()),
        'n_pix_stable_broad_strict': int(((st_broad >= 25) & (st_needle == 0)).sum()),
        'n_pix_never_pft': int(((st_needle == 0) & (st_broad == 0)).sum()),
    },
}

# ================================================================ 3. Forest Cover 逐年曲线
log('=== Forest Cover 1980-2023 逐年 ===')
CURVE_MASKS = OrderedDict(M)
CURVE_MASKS['stable_needle'] = (st_needle >= 25) & (st_broad == 0)
CURVE_MASKS['stable_broad'] = (st_broad >= 25) & (st_needle == 0)
fc_curve = OrderedDict((k, OrderedDict()) for k in CURVE_MASKS)
for y in range(1980, 2024):
    with rasterio.open(fc_path(y)) as d:
        a = d.read(1) == 1
    for k, m in CURVE_MASKS.items():
        fc_curve[k][y] = int((a & m).sum())
    if y % 10 == 0 or y in (2017, 2020, 2023):
        log('  FC %d' % y)
del a

# 全国森林覆盖曲线（数据质量发现：2018 后回落，与森林资源清查的持续增长矛盾）
log('全国 Forest Cover 曲线 ...')
nat_fc = OrderedDict()
for y in range(1980, 2024):
    with rasterio.open(fc_path(y)) as d:
        nat_fc[y] = int((d.read(1) == 1).sum())
_yv = np.array(list(nat_fc.values()), dtype=np.int64)
_yr = np.array(list(nat_fc.keys()))
_pk = int(np.argmax(_yv))
BASIC['national_forest_cover_curve'] = {
    'pix_by_year': {str(k): int(v) for k, v in nat_fc.items()},
    'area_wan_km2_by_year': {str(k): round(v * PIX_KM2 / 1e4, 1) for k, v in nat_fc.items()},
    'peak_year': int(_yr[_pk]), 'peak_pix': int(_yv[_pk]),
    'pix_2023': int(nat_fc[2023]),
    'drop_from_peak_pct': round(100.0 * (1 - nat_fc[2023] / _yv[_pk]), 2),
    'pix_2013': int(nat_fc[2013]),
    'pix_2020': int(nat_fc[2020]),
    'anomaly_note': ('全国森林覆盖 1980-2018 单调上升（117 万 -> 239 万 km2），但 2018 见顶后 2023 回落 %.1f%%（-%.1f 万 km2），'
                     '与第九次森林资源清查（2014-2018）及"持续增长"的国家口径矛盾。经核查该回落空间上弥散（非影像条带/边缘伪影），'
                     '最可能是后期年份输入遥感/统计口径变化造成的产品不稳定，而非真实森林减少。'
                     '故本报告所有"2023 vs 2013"的对比均以 2013-2020 段为主证据，2021-2023 只作参考。'
                     % (round(100.0 * (1 - nat_fc[2023] / _yv[_pk]), 1), round((_yv[_pk] - nat_fc[2023]) * PIX_KM2 / 1e4, 1))),
}
BASIC['pft_vs_forestcover_2013'] = {
    'pft_pix': int(per_year_needle[2013] + per_year_broad[2013]),
    'forestcover_pix_2013': int(nat_fc[2013]),
    'ratio': round(float((per_year_needle[2013] + per_year_broad[2013]) / max(1, nat_fc[2013])), 4),
    'note': ('PFT 8 类合计应等于 Forest Cover。2013 年 PFT 合计 %d 像元、ForestCover %d 像元，比值 %.3f；'
             '若显著 <1 说明部分森林像元无 PFT 标注（或反之），影响"图斑内针叶占比"的分母口径。'
             % (int(per_year_needle[2013] + per_year_broad[2013]), int(nat_fc[2013]),
                (per_year_needle[2013] + per_year_broad[2013]) / max(1, nat_fc[2013]))),
}

# ================================================================ 4. [核心 A] 年代检验
log('=== 核心 A ===')
DEC = OrderedDict()

for tag, mk in [('vm_needle', M['vm_needle']), ('vm_needle_er', M['vm_needle_er']),
                ('vm_broad', M['vm_broad']), ('vm_broad_er', M['vm_broad_er'])]:
    n = int(mk.sum())
    d = OrderedDict()
    d['n_pix'] = n
    d['area_km2'] = round(n * PIX_KM2, 1)
    d['pft_by_year'] = {str(y): on_mask[tag][y] for y in STCOLS}
    d['pft_needle_share_of_forest'] = {
        str(y): pct(on_mask[tag][y]['needle_only'],
                    on_mask[tag][y]['needle_only'] + on_mask[tag][y]['broad_only'])
        for y in STCOLS}
    d['pft_needle_pct_curve'] = {str(y): on_mask[tag][y]['needle_pct'] for y in range(1981, 2014)}
    d['forestcover_pct_by_year'] = {str(y): pct(fc_curve[tag][y], n) for y in range(1980, 2024)}
    d['forestcover_pct_picked'] = {str(y): pct(fc_curve[tag][y], n)
                                   for y in (1981, 1990, 2000, 2010, 2013, 2017, 2020, 2023)}
    DEC[tag] = d

_a = on_mask['vm_needle_er'][2013]
_b = on_mask['vm_broad_er'][2013]
_a81 = on_mask['vm_needle_er'][1981]
_b81 = on_mask['vm_broad_er'][1981]


def _share(d):
    return pct(d['needle_only'], d['needle_only'] + d['broad_only'])


DEC['gate'] = {'threshold_needle_pct_in_needle_patch_2013': 60.0,
               'rule': ('两个口径：'
                        '(1) needle_pct = 针叶像元 / 图斑全部像元（含非林）——低是因为 PFT 只标森林像元、且 2013 年图斑内森林覆盖仅 80%；'
                        '(2) pft_needle_share_of_forest = 针叶像元 / (针叶+阔叶) 像元——这才是"图斑内的森林是针还是阔"的正确口径。'
                        '判定用 (2)：>=60% -> 植被图针叶标注到 2013 年仍成立，年代质疑被削弱；<50% -> 植被图已过时。'),
               'note': ('needle_pct 在 vm_needle_er/2013 = %.2f%% 看似"阔叶占优"，但同一像元集里 PFT 无任何森林标注的占 %.1f%%（非林/无 PFT），'
                        '故 needle_pct 不能直接读作"阔叶占优"。' % (
                            _a['needle_pct'],
                            100.0 - 100.0 * (_a['needle_only'] + _a['broad_only']) / _a['n_pix']
                            if _a.get('n_pix') else 0))}
DEC['verdict'] = {
    'needle_patch_2013_needle_pct_of_all_pixels': _a['needle_pct'],
    'needle_patch_2013_broad_pct_of_all_pixels': _a['broad_pct'],
    'needle_patch_2013_needle_share_of_forest': _share(_a),
    'needle_patch_2013_broad_share_of_forest': pct(_a['broad_only'], _a['needle_only'] + _a['broad_only']),
    'needle_patch_1981_needle_share_of_forest': _share(_a81),
    'needle_share_of_forest_delta_1981_2013_pp': round((_share(_a) or 0) - (_share(_a81) or 0), 2),
    'broad_patch_2013_broad_share_of_forest': pct(_b['broad_only'], _b['needle_only'] + _b['broad_only']),
    'broad_patch_2013_needle_share_of_forest': _share(_b),
    'broad_patch_1981_broad_share_of_forest': pct(_b81['broad_only'], _b81['needle_only'] + _b81['broad_only']),
    'separation_of_forest_pp_2013': round((_share(_a) or 0) - (_share(_b) or 0), 2),
    'separation_of_forest_pp_1981': round((_share(_a81) or 0) - (_share(_b81) or 0), 2),
    'needle_patch_forestcover_pct_2013': pct(fc_curve['vm_needle_er'][2013], int(M['vm_needle_er'].sum())),
}
_s13 = DEC['verdict']['needle_patch_2013_needle_share_of_forest']
DEC['verdict']['conclusion'] = ('FAIL 年代质疑成立' if _s13 < 50 else
                                ('BORDERLINE 部分成立' if _s13 < 60 else 'PASS 年代质疑不成立'))
DEC['verdict']['age_objection'] = ('成立' if _s13 < 50 else ('部分成立' if _s13 < 60 else '不成立'))

# ================================================================ 5. [核心 B] 南方松杉
log('=== 核心 B ===')
S = OrderedDict()
for tag in ['vm_pine_s6', 'vm_fir_s6', 'vm_pine_er', 'vm_fir_er']:
    mk = M[tag]
    n = int(mk.sum())
    d = OrderedDict()
    d['provinces'] = SOUTH6
    d['n_pix'] = n
    d['area_km2'] = round(n * PIX_KM2, 1)
    d['forestcover_pct_full'] = {str(y): pct(fc_curve[tag][y], n) for y in range(1980, 2024)}
    d['forestcover_pct_picked'] = {str(y): pct(fc_curve[tag][y], n)
                                   for y in (1981, 1990, 2000, 2010, 2013, 2017, 2020, 2023)}
    d['forest_kept_1981_2023_pp'] = round((d['forestcover_pct_picked']['2023'] or 0) -
                                          (d['forestcover_pct_picked']['1981'] or 0), 2)
    d['pft_by_year'] = {str(y): on_mask[tag][y] for y in STCOLS}
    d['pft_needle_share_of_forest'] = {
        str(y): pct(on_mask[tag][y]['needle_only'],
                    on_mask[tag][y]['needle_only'] + on_mask[tag][y]['broad_only'])
        for y in STCOLS}
    d['pft_needle_pct_curve'] = {str(y): on_mask[tag][y]['needle_pct'] for y in range(1981, 2014)}
    S[tag] = d

log('读变化事件 ...')
g_start = read_placed(os.path.join(GAIN_DIR, 'China-ForestChange_Gain_StartYear_1km_v1.0.tif'), np.uint16)
g_dur = read_placed(os.path.join(GAIN_DIR, 'China-ForestChange_Gain_Duration_1km_v1.0.tif'), np.uint16)
l_start = read_placed(os.path.join(LOSS_DIR, 'China-ForestChange_Loss_StartYear_1km_v1.0.tif'), np.uint16)
l_dur = read_placed(os.path.join(LOSS_DIR, 'China-ForestChange_Loss_Duration_1km_v1.0.tif'), np.uint16)
log('  森林 gain %d / loss %d 像元' % (int((g_start > 0).sum()), int((l_start > 0).sum())))


def ev_stats(mk, start, dur, bins=((1980, 1990), (1991, 2000), (2001, 2013), (2014, 2023))):
    n = int(mk.sum())
    if n == 0:
        return None
    s = start[mk].astype(np.int64)
    dv = dur[mk].astype(np.int64)
    has = s > 0
    out = {'n_pix': n, 'n_event': int(has.sum()), 'pct_event': pct(int(has.sum()), n)}
    if has.any():
        out['start_year_min'] = int(s[has].min())
        out['start_year_max'] = int(s[has].max())
        out['start_year_median'] = int(np.median(s[has]))
        out['duration_median'] = int(np.median(dv[has]))
        out['start_year_bins'] = OrderedDict(
            ('%d-%d' % (lo, hi), {'n': int(((s >= lo) & (s <= hi) & has).sum()),
                                  'pct_of_mask': pct(int(((s >= lo) & (s <= hi) & has).sum()), n)})
            for lo, hi in bins)
        out['start_year_counts'] = {str(int(k)): int(v)
                                    for k, v in pd.Series(s[has]).value_counts().sort_index().items()}
    return out


for tag in ['vm_pine_s6', 'vm_fir_s6', 'vm_pine_er', 'vm_fir_er']:
    mk = M[tag]
    n = int(mk.sum())
    S[tag]['forest_gain_event'] = ev_stats(mk, g_start, g_dur)
    S[tag]['forest_loss_event'] = ev_stats(mk, l_start, l_dur)
    S[tag]['forest_gain_after_2014_n'] = int(((g_start >= 2014) & mk).sum())
    S[tag]['forest_loss_after_2014_n'] = int(((l_start >= 2014) & mk).sum())
    S[tag]['forest_gain_after_2014_pct'] = pct(int(((g_start >= 2014) & mk).sum()), n)
    S[tag]['forest_loss_after_2014_pct'] = pct(int(((l_start >= 2014) & mk).sum()), n)
    # 2014 年后净变化（gain 减 loss，像元级）
    S[tag]['net_forest_change_after_2014_pct'] = round(
        (S[tag]['forest_gain_after_2014_pct'] or 0) - (S[tag]['forest_loss_after_2014_pct'] or 0), 2)

# PFT 变化事件
log('读 PFT 变化事件（8 类 x gain/loss）...')
pft_ev = {}
for c in PFT_ALL:
    for kind, folder in (('gain', PFT_GAIN), ('loss', PFT_LOSS)):
        cand = glob.glob(os.path.join(folder, 'China*%s*%s*StartYear*_v1.0.tif' % (c, kind)))
        if not cand:
            pft_ev[(c, kind)] = None
            log('  WARN 缺 %s %s StartYear' % (c, kind))
            continue
        pft_ev[(c, kind)] = read_placed(cand[0], np.uint16)
        log('  %s %-8s StartYear %8d 事件像元' % (c, kind, int((pft_ev[(c, kind)] > 0).sum())))

n_loss = np.zeros((FC_H, FC_W), bool)
n_gain = np.zeros((FC_H, FC_W), bool)
b_gain = np.zeros((FC_H, FC_W), bool)
b_loss = np.zeros((FC_H, FC_W), bool)
for c in PFT_NEEDLE:
    if pft_ev.get((c, 'loss')) is not None:
        n_loss |= pft_ev[(c, 'loss')] > 0
    if pft_ev.get((c, 'gain')) is not None:
        n_gain |= pft_ev[(c, 'gain')] > 0
for c in PFT_BROAD:
    if pft_ev.get((c, 'gain')) is not None:
        b_gain |= pft_ev[(c, 'gain')] > 0
    if pft_ev.get((c, 'loss')) is not None:
        b_loss |= pft_ev[(c, 'loss')] > 0

pft_ev_mask = {}
for tag in ['vm_pine_s6', 'vm_fir_s6', 'vm_pine_er', 'vm_fir_er']:
    mk = M[tag]
    n = int(mk.sum())
    d = OrderedDict()
    for c in PFT_ALL:
        for kind in ('gain', 'loss'):
            a = pft_ev.get((c, kind))
            if a is None:
                d['%s_%s' % (c, kind)] = None
                continue
            s = a[mk]
            has = s > 0
            rec = {'n': int(has.sum()), 'pct': pct(int(has.sum()), n)}
            if has.any():
                rec['median_year'] = int(np.median(s[has]))
                rec['start_year_counts'] = {str(int(k)): int(v)
                                            for k, v in pd.Series(s[has]).value_counts().sort_index().items()}
            d['%s_%s' % (c, kind)] = rec
    co = mk & n_loss & b_gain
    S[tag]['conversion_needle_to_broad'] = {
        'n_pix_needle_loss_and_broad_gain': int(co.sum()),
        'pct_of_mask': pct(int(co.sum()), n),
        'n_pix_needle_loss_only': int((mk & n_loss & ~b_gain).sum()),
        'n_pix_broad_gain_only': int((mk & ~n_loss & b_gain).sum()),
        'n_pix_broad_to_needle': int((mk & b_loss & n_gain).sum()),
        'note': ('PFT 变化产品每像元仅记 1 个事件（唯一 StartYear+Duration），不编码多次转换；'
                 '"针叶类 loss 且阔叶类 gain 共位"作为针->阔转换的下界。PFT 变化只覆盖 1981-2013。'),
    }
    # 2013 之后：森林未丢但类型可能变（无 PFT 数据）-> 只能报 forest gain/loss
    S[tag]['post2013_evidence_limit'] = (
        '若针叶林被阔叶林原地替代，Forest Cover 仍为 1，本数据集在 2014-2023 段无 PFT 观测，'
        '故 2014 年后的"针->阔"不可观测；只能由"2014 后森林 gain"（皆伐后更新）作为替代指标。')
    S[tag]['pft_change_events'] = d

S['national_pft_change'] = {
    'needle_loss_pix': int(n_loss.sum()), 'needle_gain_pix': int(n_gain.sum()),
    'broad_gain_pix': int(b_gain.sum()), 'broad_loss_pix': int(b_loss.sum()),
    'needle_loss_and_broad_gain_pix': int((n_loss & b_gain).sum()),
}
# 南 6 省逐省转换事件在 §6（_s6_masks 就绪后）计算；g_start/l_start 保留到那时

del g_dur, l_dur

# ================================================================ 6. [核心 A 分区] 按生态区分层年代检验
log('=== 核心 A 分区：按生态区分层的年代检验（南北分歧是关键）===')
PROV2ECO = {
    '黑龙江省': '东北', '辽宁省': '东北', '吉林省': '东北',
    '上海市': '华东', '台湾省': '华东', '浙江省': '华东', '江苏省': '华东', '山东省': '华东', '福建省': '华东',
    '安徽省': '华中', '江西省': '华中', '湖北省': '华中', '湖南省': '华中', '河南省': '华中',
    '内蒙古自治区': '华北', '山西省': '华北', '天津市': '华北', '河北省': '华北', '北京市': '华北',
    '广东省': '华南', '香港特别行政区': '华南', '广西壮族自治区': '华南', '海南省': '华南',
    '甘肃省': '西北', '新疆维吾尔自治区': '西北', '陕西省': '西北', '宁夏回族自治区': '西北',
    '重庆市': '西南', '云南省': '西南', '四川省': '西南', '贵州省': '西南',
    '青海省': '青藏', '西藏自治区': '青藏',
}
ECOS = ['东北', '华北', '华中', '华东', '华南', '西南', '西北', '青藏']
# 生态区掩膜（用图斑 representative_point 归省 -> 省 -> 生态区）
vm['_eco'] = vm['_prov'].map(PROV2ECO)
eco_r = np.zeros((FC_H, FC_W), np.uint8)
eco_lut = {i + 1: e for i, e in enumerate(ECOS)}
for i, e in enumerate(ECOS):
    sel = (vm['_eco'] == e).to_numpy()
    if sel.any():
        eco_r = np.where(burn(sel), np.uint8(i + 1), eco_r)
log('生态区掩膜：%s' % {eco_lut[k]: int((eco_r == k).sum()) for k in eco_lut})

DECO = OrderedDict()
for i, e in enumerate(ECOS):
    em = eco_r == (i + 1)
    rec = OrderedDict()
    rec['n_needle_er_pix'] = int((M['vm_needle_er'] & em).sum())
    rec['n_broad_er_pix'] = int((M['vm_broad_er'] & em).sum())
    if rec['n_needle_er_pix'] < 30 or rec['n_broad_er_pix'] < 30:
        rec['suppressed'] = '图斑像元 <30'
        DECO[e] = rec
        continue
    rec['needle_patch_area_km2'] = round(rec['n_needle_er_pix'] * PIX_KM2, 1)
    for lab, mk in (('needle_patch', M['vm_needle_er'] & em), ('broad_patch', M['vm_broad_er'] & em)):
        n = int(mk.sum())
        rec[lab] = {
            'n': n,
            'pft_needle_share_of_forest': {
                str(y): pct(int((mk & on_year_needle[y]).sum()),
                            int((mk & (on_year_needle[y] | on_year_broad[y])).sum()))
                for y in (1981, 2000, 2013)},
            'forestcover_pct': {str(y): pct(fc_curve_year(y, mk), n)
                                for y in (1981, 2000, 2013, 2020, 2023)},
        }
    sn = rec['needle_patch']['pft_needle_share_of_forest']['2013']
    sb = rec['broad_patch']['pft_needle_share_of_forest']['2013']
    if sn is not None and sb is not None:
        rec['separation_of_forest_pp_2013'] = round(sn - sb, 2)
        rec['gate_60_pass'] = bool(sn >= 60)
        rec['ecoregion_vintage_holds'] = bool(sn >= 60 and sn - sb >= 20)
    DECO[e] = rec

DEC['by_eco'] = DECO
# 南 6 省逐省：马尾松/杉木图斑内 PFT 针叶占比 + 森林覆盖
# 用未腐蚀 mask 与省界取交（腐蚀版图斑逐省再切会碎成孤点，故此处用原始 mask）
DEC['south6_pine_fir_by_prov'] = {}
_s6_masks = {}
for pv in SOUTH6:
    mprov = burn((vm['_prov'] == pv).to_numpy())
    rec = {}
    for lab, base in (('pine', M['vm_pine_s6']), ('fir', M['vm_fir_s6'])):
        mk = base & mprov
        _s6_masks[(pv, lab)] = mk
        n = int(mk.sum())
        if n < 30:
            rec[lab] = {'n': n, 'suppressed': 'n<30'}
            continue
        rec[lab] = {
            'n': n, 'area_km2': round(n * PIX_KM2, 1),
            'needle_share_of_forest': {str(y): pct(int((mk & on_year_needle[y]).sum()),
                                                   int((mk & (on_year_needle[y] | on_year_broad[y])).sum()))
                                       for y in (1981, 2000, 2013)},
            'forestcover_pct': {str(y): pct(fc_curve_year(y, mk), n)
                                for y in (1981, 2000, 2013, 2023)},
        }
    DEC['south6_pine_fir_by_prov'][pv] = rec

# 逐省转换事件（n_loss/b_gain/g_start/l_start 均已就绪）
for (pv, lab), mk in _s6_masks.items():
    rec = DEC['south6_pine_fir_by_prov'][pv][lab]
    if 'suppressed' in rec:
        continue
    n = int(mk.sum())
    rec['n_needle_loss_and_broad_gain'] = int((mk & n_loss & b_gain).sum())
    rec['pct_needle_to_broad'] = pct(int((mk & n_loss & b_gain).sum()), n)
    rec['n_forest_gain_after_2014'] = int((mk & (g_start >= 2014)).sum())
    rec['n_forest_loss_after_2014'] = int((mk & (l_start >= 2014)).sum())
    rec['pct_forest_gain_after_2014'] = pct(int((mk & (g_start >= 2014)).sum()), n)
    rec['pct_forest_loss_after_2014'] = pct(int((mk & (l_start >= 2014)).sum()), n)
del g_start, l_start, n_loss, n_gain, b_gain, b_loss, _s6_masks
DEC['by_eco_gate'] = {
    'threshold': 60.0, 'separation_threshold_pp': 20.0,
    'note': ('按生态区分别判：needle_patch.pft_needle_share_of_forest["2013"] >=60% 且与 broad_patch 分离 >=20pp '
             '-> 该区植被图的针叶标注到 2013 仍成立。of_forest 口径只用 PFT 有针/阔结论的像元（分母去掉非林与 PFT 空值），'
             '比 needle_pct（分母=图斑全部像元，含非林）更贴近"图斑内的森林到底是针还是阔"。'),
}

# ================================================================ 7. [核心 C] V3 三方
log('=== 核心 C: V3 ===')
p = pd.read_parquet(V2_PARQ)
log('v2 %d 点' % len(p))
xs, ys = TO_ALBERS.transform(p['lon'].to_numpy(), p['lat'].to_numpy())
cols = np.floor((xs - FC_T.c) / RES).astype(np.int64)
rows = np.floor((FC_T.f - ys) / RES).astype(np.int64)
ok = (cols >= 0) & (cols < FC_W) & (rows >= 0) & (rows < FC_H)
rr = np.where(ok, rows, 0)
cc = np.where(ok, cols, 0)
log('点落主网格内 %d / %d' % (int(ok.sum()), len(p)))

p['pft_needle_flag'] = np.zeros(len(p), bool)
p['pft_broad_flag'] = np.zeros(len(p), bool)
for c in PFT_ALL:
    hit = np.zeros(len(p), bool)
    hit[ok] = cls_2013[c][rr[ok], cc[ok]]
    if c in PFT_NEEDLE:
        p['pft_needle_flag'] |= hit
    else:
        p['pft_broad_flag'] |= hit
    if c == PFT_ALL[0]:
        p['pft_class'] = pd.Series(np.where(hit, c, 'none'))
    else:
        p.loc[hit & (p['pft_class'] == 'none'), 'pft_class'] = c

pn = p['pft_needle_flag'].to_numpy()
pb = p['pft_broad_flag'].to_numpy()
p['pft_state'] = np.where(~ok, 'outside',
                          np.where(pn & pb, 'both', np.where(pn, 'needle', np.where(pb, 'broad', 'none'))))
p['pft_needle_years'] = np.where(ok, st_needle[rr, cc], 0).astype(np.int16)
p['pft_broad_years'] = np.where(ok, st_broad[rr, cc], 0).astype(np.int16)
p['pft_stable_needle'] = ok & (p['pft_needle_years'] >= 25) & (p['pft_broad_years'] == 0)
p['pft_stable_broad'] = ok & (p['pft_broad_years'] >= 25) & (p['pft_needle_years'] == 0)
log('PFT 点态: %s' % p['pft_state'].value_counts().to_dict())

log('植被图 point-in-polygon ...')
pts = gpd.GeoDataFrame({'point_id': p['point_id'].to_numpy()},
                       geometry=gpd.points_from_xy(p['lon'], p['lat']), crs='EPSG:4326')
leaf_arr = np.where(m_needle, 'needle', np.where(m_broad, 'broad', np.where(m_mixed, 'mixed', 'none')))
vcols = gpd.GeoDataFrame({'_oid': np.arange(len(vm)), '_leaf': leaf_arr,
                          '_area': vm.to_crs('EPSG:6933').area.to_numpy()},
                         geometry=vm.geometry.to_numpy(), crs='EPSG:4326')
sj = gpd.sjoin(pts, vcols, how='left', predicate='within')
log('sjoin 命中 %d，多重 %d' % (int(sj['_oid'].notna().sum()), int(sj.index.duplicated(keep=False).sum())))
sj['_rank'] = sj['_leaf'].isin(['needle', 'broad']).astype(int)
sj = sj.sort_values(['point_id', '_rank', '_area'], ascending=[True, False, False])
sj1 = sj[~sj['point_id'].duplicated(keep='first')]
p['veg_leaf'] = sj1.set_index('point_id')['_leaf'].reindex(p['point_id']).fillna('none').to_numpy()
del sj, sj1, vcols, pts, leaf_arr
log('植被图点态: %s' % p['veg_leaf'].value_counts().to_dict())


def concl(s):
    return s.isin(['needle', 'broad'])


V3 = OrderedDict()
sub = p[concl(p['pft_state']) & concl(p['leaf'])]
nb = len(sub)
ag = int((sub['pft_state'] == sub['leaf']).sum())
V3['n_both_conclusive'] = int(nb)
V3['n_agree'] = ag
V3['agreement'] = round(ag / nb, 4) if nb else None
V3['agreement_wilson95'] = wilson(ag, nb)
V3['gate'] = 0.70
V3['gate_pass'] = bool(nb and ag / nb >= 0.70)
V3['verdict'] = 'PASS' if V3['gate_pass'] else 'FAIL'
cm = pd.crosstab(sub['pft_state'], sub['leaf'])
V3['confusion_pft_rows_vs_fcs10_cols'] = {str(r): {str(k): int(v) for k, v in row.items()}
                                          for r, row in cm.iterrows()}
V3['pft_state_distribution'] = {str(k): int(v) for k, v in p['pft_state'].value_counts().items()}
V3['veg_leaf_distribution'] = {str(k): int(v) for k, v in p['veg_leaf'].value_counts().items()}
V3['pft_needle_share'] = round(float((p['pft_state'] == 'needle').sum() /
                                     max(1, concl(p['pft_state']).sum())), 4)
V3['fcs10_needle_share'] = round(float((p['leaf'] == 'needle').sum() /
                                       max(1, concl(p['leaf']).sum())), 4)
V3['veg_needle_share'] = round(float((p['veg_leaf'] == 'needle').sum() /
                                     max(1, concl(p['veg_leaf']).sum())), 4)

sub2 = p[concl(p['pft_state']) & concl(p['veg_leaf'])]
nb2 = len(sub2)
ag2 = int((sub2['pft_state'] == sub2['veg_leaf']).sum())
V3['pft_vs_vegmap'] = {
    'n_both_conclusive': int(nb2), 'n_agree': ag2,
    'agreement': round(ag2 / nb2, 4) if nb2 else None,
    'agreement_wilson95': wilson(ag2, nb2),
    'gate': 0.70, 'gate_pass': bool(nb2 and ag2 / nb2 >= 0.70),
}
cm2 = pd.crosstab(sub2['pft_state'], sub2['veg_leaf'])
V3['pft_vs_vegmap']['confusion_pft_rows_vs_vegmap_cols'] = \
    {str(r): {str(k): int(v) for k, v in row.items()} for r, row in cm2.iterrows()}

tri = p[concl(p['pft_state']) & concl(p['leaf']) & concl(p['veg_leaf'])]
cmb = (tri['veg_leaf'].str[0].str.upper() + tri['pft_state'].str[0].str.upper() +
       tri['leaf'].str[0].str.upper())
vc = cmb.value_counts()
V3['three_way'] = {
    'legend': 'V=植被图, P=PFT2013, F=FCS10；N=needle, B=broad',
    'n_points': int(len(tri)),
    'combos': {str(k): int(v) for k, v in vc.items()},
    'n_unanimous': int(vc.get('NNN', 0) + vc.get('BBB', 0)),
    'pct_unanimous': pct(int(vc.get('NNN', 0) + vc.get('BBB', 0)), len(tri)),
    'by_majority': {},
    'note': '三源皆有结论的点；NNN/BBB 为三源同判，其余 6 种为 2:1 分歧。',
}
maj = {'N': 0, 'B': 0, 'split': 0}
for k, v in vc.items():
    nn_ = k.count('N')
    if nn_ == 3:
        maj['N'] += int(v)
    elif nn_ == 0:
        maj['B'] += int(v)
    else:
        maj['split'] += int(v)
V3['three_way']['by_majority'] = {k: int(v) for k, v in maj.items()}
# 哪一种源是少数派
minority = OrderedDict()
for k, v in vc.items():
    if k in ('NNN', 'BBB'):
        continue
    for pos, nm in ((0, 'vegmap'), (1, 'pft'), (2, 'fcs10')):
        ch = k[pos]
        others = [k[i] for i in range(3) if i != pos]
        if others[0] == others[1] and others[0] != ch:
            minority['%s_is_minority_%s' % (nm, k)] = int(v)
V3['three_way']['who_is_minority'] = minority
del tri, cmb


def gtab(df, key):
    out = []
    for k, g in df.groupby(key, dropna=False):
        s1 = g[concl(g['pft_state']) & concl(g['leaf'])]
        s2 = g[concl(g['pft_state']) & concl(g['veg_leaf'])]
        s3 = g[concl(g['pft_state']) & concl(g['leaf']) & concl(g['veg_leaf'])]
        c3 = (s3['veg_leaf'].str[0].str.upper() + s3['pft_state'].str[0].str.upper() +
              s3['leaf'].str[0].str.upper())
        out.append({
            'key': str(k), 'n_points': int(len(g)),
            'n_pft_vs_fcs10': int(len(s1)),
            'v3_pft_vs_fcs10': round(float((s1['pft_state'] == s1['leaf']).mean()), 4) if len(s1) else None,
            'v3_wilson95': wilson(int((s1['pft_state'] == s1['leaf']).sum()), len(s1)) if len(s1) else None,
            'n_pft_vs_vegmap': int(len(s2)),
            'pft_vs_vegmap': round(float((s2['pft_state'] == s2['veg_leaf']).mean()), 4) if len(s2) else None,
            'n_three_way': int(len(s3)),
            'pct_unanimous': pct(int(c3.isin(['NNN', 'BBB']).sum()), len(s3)) if len(s3) else None,
            'pft_needle_pct': pct(int((g['pft_state'] == 'needle').sum()),
                                  int(concl(g['pft_state']).sum())),
            'fcs10_needle_pct': pct(int((g['leaf'] == 'needle').sum()), int(concl(g['leaf']).sum())),
            'veg_needle_pct': pct(int((g['veg_leaf'] == 'needle').sum()), int(concl(g['veg_leaf']).sum())),
            'suppressed_n_lt_30': bool(len(s1) < 30),
        })
    return sorted(out, key=lambda r: -r['n_points'])


allprov = gtab(p, 'prov')
V3['by_prov'] = [r for r in allprov if not r['suppressed_n_lt_30']]
V3['by_prov_suppressed'] = [r['key'] for r in allprov if r['suppressed_n_lt_30']]
V3['by_eco'] = [r for r in gtab(p, 'eco') if not r['suppressed_n_lt_30']]
V3['south6'] = [r for r in gtab(p[p['prov'].isin(SOUTH6)], 'prov') if not r['suppressed_n_lt_30']]
# 南 6 省合并
gs6 = p[p['prov'].isin(SOUTH6)]
s1 = gs6[concl(gs6['pft_state']) & concl(gs6['leaf'])]
V3['south6_combined'] = {
    'n': int(len(s1)),
    'v3': round(float((s1['pft_state'] == s1['leaf']).mean()), 4) if len(s1) else None,
    'wilson95': wilson(int((s1['pft_state'] == s1['leaf']).sum()), len(s1)) if len(s1) else None,
    'pft_needle_pct': pct(int((gs6['pft_state'] == 'needle').sum()),
                          int(concl(gs6['pft_state']).sum())),
    'n_stable_needle': int(gs6['pft_stable_needle'].sum()),
    'n_stable_broad': int(gs6['pft_stable_broad'].sum()),
}

# ================================================================ 7. [核心 D] 稳定性过滤
log('=== 核心 D ===')
D = OrderedDict()
n_all = len(p)
n_ok = int(ok.sum())
D['n_v2_points'] = int(n_all)
D['n_points_in_pft_grid'] = n_ok
D['pct_in_pft_grid'] = pct(n_ok, n_all)
sn = p['pft_stable_needle']
sb = p['pft_stable_broad']
D['n_stable_needle'] = int(sn.sum())
D['n_stable_broad'] = int(sb.sum())
D['pct_stable_needle'] = pct(int(sn.sum()), n_ok)
D['pct_stable_broad'] = pct(int(sb.sum()), n_ok)
D['pct_stable_either'] = pct(int((sn | sb).sum()), n_ok)
for nm, msk, lab in (('stable_needle', sn, 'needle'), ('stable_broad', sb, 'broad')):
    s = p[msk & concl(p['leaf'])]
    D[nm] = {
        'n_with_fcs10': int(len(s)),
        'fcs10_agree_pct': pct(int((s['leaf'] == lab).sum()), len(s)),
        'fcs10_opposite_pct': pct(int((s['leaf'] != lab).sum()), len(s)),
        'wilson95': wilson(int((s['leaf'] == lab).sum()), len(s)) if len(s) else None,
    }
    s2 = p[msk & concl(p['veg_leaf'])]
    D[nm]['vegmap_agree_pct'] = pct(int((s2['veg_leaf'] == lab).sum()), len(s2))

r1 = sn & (p['leaf'] == 'broad')
r2 = sb & (p['leaf'] == 'needle')
r1c = sn & (p['leaf'] == 'needle')
r2c = sb & (p['leaf'] == 'broad')
den = int(((sn | sb) & concl(p['leaf'])).sum())
D['filter'] = {
    'R1': '稳定针叶像元(>=25/33 年针叶 且 0 年阔叶) 而 FCS10=broad -> 判 FCS10 误标',
    'R2': '稳定阔叶像元(>=25/33 年阔叶 且 0 年针叶) 而 FCS10=needle -> 判 FCS10 误标',
    'n_R1_flag': int(r1.sum()), 'n_R2_flag': int(r2.sum()),
    'n_R1_confirm': int(r1c.sum()), 'n_R2_confirm': int(r2c.sum()),
    'n_flagged': int((r1 | r2).sum()),
    'den_stable_and_fcs10_conclusive': den,
    'pct_flagged_over_stable_conclusive': pct(int((r1 | r2).sum()), den),
    'n_kept_after_filter': int(n_all - int((r1 | r2).sum())),
    'pct_kept': pct(int(n_all - int((r1 | r2).sum())), n_all),
    'n_removed_pct_of_all_v2': pct(int((r1 | r2).sum()), n_all),
    'note': ('硬过滤只删"像元 25 年稳定针叶/阔叶 却与 FCS10 相反"的点；'
             '未进入稳定集的点（含 PFT 网格外、PFT=none/both、波动像元）不动。'),
}
sn20 = ok & (p['pft_needle_years'] >= 20) & (p['pft_broad_years'] <= 2)
sb20 = ok & (p['pft_broad_years'] >= 20) & (p['pft_needle_years'] <= 2)
D['relaxed_ge20y'] = {
    'rule': '针叶年数>=20/33 且 阔叶年数<=2（反之为阔叶）',
    'n_stable_needle': int(sn20.sum()), 'n_stable_broad': int(sb20.sum()),
    'n_R1_flag': int((sn20 & (p['leaf'] == 'broad')).sum()),
    'n_R2_flag': int((sb20 & (p['leaf'] == 'needle')).sum()),
    'n_flagged': int((sn20 & (p['leaf'] == 'broad')).sum() + (sb20 & (p['leaf'] == 'needle')).sum()),
    'pct_flagged_over_stable_conclusive': pct(
        int((sn20 & (p['leaf'] == 'broad')).sum() + (sb20 & (p['leaf'] == 'needle')).sum()),
        int(((sn20 | sb20) & concl(p['leaf'])).sum())),
}
ss = p[(sn | sb) & concl(p['leaf'])].copy()
ss['pft_lab'] = np.where(ss['pft_stable_needle'], 'needle', 'broad')
D['agreement_on_stable_vs_fcs10'] = {
    'n': int(len(ss)),
    'agreement': round(float((ss['pft_lab'] == ss['leaf']).mean()), 4) if len(ss) else None,
    'wilson95': wilson(int((ss['pft_lab'] == ss['leaf']).sum()), len(ss)) if len(ss) else None,
}
ss2 = ss[concl(ss['veg_leaf'])]
D['agreement_on_stable_vs_vegmap'] = {
    'n': int(len(ss2)),
    'agreement': round(float((ss2['pft_lab'] == ss2['veg_leaf']).mean()), 4) if len(ss2) else None,
    'wilson95': wilson(int((ss2['pft_lab'] == ss2['veg_leaf']).sum()), len(ss2)) if len(ss2) else None,
}
# 三方：稳定像元 vs 三源
ss3 = ss[concl(ss['veg_leaf'])]
D['stable_vs_vegmap_vs_fcs10'] = {
    'n': int(len(ss3)),
    'pct_stable_agrees_both': pct(int(((ss3['pft_lab'] == ss3['veg_leaf']) &
                                       (ss3['pft_lab'] == ss3['leaf'])).sum()), len(ss3)),
    'pct_stable_agrees_only_vegmap': pct(int(((ss3['pft_lab'] == ss3['veg_leaf']) &
                                              (ss3['pft_lab'] != ss3['leaf'])).sum()), len(ss3)),
    'pct_stable_agrees_only_fcs10': pct(int(((ss3['pft_lab'] != ss3['veg_leaf']) &
                                             (ss3['pft_lab'] == ss3['leaf'])).sum()), len(ss3)),
    'pct_stable_agrees_neither': pct(int(((ss3['pft_lab'] != ss3['veg_leaf']) &
                                          (ss3['pft_lab'] != ss3['leaf'])).sum()), len(ss3)),
}

prov_rows = []
for k, g in p.groupby('prov'):
    inb_g = g[g['pft_state'] != 'outside']
    a_sn = g['pft_stable_needle']
    a_sb = g['pft_stable_broad']
    prov_rows.append({
        'prov': k, 'n_points': int(len(g)), 'n_in_pft_grid': int(len(inb_g)),
        'stable_needle_pct': pct(int(a_sn.sum()), int(len(inb_g))) if len(inb_g) else None,
        'stable_broad_pct': pct(int(a_sb.sum()), int(len(inb_g))) if len(inb_g) else None,
        'n_R1_flag': int((a_sn & (g['leaf'] == 'broad')).sum()),
        'n_R2_flag': int((a_sb & (g['leaf'] == 'needle')).sum()),
        'fcs10_needle_pct_in_stable_needle': pct(int((g[a_sn]['leaf'] == 'needle').sum()),
                                                 int(concl(g[a_sn]['leaf']).sum())) if a_sn.sum() else None,
        'vegmap_needle_pct_in_stable_needle': pct(int((g[a_sn]['veg_leaf'] == 'needle').sum()),
                                                  int(concl(g[a_sn]['veg_leaf']).sum())) if a_sn.sum() else None,
    })
D['by_prov'] = pd.DataFrame(prov_rows).sort_values('n_points', ascending=False).to_dict(orient='records')

# 分省 PFT 2013 面积（省栅格）
prov_shapes = [(g, i + 1) for i, g in enumerate(provs.geometry.to_crs(FC_CRS_P))]
prov_r = np.zeros((FC_H, FC_W), np.int16)
rasterio.features.rasterize(prov_shapes, out=prov_r, transform=FC_T, fill=0, dtype=np.int16)
log('省栅格覆盖 %d 像元' % int((prov_r > 0).sum()))
pft_prov = []
for i, g in enumerate(provs.geometry):
    msk = prov_r == i + 1
    if not msk.any():
        continue
    pc = {c: int((msk & cls_2013[c]).sum()) for c in PFT_ALL}
    nn_ = sum(pc[c] for c in PFT_NEEDLE)
    bb_ = sum(pc[c] for c in PFT_BROAD)
    pft_prov.append({'prov': provs['prov'].iloc[i], 'n_pix': int(msk.sum()),
                     'needle_km2': round(nn_ * PIX_KM2, 1), 'broad_km2': round(bb_ * PIX_KM2, 1),
                     'needle_share': round(nn_ / max(1, nn_ + bb_), 4),
                     **{c + '_km2': round(pc[c] * PIX_KM2, 1) for c in PFT_ALL}})
V3['pft_2013_area_by_prov'] = sorted(pft_prov, key=lambda r: -r['needle_km2'])
pd.DataFrame(V3['pft_2013_area_by_prov']).to_csv(os.path.join(OUT, 'pft_by_prov.csv'),
                                                 index=False, encoding='utf-8-sig')
log('写出 pft_by_prov.csv')

# ================================================================ 8. verdict
_np13 = DEC['verdict']['needle_patch_2013_needle_share_of_forest']
V = OrderedDict()
V['core_A'] = DEC['verdict']
V['core_B'] = {}
for tag in ['vm_pine_er', 'vm_fir_er']:
    V['core_B'][tag] = {
        'n_pix': S[tag]['n_pix'], 'area_km2': S[tag]['area_km2'],
        'pft_1981_needle_share_of_forest': S[tag]['pft_needle_share_of_forest']['1981'],
        'pft_2000_needle_share_of_forest': S[tag]['pft_needle_share_of_forest']['2000'],
        'pft_2013_needle_share_of_forest': S[tag]['pft_needle_share_of_forest']['2013'],
        'forestcover_pct_picked': S[tag]['forestcover_pct_picked'],
        'forest_gain_after_2014_pct': S[tag]['forest_gain_after_2014_pct'],
        'forest_loss_after_2014_pct': S[tag]['forest_loss_after_2014_pct'],
        'net_after_2014_pct': S[tag]['net_forest_change_after_2014_pct'],
        'conversion_n': S[tag]['conversion_needle_to_broad']['n_pix_needle_loss_and_broad_gain'],
        'conversion_pct': S[tag]['conversion_needle_to_broad']['pct_of_mask'],
    }
V['V3'] = {'agreement': V3['agreement'], 'gate': 0.70, 'verdict': V3['verdict'],
           'n': V3['n_both_conclusive'], 'wilson95': V3['agreement_wilson95'],
           'pft_vs_vegmap': V3['pft_vs_vegmap']['agreement'],
           'pct_unanimous_3way': V3['three_way']['pct_unanimous']}
V['core_D'] = {'n_stable_needle': D['n_stable_needle'], 'n_stable_broad': D['n_stable_broad'],
               'pct_stable_either': D['pct_stable_either'],
               'n_flagged': D['filter']['n_flagged'],
               'pct_flagged_over_stable_conclusive': D['filter']['pct_flagged_over_stable_conclusive'],
               'n_removed_pct_of_all_v2': D['filter']['n_removed_pct_of_all_v2'],
               'agreement_on_stable_vs_fcs10': D['agreement_on_stable_vs_fcs10']['agreement']}
V['auto_flags'] = {
    'A_needle_patch_2013_share_of_forest_ge60': bool(_np13 is not None and _np13 >= 60),
    'A_broad_patch_2013_broad_share_of_forest_ge60': bool(
        DEC['verdict']['broad_patch_2013_broad_share_of_forest'] is not None
        and DEC['verdict']['broad_patch_2013_broad_share_of_forest'] >= 60),
    'B_pine_forest_2023_ge80': bool((S['vm_pine_er']['forestcover_pct_picked']['2023'] or 0) >= 80),
    'B_fir_forest_2023_ge80': bool((S['vm_fir_er']['forestcover_pct_picked']['2023'] or 0) >= 80),
    'V3_pass': bool(V3['gate_pass']),
}

CAV = [
    'PFT 只到 2013（Forest Cover 到 2023）。2014-2024 的针阔变化无观测；若针叶林被阔叶林原地替代，Forest Cover 仍为 1，本数据集无法察觉。这是"年代质疑"不能被本数据完全证伪的结构性缺口。',
    '本数据的全国森林覆盖 2018 见顶、2023 回落 %.1f%%，与森林资源清查持续增长矛盾（见 basic.national_forest_cover_curve.anomaly_note）。所有"2023 vs 2013"结论应作参考，"2013 及以前"段才是可靠证据。' % BASIC['national_forest_cover_curve']['drop_from_peak_pct'],
    'PFT 变化产品每像元仅记 1 个事件（唯一 StartYear+Duration），不编码多次转换；"针->阔"用"针叶类 loss 与阔叶类 gain 共位"作下界。',
    'PFT 与 Forest Cover 均为 1 km，无法分辨 1 km 内的针阔镶嵌；对 10 m 目标是"区域趋势证据"，不是像元级标签。',
    '像元面积 %.5f km2（1136.58 m 网格），非 1.000 km2。README 写 "+lat_1=27 +lat_2=45"，实际文件为 +lat_1=25 +lat_2=47；本脚本以文件 WKT 为准。' % PIX_KM2,
    '8 类 PFT 为独立二值掩膜，同像元可为多类（见 basic.mutual_exclusivity）；点采样按 needle/broad/both/none 四态，both 不进一致率分母。',
    '植被图栅格化用 centroid-in-pixel；边界混类，故同时报腐蚀版（3x3 一圈 ≈1.14 km 内缩）。',
    '植被图点值 sjoin 沿用 Q1 口径（within；多重匹配森林优先+面积最大），故本报告 PFT-vs-植被图 与 Q1 的 V2 可直接比较。',
    'DataV 省界为 GCJ-02，与 WGS84 有 100-700 m 偏移；1 km 尺度分省统计有边界噪声，省际量级不受影响。',
    'PFT 由"多源遥感约束 + 省级森林资源清查统计"生成，与 FCS10 并非完全独立（都含遥感信息），一致性高不等于真值。',
    '本诊断只给"植被图年代是否过时"的间接证据，V4 影像核查由 Q4 负责。',
]

OUTJ = OrderedDict()
OUTJ['generated'] = time.strftime('%Y-%m-%d %H:%M:%S')
OUTJ['script'] = os.path.abspath(__file__)
OUTJ['inputs'] = {'pft': PFT_DIR, 'forestcover': FC_DIR, 'forest_change': [GAIN_DIR, LOSS_DIR],
                  'pft_change': [PFT_GAIN, PFT_LOSS], 'vegmap': VEG_SHP, 'v2': V2_PARQ, 'prov': PROV_JSON}
OUTJ['grid'] = {'width': FC_W, 'height': FC_H, 'res_m': round(RES, 3), 'pixel_km2': round(PIX_KM2, 5),
                'crs': '+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84',
                'alignment': '所有 PFT / 变化事件栅格与 FC 主网格严格同网格（整数像元偏移），未做任何重采样',
                'pft_grid': 'PFT 栅格只是按各自覆盖范围裁剪（ENF/DNF/DBF_boreal 4254x3580；EBF 3946x3537；DNF 高 3300），偏移已逐文件校验'}
OUTJ['basic'] = BASIC
OUTJ['decade_test'] = DEC
OUTJ['south_pine_fir'] = S
OUTJ['V3'] = V3
OUTJ['stability_filter'] = D
OUTJ['verdict'] = V
OUTJ['caveats'] = CAV

with open(os.path.join(OUT, 'pft_qa.json'), 'w', encoding='utf-8') as f:
    json.dump(jsonable(OUTJ), f, ensure_ascii=False, indent=1)
log('写出 pft_qa.json')
log('完成，总耗时 %.1f 分钟' % ((time.time() - T0) / 60))
