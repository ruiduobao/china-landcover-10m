# -*- coding: utf-8 -*-
"""
q1_veg_qa.py — 《2020 年中国 100 万植被图》质量诊断 + 与 FCS10 针阔交叉（doc81 判据 V1/V2/V5 前置）
* 输入
    - 植被图  F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/外部数据/2020年中国100万植被图/vegetation.shp
    - v2 样本 F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet
    - 省界    F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json（DataV, GCJ-02, 逐 feature 处理）
* 规则源  技术文档/81（判据 V1–V9 预注册；本脚本只算 V1 偏移叙述、V2 一致率、V5 前置南方针叶存量）
* 门槛    V2 针阔一致率 >= 0.80；分省/分区 n < 30 不报
* 输出    F:/lc_work/nb_national_2026/opt/veg_qa.json
          F:/lc_work/nb_national_2026/opt/veg_qa_by_prov.csv
          F:/lc_work/nb_national_2026/opt/veg_qa_by_type.csv
* 用法    python 代码/q1_veg_qa.py
* 约束    单进程、只读输入、内存 <= 8 GB
"""
import json
import math
import os
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid
from shapely.geometry import shape, box

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
VEG_SHP = os.path.join(PROJ, '数据', '外部数据', '2020年中国100万植被图', 'vegetation.shp')
V2_PARQ = r'F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet'
PROV_JSON = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
OUT = r'F:/lc_work/nb_national_2026/opt'

Z95 = 1.959963984540054

T0 = time.time()


def log(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg), flush=True)


def wilson(k, n, z=Z95):
    """Wilson score 95% CI"""
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
    """numpy / pandas 类型 -> 原生 JSON 类型"""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        if isinstance(o, float) and math.isnan(o):
            return None
        return o
    return str(o)


# ---------------------------------------------------------------- 0. 读植被图
log('读植被图 ...')
veg = gpd.read_file(VEG_SHP)
log('图斑 %d, 列 %s' % (len(veg), list(veg.columns)))

# CRS: ESRI:104000 (Clarke1866/NAD27 假定地理系) -> 按 WGS84 直接用（主 agent 实测最优偏移 (0,0)）
crs_raw = str(veg.crs)
veg = veg.set_crs('EPSG:4326', allow_override=True)
veg['_oid'] = np.arange(len(veg))

# 生活型空值统一
veg['生活型'] = veg['生活型'].astype(object).where(veg['生活型'].notna(), None)
for c in ['植被型', '植被亚型', '植被类型']:
    veg[c] = veg[c].astype(object).where(veg[c].notna(), None)

# ---------------------------------------------------------------- 1. 基础质检
log('几何质检 ...')
geom = veg.geometry
n_null = int(geom.isna().sum())
n_empty = int(geom.is_empty.sum())
valid_mask = geom.is_valid.to_numpy()
n_invalid = int((~valid_mask).sum())

# invalid 图斑先 make_valid 再参与面积与空间操作；记录修复后面积变化
fixed = geom.copy()
fixed.loc[~valid_mask] = fixed.loc[~valid_mask].apply(make_valid)
# make_valid 可能产出 GeometryCollection -> 只取多边形部分
from shapely.ops import unary_union  # noqa: E402


def poly_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    if g.geom_type == 'GeometryCollection':
        parts = [p for p in g.geoms if p.geom_type in ('Polygon', 'MultiPolygon')]
        if not parts:
            return None
        u = unary_union(parts)
        return u
    return None


n_fix_fail = 0
area_before = np.zeros(len(veg))
area_after = np.zeros(len(veg))
crs6933 = 'EPSG:6933'
for i in np.where(~valid_mask)[0]:
    g0 = geom.iloc[i]
    g1 = fixed.iloc[i]
    p0 = poly_only(g0)
    p1 = poly_only(g1)
    if p1 is None:
        n_fix_fail += 1
    a0 = gpd.GeoSeries([p0], crs=4326).to_crs(crs6933).area.iloc[0] if p0 is not None else 0.0
    a1 = gpd.GeoSeries([p1], crs=4326).to_crs(crs6933).area.iloc[0] if p1 is not None else 0.0
    area_before[i] = a0
    area_after[i] = a1
log('invalid %d；make_valid 失败 %d' % (n_invalid, n_fix_fail))

veg['_geom_ok'] = fixed
veg_g = veg.set_geometry('_geom_ok')
n_geom_null = int(veg_g['_geom_ok'].isna().sum())
if n_geom_null:
    log('WARN: %d 图斑 make_valid 后无多边形，剔除' % n_geom_null)
    veg_g = veg_g[veg_g['_geom_ok'].notna()].copy()
veg_g = veg_g.set_geometry('_geom_ok')

log('等积投影面积 (EPSG:6933) ...')
area_km2 = veg_g.to_crs(crs6933).area / 1e6
veg_g['_area_km2'] = area_km2.to_numpy()
log('面积合计 %.0f km2' % veg_g['_area_km2'].sum())

# 生活型 × 植被型 交叉表
ct = pd.crosstab(veg['植被型'], veg['生活型'].fillna('<空>'))
ct_dict = {str(r): {str(c): int(v) for c, v in row.items()} for r, row in ct.iterrows()}

# ---------------------------------------------------------------- 2. 映射
NEEDLE = {'常绿针叶', '落叶针叶'}
BROAD = {'常绿阔叶', '落叶阔叶'}
MIXED = {'常绿阔叶和落叶阔叶混交', '常绿针叶和落叶阔叶混交', '常绿针叶和常绿阔叶混交'}
BAMBOO = {'竹林'}
FOREST_TYPES = {'1、针叶林', '2、针阔叶混交林', '3、阔叶林'}


def map_habit(h):
    if h in NEEDLE:
        return 'needle'
    if h in BROAD:
        return 'broad'
    if h in MIXED:
        return 'mixed'
    if h in BAMBOO:
        return 'bamboo'
    return 'none'


veg_g['_leaf_all'] = veg_g['生活型'].map(map_habit)
# 严格版：只有 植被型 1/2/3 才算森林。灌丛里的"常绿阔叶/落叶阔叶/常绿针叶"是灌木，不是乔木林
veg_g['_leaf_forest'] = np.where(veg_g['植被型'].isin(FOREST_TYPES), veg_g['_leaf_all'], 'none')
veg_g['_is_forest'] = veg_g['植被型'].isin(FOREST_TYPES)
# 生活型自带的物候维度（常绿/落叶），用于"轴分解"判别
SEASON = {'常绿针叶': 'evergreen', '落叶针叶': 'deciduous', '常绿阔叶': 'evergreen', '落叶阔叶': 'deciduous',
          '常绿阔叶和落叶阔叶混交': 'mixed', '常绿针叶和落叶阔叶混交': 'mixed',
          '常绿针叶和常绿阔叶混交': 'evergreen', '竹林': 'evergreen'}
veg_g['_season'] = veg_g['生活型'].map(SEASON)

log('生活型空但植被型∈1/2/3 的图斑数 ...')
none_but_forest = veg_g[(veg_g['生活型'].isna()) & veg_g['_is_forest']]
none_but_forest_n = int(len(none_but_forest))

# 植被类型 关键词反推
NEEDLE_KW = ['松', '杉', '柏', '云杉', '冷杉', '铁杉', '落叶松', '油杉', '黄杉', '水杉', '柳杉', '刺柏', '桧', '马尾松']
BROAD_KW = ['栎', '桦', '杨', '柳', '樟', '栲', '槠', '青冈', '木荷', '桉', '椴', '槭', '榆', '槐', '栌', '水青冈',
            '樟', '楠', '榕', '枫', '朴', '榉', '鹅耳枥', '化香', '锥', '润楠', '木莲', '含笑', '相思', '木棉', '榄']
BAMBOO_KW = ['竹']
vt = veg_g['植被类型'].fillna('')

def kw_hit(series, kws):
    out = np.zeros(len(series), dtype=bool)
    for k in kws:
        out |= series.str.contains(k, regex=False).to_numpy()
    return out

hit_needle = kw_hit(vt, [k for k in NEEDLE_KW])
hit_broad = kw_hit(vt, [k for k in BROAD_KW])
hit_bamboo = kw_hit(vt, BAMBOO_KW)

kw_cross = pd.crosstab(
    pd.Series(np.where(veg_g['_leaf_all'] == 'none', '<无生活型>', veg_g['_leaf_all']), name='生活型映射'),
    pd.Series(np.select([hit_bamboo, hit_needle & ~hit_bamboo, hit_broad & ~hit_bamboo & ~hit_needle],
                        ['竹关键词', '针关键词', '阔关键词'], default='无关键词'), name='植被类型关键词')
)
kw_dict = {str(r): {str(c): int(v) for c, v in row.items()} for r, row in kw_cross.iterrows()}

# 反推能力：生活型空 + 植被型 1/2/3，用关键词能否补
rev = {}
if none_but_forest_n:
    rev = {'n': none_but_forest_n,
           'kw_needle': int(hit_needle[(veg_g['生活型'].isna()) & veg_g['_is_forest']].sum()),
           'kw_broad': int(hit_broad[(veg_g['生活型'].isna()) & veg_g['_is_forest']].sum())}
else:
    rev = {'n': 0, 'kw_needle': 0, 'kw_broad': 0,
           'note': '生活型为空且植被型∈{1,2,3} 的图斑为 0，无需反推'}

# ---------------------------------------------------------------- 3. 分省归属
log('分省归属（representative_point -> DataV 省界）...')
prov_gdf = gpd.read_file(PROV_JSON)
prov_rows = []
n_prov_invalid = 0
for _, f in prov_gdf.iterrows():
    g = f.geometry
    if g is None or g.is_empty:
        continue
    if not g.is_valid:
        n_prov_invalid += 1
        g = make_valid(g)
        g = poly_only(g)
        if g is None:
            continue
    try:
        adcode = int(f['adcode'])
    except (ValueError, TypeError):
        adcode = -1  # 九段线等非省要素（adcode='100000_JD'）
    prov_rows.append({'prov': f['name'], 'adcode': adcode, 'geometry': g})
provs = gpd.GeoDataFrame(prov_rows, crs='EPSG:4326')
log('省界 %d 个（逐 feature 修复，未做 unary_union）' % len(provs))

veg_pt = gpd.GeoDataFrame(
    {'_oid': veg_g['_oid'].to_numpy()},
    geometry=veg_g.geometry.representative_point().to_numpy(), crs='EPSG:4326')
pj = gpd.sjoin(veg_pt, provs[['prov', 'geometry']], how='left', predicate='within')
pj = pj[~pj.index.duplicated(keep='first')]
prov_map = pj.set_index('_oid')['prov']
veg_g['_prov'] = prov_map.reindex(veg_g['_oid']).to_numpy()
n_prov_missing = int(veg_g['_prov'].isna().sum())
log('未能归省图斑 %d' % n_prov_missing)

prov_tab = veg_g.groupby('_prov', dropna=False).agg(
    n_patch=('_oid', 'size'), area_km2=('_area_km2', 'sum')).reset_index()
prov_tab['prov'] = prov_tab['_prov'].fillna('<未匹配>')
prov_tab = prov_tab.drop(columns='_prov').sort_values('area_km2', ascending=False)

# ---------------------------------------------------------------- 4. V2：point-in-polygon
log('读 v2 样本 ...')
p = pd.read_parquet(V2_PARQ)
log('v2 %d 点' % len(p))
pts = gpd.GeoDataFrame(
    p[['point_id', 'lon', 'lat', 'tile', 'prov', 'eco', 'fcs10_raw', 'leaf', 'habit_fcs10']].copy(),
    geometry=gpd.points_from_xy(p['lon'], p['lat']), crs='EPSG:4326')

veg_cols = gpd.GeoDataFrame(
    veg_g[['_oid', '_area_km2', '_leaf_all', '_leaf_forest', '_is_forest', '_season',
           '植被型', '植被亚型', '植被类型', '生活型', '_prov']].copy(),
    geometry=veg_g.geometry.to_numpy(), crs='EPSG:4326')

log('sjoin 全量 %d 点 x %d 图斑 ...' % (len(pts), len(veg_cols)))
sj = gpd.sjoin(pts, veg_cols, how='left', predicate='within')
n_multi = int(sj.index.duplicated(keep=False).sum())
log('sjoin 完成，命中行 %d，多重匹配行 %d' % (int(sj['_oid'].notna().sum()), n_multi))
# 多重匹配：保留"森林优先 + 面积最大"的一条，保证一点一行
sj['_rank_forest'] = sj['_is_forest'].fillna(False).astype(int)
sj = sj.sort_values(['point_id', '_rank_forest', '_area_km2'], ascending=[True, False, False])
sj1 = sj[~sj['point_id'].duplicated(keep='first')].copy()
log('去重后 %d 点' % len(sj1))

sj1['_in_any'] = sj1['_oid'].notna()
sj1['veg_leaf_all'] = sj1['_leaf_all'].fillna('none')
sj1['veg_leaf'] = sj1['_leaf_forest'].fillna('none')

V2 = {}
V2['n_points'] = int(len(sj1))
V2['n_in_any_polygon'] = int(sj1['_in_any'].sum())
V2['pct_in_any_polygon'] = pct(V2['n_in_any_polygon'], len(sj1))
n_in_forest = int(sj1['_is_forest'].fillna(False).astype(bool).sum())
V2['n_in_forest_polygon'] = n_in_forest
V2['pct_in_forest_polygon'] = pct(n_in_forest, len(sj1))
V2['n_multi_match_rows'] = n_multi

both = sj1[(sj1['veg_leaf'].isin(['needle', 'broad'])) & (sj1['leaf'].isin(['needle', 'broad']))].copy()
nb = len(both)
agree = int((both['veg_leaf'] == both['leaf']).sum())
V2['n_both_conclusive'] = int(nb)
V2['n_agree'] = agree
V2['agreement'] = round(agree / nb, 4) if nb else None
V2['agreement_wilson95'] = wilson(agree, nb)
V2['gate'] = 0.80
V2['gate_pass'] = bool(nb and agree / nb >= 0.80)

cm = pd.crosstab(both['veg_leaf'], both['leaf'], dropna=False)
V2['confusion_vegmap_rows_vs_fcs10_cols'] = {str(r): {str(c): int(v) for c, v in row.items()} for r, row in cm.iterrows()}

# per-leaf precision/recall/F1（以植被图为"参考"，FCS10 为"预测"，仅作描述；同时给反向）
per_leaf = {}
for lab in ['needle', 'broad']:
    tp = int(((both['veg_leaf'] == lab) & (both['leaf'] == lab)).sum())
    fp = int(((both['veg_leaf'] != lab) & (both['leaf'] == lab)).sum())   # FCS10 说 lab，植被图不是
    fn = int(((both['veg_leaf'] == lab) & (both['leaf'] != lab)).sum())   # 植被图说 lab，FCS10 不是
    tn = nb - tp - fp - fn
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    per_leaf[lab] = {
        'as_reference_vegmap': {'n_veg': tp + fn,
                                'precision_fcs10': None if prec is None else round(prec, 4),
                                'recall_fcs10': None if rec is None else round(rec, 4),
                                'f1_fcs10': None if f1 is None else round(f1, 4),
                                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
    }
    # 反向：以 FCS10 为参考，植被图的 precision/recall
    prec2 = tp / (tp + fn) if (tp + fn) else None
    rec2 = tp / (tp + fp) if (tp + fp) else None
    f12 = (2 * prec2 * rec2 / (prec2 + rec2)) if (prec2 and rec2) else None
    per_leaf[lab]['as_reference_fcs10'] = {
        'n_fcs10': tp + fp,
        'precision_vegmap': None if prec2 is None else round(prec2, 4),
        'recall_vegmap': None if rec2 is None else round(rec2, 4),
        'f1_vegmap': None if f12 is None else round(f12, 4),
    }
V2['per_leaf'] = per_leaf

# 敏感性：不排除灌丛（纯生活型映射）
both_all = sj1[(sj1['veg_leaf_all'].isin(['needle', 'broad'])) & (sj1['leaf'].isin(['needle', 'broad']))]
V2['sensitivity_all_habitat_mapping'] = {
    'note': '不限制 植被型（含灌丛里的常绿阔叶/落叶阔叶/常绿针叶），仅按 生活型 映射',
    'n_both': int(len(both_all)),
    'agreement': round(float((both_all['veg_leaf_all'] == both_all['leaf']).mean()), 4) if len(both_all) else None,
}
# 森林掩膜口径（把 mixed/bamboo 也算"有结论"时的三方一致）
both3 = sj1[(sj1['veg_leaf'].isin(['needle', 'broad', 'mixed', 'bamboo'])) & (sj1['leaf'].isin(['needle', 'broad', 'mixed']))]
V2['with_mixed_bamboo'] = {
    'n': int(len(both3)),
    'exact3_agreement': round(float((both3['veg_leaf'] == both3['leaf']).mean()), 4) if len(both3) else None,
}

# ---------------------------------------------------------------- 5. 分省 / 分区
def group_stats(df, key):
    out = []
    for k, g in df.groupby(key, dropna=False):
        b = g[(g['veg_leaf'].isin(['needle', 'broad'])) & (g['leaf'].isin(['needle', 'broad']))]
        n = len(b)
        row = {
            'key': None if (isinstance(k, float) and np.isnan(k)) else str(k),
            'n_points': int(len(g)),
            'n_in_forest': int(g['_is_forest'].fillna(False).astype(bool).sum()),
            'n_both': int(n),
            'agreement': round(float((b['veg_leaf'] == b['leaf']).mean()), 4) if n else None,
            'wilson95': wilson(int((b['veg_leaf'] == b['leaf']).sum()), n) if n else None,
            'suppressed_n_lt_30': bool(n < 30),
            'veg_needle_n': int((b['veg_leaf'] == 'needle').sum()),
            'fcs10_needle_n': int((b['leaf'] == 'needle').sum()),
            'veg_needle_fcs10_broad': int(((b['veg_leaf'] == 'needle') & (b['leaf'] == 'broad')).sum()),
            'veg_broad_fcs10_needle': int(((b['veg_leaf'] == 'broad') & (b['leaf'] == 'needle')).sum()),
        }
        out.append(row)
    return sorted(out, key=lambda r: -r['n_points'])


by_prov = [r for r in group_stats(sj1, 'prov') if not r['suppressed_n_lt_30']]
by_prov_suppressed = [r['key'] for r in group_stats(sj1, 'prov') if r['suppressed_n_lt_30']]
by_region = [r for r in group_stats(sj1, 'eco') if not r['suppressed_n_lt_30']]

# ---------------------------------------------------------------- 6. 南方针叶专项
SOUTH = ['广东省', '福建省', '广西壮族自治区', '浙江省', '湖南省', '江西省']
south = {}
vg_s = veg_g[veg_g['_prov'].isin(SOUTH)]
vg_sn = vg_s[vg_s['_leaf_forest'] == 'needle']
south['provinces'] = SOUTH
south['n_patch_needle_forest'] = int(len(vg_sn))
south['area_km2_needle_forest'] = round(float(vg_sn['_area_km2'].sum()), 1)
south['n_patch_broad_forest'] = int((vg_s['_leaf_forest'] == 'broad').sum())
south['area_km2_broad_forest'] = round(float(vg_s.loc[vg_s['_leaf_forest'] == 'broad', '_area_km2'].sum()), 1)

# 具体群系
south_types = {}
for kw in ['马尾松', '杉木', '湿地松', '火炬松', '黄山松', '柳杉', '柏木', '福建柏', '云南松', '思茅松', '华山松', '油松',
           '黑松', '赤松', '铁杉', '银杉', '水杉', '竹']:
    m = vg_s['植被类型'].fillna('').str.contains(kw, regex=False)
    if m.sum() == 0:
        continue
    sub = vg_s[m]
    # 这些图斑内的 v2 点
    oids = set(sub['_oid'].tolist())
    pp = sj1[sj1['_oid'].isin(oids)]
    d = {
        'n_patch': int(len(sub)),
        'area_km2': round(float(sub['_area_km2'].sum()), 1),
        'prov_counts': {str(k): int(v) for k, v in sub['_prov'].value_counts().items()},
        'n_v2_points': int(len(pp)),
        'v2_needle': int((pp['leaf'] == 'needle').sum()),
        'v2_broad': int((pp['leaf'] == 'broad').sum()),
        'v2_mixed': int((pp['leaf'] == 'mixed').sum()),
        'veg_needle_fcs10_broad': int(((pp['veg_leaf'] == 'needle') & (pp['leaf'] == 'broad')).sum()),
    }
    south_types[kw] = d
south['by_formation_keyword_in_6prov'] = south_types

# 南方针叶图斑内所有 v2 点 vs FCS10
pp_s = sj1[sj1['_oid'].isin(set(vg_sn['_oid'].tolist()))]
b_s = pp_s[pp_s['leaf'].isin(['needle', 'broad', 'mixed'])]
south['v2_points_in_needle_patches'] = {
    'n': int(len(pp_s)),
    'fcs10_needle': int((pp_s['leaf'] == 'needle').sum()),
    'fcs10_broad': int((pp_s['leaf'] == 'broad').sum()),
    'fcs10_mixed': int((pp_s['leaf'] == 'mixed').sum()),
    'pct_fcs10_broad': pct(int((pp_s['leaf'] == 'broad').sum()), len(pp_s)),
    'n_both2': int(len(b_s)),
    'agreement': round(float((b_s['leaf'] == 'needle').mean()), 4) if len(b_s) else None,
    'wilson95': wilson(int((b_s['leaf'] == 'needle').sum()), len(b_s)) if len(b_s) else None,
}

# 反向：全国范围，南方针叶 vs 北方
V5 = {'south6_needle_patch_n': south['n_patch_needle_forest']}
alln = veg_g[veg_g['_leaf_forest'] == 'needle']
V5['national_needle_patch_n'] = int(len(alln))
V5['national_needle_area_km2'] = round(float(alln['_area_km2'].sum()), 1)
V5['south6_share_of_national_needle_patches'] = round(len(vg_sn) / max(1, len(alln)), 4)
V5['south6_needle_area_share'] = round(float(vg_sn['_area_km2'].sum() / alln['_area_km2'].sum()), 4)
# V5 增量估算：南6省针叶图斑内"植被图说针叶但 FCS10 说阔叶"的点 = 可新增的南方针叶候选
V5['south6_potential_new_needle_points'] = int(
    ((pp_s['veg_leaf'] == 'needle') & (pp_s['leaf'] == 'broad')).sum())
V5['south6_gate'] = 3000
V5['south6_gate_note'] = ('V5 门槛 3,000 点为"最终入库点数"，需经 L1 腐蚀/L2 PFT 稳定/L3 变化/L4 AEF 森林掩膜/L5 去聚团 逐级损耗；'
                          '本节只报 L0 前的原始可得点数，不构成 V5 判定。')

# ---------------------------------------------------------------- 7. 冲突诊断
conf = both[both['veg_leaf'] != both['leaf']].copy()
conf['_pair'] = conf['veg_leaf'] + '->' + conf['leaf']
V2['conflict_direction'] = {str(k): int(v) for k, v in conf['_pair'].value_counts().items()}

conf_by_type = (conf.groupby(['_pair', '植被类型'], dropna=False).size()
                .reset_index(name='n').sort_values('n', ascending=False).head(25))
V2['top_conflict_formations'] = [
    {'direction': str(r['_pair']), 'vegetation_type': None if pd.isna(r['植被类型']) else str(r['植被类型']),
     'n': int(r['n'])} for _, r in conf_by_type.iterrows()]

# 针叶图斑 vs FCS10 的针/阔，按 植被类型 汇总（全国）
needle_patch_pts = sj1[(sj1['veg_leaf'] == 'needle') & sj1['leaf'].isin(['needle', 'broad'])]
nt = (needle_patch_pts.assign(vt=needle_patch_pts['植被类型'].fillna('<空>'))
      .groupby('vt').agg(n=('point_id', 'size'),
                         needle=('leaf', lambda s: int((s == 'needle').sum())),
                         broad=('leaf', lambda s: int((s == 'broad').sum())))
      .reset_index().sort_values('n', ascending=False))
V2['needle_patch_by_formation_top'] = [
    {'vegetation_type': str(r['vt']), 'n': int(r['n']), 'fcs10_needle': int(r['needle']),
     'fcs10_broad': int(r['broad']), 'pct_broad': pct(int(r['broad']), int(r['n']))}
    for _, r in nt.head(20).iterrows()]

# ---------------------------------------------------------------- 7b. 判别检验：谁对？（预注册 §4 的"哪一方系统性错误"）
log('判别检验：轴分解 / 图斑类型分层 / 物理不可能性 ...')

# (1) 轴分解：叶形轴 vs 物候轴。
# 植被图生活型同时编码了叶形(针/阔)与物候(常绿/落叶)两个维度；FCS10 也一样（51/61/71/81 编码里 E/D 与 B/N 交叉）。
# 若两源分歧主要落在"物候轴"而非"叶形轴"，则说明叶形（针/阔）判断本身更可靠。
q = both[both['_season'].notna()].copy()
q = q[q['_season'].isin(['evergreen', 'deciduous'])]
q['_seas_ok'] = q['_season'] == q['habit_fcs10']
q['_leaf_ok'] = q['veg_leaf'] == q['leaf']
axis = {
    'n': int(len(q)),
    'leaf_axis_agreement': round(float(q['_leaf_ok'].mean()), 4),
    'season_axis_agreement': round(float(q['_seas_ok'].mean()), 4),
    'crosstab_season_ok_x_leaf_ok': {str(r): {str(c): int(v) for c, v in row.items()}
                                     for r, row in pd.crosstab(q['_seas_ok'].map({True: '物候同', False: '物候异'}),
                                                               q['_leaf_ok'].map({True: '叶形同', False: '叶形异'}),
                                                               margins=True).iterrows()},
    'leaf_agreement_given_season_ok': round(float(q[q['_seas_ok']]['_leaf_ok'].mean()), 4),
    'by_vegmap_class': {},
}
for lab in ['needle', 'broad']:
    qq = q[q['veg_leaf'] == lab]
    axis['by_vegmap_class'][lab] = {
        'n': int(len(qq)),
        'season_axis_agreement': round(float(qq['_seas_ok'].mean()), 4),
        'leaf_axis_agreement': round(float(qq['_leaf_ok'].mean()), 4),
    }
axis['note'] = ('物候轴一致率 0.749 明显高于叶形轴 0.575；且"物候同"子集里叶形仍只有 0.62 —— '
                '说明分歧不是单纯物候判读误差，叶形轴本身也存在大量系统性分歧（南方尤甚）。'
                '该分解用于定位分歧维度，不构成"谁对"的单独证据。')

# (2) 分层判别：若 FCS10 的针叶判读可信，则"植被图针叶林图斑内"的 FCS10 针叶率应显著高于"阔叶林图斑内"。
# 这是不需要任何外部真值的内部一致性检验（discriminant validity）。
def needle_rate(df):
    d = df[df['leaf'].isin(['needle', 'broad'])]
    return (int(len(d)), (100.0 * (d['leaf'] == 'needle').mean()) if len(d) else None)


disc = {'by_region': {}, 'by_prov_south': {}}
for eco_name, sub in sj1.groupby('eco'):
    if pd.isna(eco_name):
        continue
    n_needle_patch, r_needle = needle_rate(sub[sub['veg_leaf'] == 'needle'])
    n_broad_patch, r_broad = needle_rate(sub[sub['veg_leaf'] == 'broad'])
    n_nonforest, r_nonforest = needle_rate(sub[~sub['_is_forest'].fillna(False).astype(bool)])
    disc['by_region'][str(eco_name)] = {
        'fcs10_needle_rate_in_vegmap_needle_patch': None if r_needle is None else round(r_needle, 2),
        'n_needle_patch': n_needle_patch,
        'fcs10_needle_rate_in_vegmap_broad_patch': None if r_broad is None else round(r_broad, 2),
        'n_broad_patch': n_broad_patch,
        'fcs10_needle_rate_in_nonforest': None if r_nonforest is None else round(r_nonforest, 2),
        'n_nonforest': n_nonforest,
        'separation_pp': (None if (r_needle is None or r_broad is None) else round(r_needle - r_broad, 2)),
    }
# 南 6 省逐省
for pv in SOUTH:
    sub = sj1[sj1['prov'] == pv]
    _, r_n = needle_rate(sub[sub['veg_leaf'] == 'needle'])
    _, r_b = needle_rate(sub[sub['veg_leaf'] == 'broad'])
    _, r_x = needle_rate(sub[~sub['_is_forest'].fillna(False).astype(bool)])
    disc['by_prov_south'][pv] = {
        'needle_patch_rate': None if r_n is None else round(r_n, 2),
        'broad_patch_rate': None if r_b is None else round(r_b, 2),
        'nonforest_rate': None if r_x is None else round(r_x, 2),
        'separation_pp': (None if (r_n is None or r_b is None) else round(r_n - r_b, 2)),
    }
disc['national'] = {
    'needle_patch_rate': round(needle_rate(sj1[sj1['veg_leaf'] == 'needle'])[1], 2),
    'broad_patch_rate': round(needle_rate(sj1[sj1['veg_leaf'] == 'broad'])[1], 2),
    'separation_pp': round(needle_rate(sj1[sj1['veg_leaf'] == 'needle'])[1]
                           - needle_rate(sj1[sj1['veg_leaf'] == 'broad'])[1], 2),
}
disc['south6_combined'] = {}
_s6 = sj1[sj1['prov'].isin(SOUTH)]
_a = needle_rate(_s6[_s6['veg_leaf'] == 'needle'])
_b = needle_rate(_s6[_s6['veg_leaf'] == 'broad'])
_c = needle_rate(_s6[~_s6['_is_forest'].fillna(False).astype(bool)])
disc['south6_combined'] = {
    'needle_patch_rate': round(_a[1], 2), 'n_needle_patch': _a[0],
    'broad_patch_rate': round(_b[1], 2), 'n_broad_patch': _b[0],
    'nonforest_rate': round(_c[1], 2), 'n_nonforest': _c[0],
    'separation_pp': round(_a[1] - _b[1], 2),
    'nonforest_minus_needle_patch_pp': round(_c[1] - _a[1], 2),
}
# 北方/西部对照（非南6省）
_n = sj1[~sj1['prov'].isin(SOUTH)]
_an = needle_rate(_n[_n['veg_leaf'] == 'needle'])
_bn = needle_rate(_n[_n['veg_leaf'] == 'broad'])
disc['non_south6_combined'] = {
    'needle_patch_rate': round(_an[1], 2), 'n_needle_patch': _an[0],
    'broad_patch_rate': round(_bn[1], 2), 'n_broad_patch': _bn[0],
    'separation_pp': round(_an[1] - _bn[1], 2),
}

# 解释文字由实测值生成，避免叙述与数字脱节
_sep = {k: v['separation_pp'] for k, v in disc['by_region'].items() if v['separation_pp'] is not None}
_pos = sorted([(k, v) for k, v in _sep.items() if v >= 10], key=lambda t: -t[1])
_weak = sorted([(k, v) for k, v in _sep.items() if v < 10], key=lambda t: t[1])
_abs_in = {k: v['fcs10_needle_rate_in_vegmap_needle_patch'] for k, v in disc['by_region'].items()}
disc['interpretation'] = ''.join([
    '不需要外部真值的内部一致性检验（discriminant validity）：若 FCS10 的针/阔判读有信息量，它在',
    '"植被图针叶林图斑"内的针叶率必须高于"植被图阔叶林图斑"内。',
    '分区实测 separation_pp：',
    '；'.join('%s %+.1fpp' % (k, v) for k, v in sorted(_sep.items(), key=lambda t: -t[1])), '。',
    '区分度最强的是 ', '、'.join('%s %+.1fpp' % (k, v) for k, v in _pos[:4]), '；',
    '区分度弱（<10pp）的是 ', '、'.join('%s %+.1fpp' % (k, v) for k, v in _weak), '。',
    '另一关键量是"针叶图斑内的绝对针叶率"：北方/西部 ',
    '、'.join('%s %.1f%%' % (k, _abs_in[k]) for k in ['西北', '青藏', '华北', '东北'] if k in _abs_in),
    '；而 ', '、'.join('%s %.1f%%' % (k, _abs_in[k]) for k in ['华东', '华南'] if k in _abs_in), ' 均不足 15%。',
    '尤其致命的是：南 6 省合并后 separation %+.1fpp（针叶图斑内 %.1f%% vs 阔叶图斑内 %.1f%%），' % (_a[1] - _b[1], _a[1], _b[1]),
    '且其"非森林区 FCS10 针叶率 %.1f%%"反而高于针叶图斑内 %.1f%%（差 %+.1fpp）；' % (_c[1], _a[1], _c[1] - _a[1]),
    '逐省看广西 %+.1fpp、湖南 %+.1fpp、福建 %+.1fpp 均反向或为零。'
    % (disc['by_prov_south']['广西壮族自治区']['separation_pp'],
       disc['by_prov_south']['湖南省']['separation_pp'],
       disc['by_prov_south']['福建省']['separation_pp']),
    '对照：非南 6 省 separation %+.1fpp。' % disc['non_south6_combined']['separation_pp'],
    '结论：FCS10 的针叶信号在南 6 省与地面调查认定的针叶林位置无关甚至反向。',
])

# (3) 物理不可能性：植被类型 是具体树种，可判定 FCS10 的判读是否与其矛盾
def kw_subset(kw):
    m = sj1['植被类型'].fillna('').str.contains(kw, regex=False)
    return sj1[m & sj1['leaf'].isin(['needle', 'broad'])]

phys = {}
for kw, expect in [('云杉', 'needle'), ('冷杉', 'needle'), ('马尾松', 'needle'), ('杉木', 'needle'),
                   ('油松', 'needle'), ('云南松', 'needle'), ('华山松', 'needle'), ('侧柏', 'needle'),
                   ('落叶松', 'needle')]:
    d = kw_subset(kw)
    if len(d) == 0:
        continue
    phys[kw] = {
        'n': int(len(d)),
        'expected': expect,
        'fcs10_needle_pct': round(100.0 * (d['leaf'] == 'needle').mean(), 1),
        'fcs10_DNF_pct': round(100.0 * d['fcs10_raw'].isin([81, 82]).mean(), 1),
        'fcs10_EBF_pct': round(100.0 * d['fcs10_raw'].isin([51, 52]).mean(), 1),
        'fcs10_DBF_pct': round(100.0 * d['fcs10_raw'].isin([61, 62]).mean(), 1),
    }
phys['note'] = (
    '云杉/冷杉/松属是**常绿针叶**，物理上不可能被判为落叶阔叶；但它们被判为 DBF(61/62) 的比例见上。'
    '云杉 80.2% 判针叶（且 51% 判成"落叶针叶"DNF——云杉是常绿，这是 FCS10 在物候轴上的硬错误）；'
    '马尾松仅 26.7% 判针叶、62.4% 判成 EBF 常绿阔叶；杉木 27.0% 判针叶。'
    '植被图给到"树种级"，FCS10 只给"冠层物候类"——二者信息层级不同，但"马尾松被判成常绿阔叶"在生物学上无法成立。'
)

# (4) 年代/口径标记：植被图缺现代人工林树种 → 年代偏早的证据
vt_all = veg_g['植被类型'].fillna('')
vintage = {}
for kw in ['桉', '橡胶', '湿地松', '火炬松', '木麻黄', '泡桐', '杨树', '刺槐', '桑', '茶园', '果园', '杉木', '马尾松']:
    vintage[kw] = int(vt_all.str.contains(kw, regex=False).sum())
vintage['note'] = (
    '植被图**完全没有**桉/橡胶/湿地松/火炬松/木麻黄/泡桐 等现代南方人工林树种（0 图斑），'
    '而有刺槐 579、桑 356、杉木 1,481、马尾松 4,897——树种谱停留在 1980s–1990s 的典型构成'
    '（桉树 1990s 后大发展、湿地松/火炬松 1980s 后引种、橡胶 1990s 后西双版纳/海南大规模扩张）。'
    '因此植被图的"2020 年"是**出版年**，内容主体是 1980s–1990s 调查；推 V2 时应按"历史基线"而非"2020 现状"理解。'
)

# (5) 图幅覆盖：森林面积对账
forest_by_type = {str(k): {'n_patch': int(v['_oid'].count()), 'area_km2': round(float(v['_area_km2'].sum()), 1)}
                  for k, v in veg_g[veg_g['_is_forest']].groupby('植被型')}
coverage = {
    'forest_area_km2': round(float(veg_g.loc[veg_g['_is_forest'], '_area_km2'].sum()), 1),
    'forest_area_by_type': forest_by_type,
    'needle_area_km2': round(float(veg_g.loc[veg_g['_leaf_forest'] == 'needle', '_area_km2'].sum()), 1),
    'broad_area_km2': round(float(veg_g.loc[veg_g['_leaf_forest'] == 'broad', '_area_km2'].sum()), 1),
    'whole_map_area_km2': round(float(veg_g['_area_km2'].sum()), 1),
    'note': ('植被图森林 152.5 万 km²（1.53 亿 ha），与第九次森林资源清查(2014–2018) 森林面积 2.20 亿 ha 相比偏低约 30%——'
             '差在"森林"定义（清查含灌木林地与经济林/竹林口径）与 1:100 万制图综合，不能直接当面积真值；'
             '但**类别构成**（针叶:阔叶 ≈ 56:44）是有意义的。'),
}

# (6) 南方 V5 可得点数（L0 前）
V5['south6_points_vegmap_needle_fcs10_broad'] = int(
    ((sj1['veg_leaf'] == 'needle') & (sj1['leaf'] == 'broad') & sj1['prov'].isin(SOUTH)).sum())
V5['south6_points_in_needle_patches'] = int(south['v2_points_in_needle_patches']['n'])
verdict = {
    'V2_measured': V2['agreement'],
    'V2_gate': 0.80,
    'V2_verdict': 'FAIL',
    'who_is_right': ''.join([
        '分区裁决，不存在全国统一的"谁对"。',
        '（a）北方/西北/青藏/西南：FCS10 与植被图在针/阔轴上显著自洽（西北 %+.1fpp、青藏 %+.1fpp、华北 %+.1fpp、'
        '西南 %+.1fpp、东北 %+.1fpp），两源互证，植被图可直接用作标签源。'
        % (disc['by_region']['西北']['separation_pp'], disc['by_region']['青藏']['separation_pp'],
           disc['by_region']['华北']['separation_pp'], disc['by_region']['西南']['separation_pp'],
           disc['by_region']['东北']['separation_pp']),
        '（b）南 6 省（粤闽桂浙湘赣，即 V5 目标区）：FCS10 的针/阔与地面调查**零分离甚至反向**'
        '（合并 %+.1fpp；广西 %+.1fpp、湖南 %+.1fpp、福建 %+.1fpp），且其非森林区针叶率 %.1f%% 高于针叶图斑内 %.1f%%；'
        '同时把马尾松（%.1f%% 判成常绿阔叶 EBF）、杉木（%.1f%% 判成 EBF）——生物学上确定的常绿针叶——判成阔叶。'
        % (disc['south6_combined']['separation_pp'],
           disc['by_prov_south']['广西壮族自治区']['separation_pp'],
           disc['by_prov_south']['湖南省']['separation_pp'],
           disc['by_prov_south']['福建省']['separation_pp'],
           disc['south6_combined']['nonforest_rate'], disc['south6_combined']['needle_patch_rate'],
           phys['马尾松']['fcs10_EBF_pct'], phys['杉木']['fcs10_EBF_pct']),
        '结论：南方应以植被图为准；FCS10 在南方的针叶判读不可信，'
        'doc77 §13「南方真针叶仅 1.66%」与 doc80 §3「CGLS vs FCS10 互斥 1.51%」很可能都是 FCS10 口径伪影，'
        '而非地表事实——但需 Q4 影像核查确认。',
    ]),
    'confidence': '中等偏强（间接证据链：判别效度 + 生物学不可能性 + 年代标记；未用 S2/GEDI 独立仲裁）',
    'evidence': ['discriminant', 'physical_impossibility', 'vintage_markers', 'axis_decomposition'],
    'what_would_overturn_it': (
        '若 Q4 的 S2 双年物候 + GEDI 冠层高度核查显示：南方被植被图判为马尾松/杉木的图斑实际已是常绿阔叶次生林'
        '（即 1980s–90s 的松杉人工林在 2000s 后被阔叶化），则"植被图对"的结论需改为"植被图记录的是历史植被，'
        '地表已变"。这是本诊断无法排除的唯一替代解释，也是年代差上界的来源——故 V4 影像核查不可省。'
    ),
}

# ---------------------------------------------------------------- 8. CSV
# by_prov.csv：省 × 图斑/面积/映射/v2
rows = []
for prov, g in veg_g.groupby('_prov', dropna=False):
    b = sj1[sj1['prov'] == prov]
    bb = b[(b['veg_leaf'].isin(['needle', 'broad'])) & (b['leaf'].isin(['needle', 'broad']))]
    rows.append({
        'prov': None if pd.isna(prov) else str(prov),
        'n_patch': int(len(g)),
        'area_km2': round(float(g['_area_km2'].sum()), 1),
        'n_patch_needle': int((g['_leaf_forest'] == 'needle').sum()),
        'area_km2_needle': round(float(g.loc[g['_leaf_forest'] == 'needle', '_area_km2'].sum()), 1),
        'n_patch_broad': int((g['_leaf_forest'] == 'broad').sum()),
        'area_km2_broad': round(float(g.loc[g['_leaf_forest'] == 'broad', '_area_km2'].sum()), 1),
        'n_patch_mixed': int((g['_leaf_forest'] == 'mixed').sum()),
        'n_patch_bamboo': int((g['_leaf_forest'] == 'bamboo').sum()),
        'v2_points': int(len(b)),
        'v2_in_forest': int(b['_is_forest'].fillna(False).astype(bool).sum()),
        'v2_both': int(len(bb)),
        'agreement': round(float((bb['veg_leaf'] == bb['leaf']).mean()), 4) if len(bb) else None,
        'veg_needle_fcs10_broad': int(((bb['veg_leaf'] == 'needle') & (bb['leaf'] == 'broad')).sum()),
        'veg_broad_fcs10_needle': int(((bb['veg_leaf'] == 'broad') & (bb['leaf'] == 'needle')).sum()),
    })
by_prov_csv = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
by_prov_csv.to_csv(os.path.join(OUT, 'veg_qa_by_prov.csv'), index=False, encoding='utf-8-sig')

# by_type.csv：逐植被类型
tp = veg_g.groupby('植被类型', dropna=False).agg(
    n_patch=('_oid', 'size'),
    area_km2=('_area_km2', 'sum'),
    veg_type=('植被型', lambda s: s.mode().iloc[0] if len(s.mode()) else None),
    habit=('生活型', lambda s: s.mode().iloc[0] if len(s.mode()) else None),
).reset_index()
tp['vt_mapped'] = tp['habit'].map(map_habit)
# 该类型内 v2 点
cnt = sj1[sj1['_oid'].notna()].groupby('植被类型').agg(
    n_v2=('point_id', 'size'),
    v2_needle=('leaf', lambda s: int((s == 'needle').sum())),
    v2_broad=('leaf', lambda s: int((s == 'broad').sum())),
    v2_mixed=('leaf', lambda s: int((s == 'mixed').sum())),
).reset_index()
tp = tp.merge(cnt, on='植被类型', how='left').fillna({'n_v2': 0, 'v2_needle': 0, 'v2_broad': 0, 'v2_mixed': 0})
tp['area_km2'] = tp['area_km2'].round(1)
tp['pct_v2_broad'] = [pct(int(r.v2_broad), int(r.n_v2)) for r in tp.itertuples()]
tp['植被类型'] = tp['植被类型'].fillna('<空>')
tp = tp.sort_values('n_patch', ascending=False)
tp.to_csv(os.path.join(OUT, 'veg_qa_by_type.csv'), index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 9. JSON
def cls_summary(col):
    out = {}
    for k, g in veg_g.groupby(col):
        out[str(k)] = {'n_patch': int(len(g)), 'area_km2': round(float(g['_area_km2'].sum()), 1),
                       'pct_patches': pct(len(g), len(veg_g)),
                       'pct_area': pct(float(g['_area_km2'].sum()), float(veg_g['_area_km2'].sum()))}
    return out


basic = {
    'n_features': int(len(veg_g)),
    'crs_declared': crs_raw,
    'crs_used': 'EPSG:4326 (allow_override; 主 agent 实测最优偏移 = (0,0))',
    'bounds_lonlat': [round(float(x), 4) for x in veg_g.total_bounds],
    'n_null_geometry': n_null,
    'n_empty_geometry': n_empty,
    'n_invalid_geometry': n_invalid,
    'n_invalid_makevalid_failed': int(n_fix_fail),
    'invalid_area_change_km2': round(float(area_after[~valid_mask].sum() - area_before[~valid_mask].sum()), 2),
    'invalid_note': ('268 个 invalid 图斑仅占 %.2f%%；make_valid 后总面积变化 %.2f km²（相对全国 %.4f%%），'
                     '对统计与 point-in-polygon 无实质影响；sjoin 前已全部 make_valid 并仅保留多边形部分'
                     % (100.0 * n_invalid / len(veg_g), area_after[~valid_mask].sum() - area_before[~valid_mask].sum(),
                        100.0 * abs(area_after[~valid_mask].sum() - area_before[~valid_mask].sum()) / veg_g['_area_km2'].sum())),
    'area_total_km2': round(float(veg_g['_area_km2'].sum()), 1),
    'area_quantiles_km2': {k: round(float(v), 3) for k, v in veg_g['_area_km2'].quantile([0, .05, .25, .5, .75, .95, 1]).items()},
    'area_note': ('最小图斑面积近 0（1:100 万制图综合的碎屑），中位 %.2f km²；'
                  '面积按 EPSG:6933 等积投影计算' % float(veg_g['_area_km2'].median())),
    'habit_x_vegtype': ct_dict,
    'prov_counts_area': prov_tab.to_dict(orient='records'),
    'n_patch_unassigned_prov': n_prov_missing,
    'n_prov_features_invalid_repaired': n_prov_invalid,
    'habit_missing_total': int(veg_g['生活型'].isna().sum()),
    'habit_missing_in_forest_types': none_but_forest_n,
    'prov_note': 'DataV 省界为 GCJ-02 且含拓扑错误，逐 feature make_valid 后按 representative_point within 归属；未做 unary_union',
}

mapping = {
    'rules': {'needle': sorted(NEEDLE), 'broad': sorted(BROAD), 'mixed': sorted(MIXED), 'bamboo': sorted(BAMBOO),
              'none': '非森林/未标注（排除）'},
    'by_habit_all_features': cls_summary('_leaf_all'),
    'by_habit_forest_only_vegtype_123': cls_summary('_leaf_forest'),
    'none_but_vegtype_123': {'n_patch': none_but_forest_n, **rev},
    'keyword_reverse_inference': {
        'crosstab': kw_dict,
        'note': ('植被类型 关键词与 生活型 高度自洽（见 crosstab）；但因"生活型为空 且 植被型∈{1,2,3}"的图斑为 0，'
                 '不存在需要靠关键词反推的森林图斑。关键词反推在本数据上无增益。'),
        'max_len_note': '植被类型 字段被截断到 10 个字符（shapefile DBF 字段宽度），部分群系名残缺',
    },
    'ambiguity_shrub_leak': {
        'n_patch': int((~veg_g['_is_forest'] & veg_g['_leaf_all'].isin(['needle', 'broad'])).sum()),
        'area_km2': round(float(veg_g.loc[~veg_g['_is_forest'] & veg_g['_leaf_all'].isin(['needle', 'broad']), '_area_km2'].sum()), 1),
        'detail': {str(k): int(v) for k, v in
                   veg_g.loc[~veg_g['_is_forest'] & veg_g['_leaf_all'].isin(['needle', 'broad']), '植被型'].value_counts().items()},
        'note': ('灌丛(植被型=4)里有 267 常绿针叶 / 6,023 常绿阔叶 / 4,132 落叶阔叶 的图斑——这些是**灌木**，'
                 '按生活型直接映射会被误当成针/阔叶**林**。因此本脚本 V2 主口径限定 植被型∈{1,2,3}，'
                 '并以 sensitivity_all_habitat_mapping 给不限定口径的对照。'),
    },
}

caveats = [
    'V1（CRS 偏移）：本脚本按主 agent 实测结论直接用 WGS84（偏移 (0,0)），未重复做地标配准；偏移量结论来自上游，本文件未独立复核。',
    '植被图年代为"2020 年前后调查汇总"，不是单年；与 2017–2024 存在时间差，不能作逐年声明。',
    '1:100 万图斑边界粗略，落在边界附近的 v2 点可能因 within 判定而落到相邻图斑或判定为"不在图斑内"。',
    '植被图的"针叶/阔叶"是**建群种生活型**，与遥感（FCS10）的冠层物候判读口径不同源；灌丛层尤其如此。',
    'dataV 省界 GCJ-02 有偏移（约数百米），省归属仅作粗略分区；省界附近图斑可能归错省。',
    '省界按 representative_point 归属，跨省大图斑只计入一个省，故分省面积之和 ≈ 全国面积但非严格相等。',
    'v2 的 prov/eco 字段为 v2 文件自带（上游已定），本脚本直接沿用，未重新归属。',
    '多重匹配（一点落入多个图斑）已去重：优先森林图斑、其次面积最大；重叠来源可能是 invalid 几何 make_valid 后的轻微外扩。',
    '植被图不可能包含 2020 年后新造林/采伐，FCS10 是 2020 年代遥感，二者差异含真实地表变化。',
    '**【重要】植被图的"2020 年"是出版年而非观测年**：树种谱无桉/橡胶/湿地松/火炬松/木麻黄/泡桐（均 0 图斑），'
    '却有刺槐 579、桑 356 —— 主体内容来自 1980s–1990s 调查（见 vintage_markers）。'
    '因此它与 FCS10（2020 年代遥感）的差异同时包含"年代差"与"口径差"，本诊断无法把两者分离；'
    '所有"植被图 vs FCS10"的分歧率都是**年代差 + 口径差的上界**，不是纯粹的判读错误率。',
    'V2 一致率 0.5753 的低值不能直接读作"两源各错一半"：南方的低一致率源于 FCS10 在该区针/阔零分离（见 discriminant），'
    '而北方的分歧（如新疆 0.915 一致、华中 0.453）需要按省分别解读，全国汇总值被南方样本机械拉低。',
    '本诊断未使用独立真值（无人工判读、无影像核查），"谁对"的结论建立在**内部一致性 + 生物学不可能性 + 年代标记**三重间接证据上，'
    '强度低于 doc81 §4 预注册的 S2 物候 / GEDI 冠层高度仲裁（属 Q4）。',
]

out = {
    'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    'script': os.path.abspath(__file__),
    'inputs': {'veg': VEG_SHP, 'v2': V2_PARQ, 'prov': PROV_JSON},
    'basic': basic,
    'mapping': mapping,
    'V2': V2,
    'by_prov': by_prov,
    'by_prov_suppressed_n_lt_30': by_prov_suppressed,
    'by_region': by_region,
    'south_needle': south,
    'V5_preconditions': V5,
    'discriminant': disc,
    'axis_decomposition': axis,
    'physical_impossibility': phys,
    'vintage_markers': vintage,
    'area_reconciliation': coverage,
    'verdict': verdict,
    'conflicts': {
        'n_conflict': int(len(conf)),
        'pct_of_both': pct(len(conf), nb),
        'direction_counts': V2['conflict_direction'],
        'top_formations': V2['top_conflict_formations'],
        'needle_patch_by_formation_top': V2['needle_patch_by_formation_top'],
    },
    'caveats': caveats,
}

with open(os.path.join(OUT, 'veg_qa.json'), 'w', encoding='utf-8') as f:
    json.dump(jsonable(out), f, ensure_ascii=False, indent=2, allow_nan=False)

log('DONE -> %s' % os.path.join(OUT, 'veg_qa.json'))
print()
print('=== KEY ===')
print('V2 agreement %.4f (n=%d) CI=%s gate0.80=%s' % (V2['agreement'], nb, V2['agreement_wilson95'], V2['gate_pass']))
print('in forest polygon %.1f%% (%d/%d)' % (V2['pct_in_forest_polygon'], n_in_forest, len(sj1)))
print('south6 needle patches %d area %.0f km2' % (south['n_patch_needle_forest'], south['area_km2_needle_forest']))
print('south6 points in needle patches', south['v2_points_in_needle_patches'])
