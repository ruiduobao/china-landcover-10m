# -*- coding: utf-8 -*-
"""f3_fuse.py — 六类林地样本 v1 融合：45/46 三档定档 + L1 构造性自洽 + L2 源间一致率 + 生态合理性过滤

* 输入（只读）
    F:/lc_work/nb_national_2026/forest6/forest_pool_v1.parquet        472,307 × 103（F1 林地池）
    F:/lc_work/nb_national_2026/forest6/mixed_candidates.parquet      472,307 × 32（F2b 混交候选）
    F:/lc_work/nb_national_2026/forest6/mixed_pheno.parquet           6,864（F2a 混交点补采物候）
    .../China_Forest_Dataset_v1.0/2_PFT_Annual_1981-2013/2013/*.tif   8 个 PFT 类（1 km, Albers）
    F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif.vat.dbf 30 m 植被类型码表
* 规则源
    技术文档/84 §4.2（2026-10-25 修正：45/46 由邻域针叶码定义，不用混交点自身物候）
    技术文档/84 §4.3（L1/L2/L3）、§4.4（生态合理性硬约束 + 分区要求）
* 预注册判据（实现前写死，见 PREREG；事后不改）
    T1 常绿/落叶轴：45 amp 中位 < 46 且比值 ≥ 2.0；share(amp>500) 45 ≤ 0.20 且 46 ≥ 0.50
    T2 可分性：六分类 OOF macro-F1 ≥ legacy_base−0.02，且 45/46 单类 F1 ≥ 0.30
    T3 独立轴支持：30m植被 × PFT 两者都判出针叶轴时的"与标签同向率" ≥ legacy_base 同口径
    T4 可用规模：prefer_mixed 口径下 45/46 各 ≥ 500 点
    T5 分布：≥6/8 生态区该档 ≥30 点
* 输出（F:/lc_work/nb_national_2026/forest6/）
    forest6_v1.parquet / f3_stats.json / f3_class_compare.csv / code/_f3_votes.parquet
* 用法
    python f3_fuse.py [--rf-trees 300] [--pure-cap 6000] [--n-jobs 8] [--skip-rf] [--force-votes]
* 资源：单进程，峰值内存 < 6 GB，CPU ≤ 10（RF n_jobs=8）；实测 ~10 min
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------- 路径
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
F6 = r'F:/lc_work/nb_national_2026/forest6'
POOL = os.path.join(F6, 'forest_pool_v1.parquet')
CAND = os.path.join(F6, 'mixed_candidates.parquet')
PHENO = os.path.join(F6, 'mixed_pheno.parquet')
VAT = r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif.vat.dbf'
PFT_DIR = os.path.join(PROJ, '数据', '外部数据', 'China_Forest_Dataset_v1.0',
                       '2_PFT_Annual_1981-2013', '2013')
VOTES = os.path.join(F6, 'code', '_f3_votes.parquet')

OUT_PQ = os.path.join(F6, 'forest6_v1.parquet')
OUT_JSON = os.path.join(F6, 'f3_stats.json')
OUT_CSV = os.path.join(F6, 'f3_class_compare.csv')

AEF = ['A%02d' % i for i in range(64)]
CLS = [41, 42, 43, 44, 45, 46]
CLS_NAME = {41: '常绿针叶林', 42: '落叶针叶林', 43: '常绿阔叶林', 44: '落叶阔叶林',
            45: '常绿针阔混交林', 46: '落叶针阔混交林'}
PURE = [41, 42, 43, 44]
MIXED = [45, 46]
LEAF_NA, LEAF_NEEDLE, LEAF_BROAD, LEAF_MIXED = 0, 1, 2, 3
H_ENF, H_DNF, H_EBF, H_DBF, H_MIX = 41, 42, 43, 44, 45
# 常绿/落叶针叶轴（独立编码，不与六类码混用）
AX_NA, AX_ENF, AX_DNF = 0, 1, 2

HARD_LO, HARD_HI = 0.35, 0.65
EV_DOM = 0.60
BLOCK_SEED = 20261025
SEED_MODEL = 20261025
Z95 = 1.959963984540054
T_SCALE = 10.0   # bio01/bio06 存 0.1 ℃（实测 bio06<−200 恰命中已知 41 点越界清单 → 单位确认）

SOUTH_NEEDLE_PROV = {'广东省', '福建省', '广西壮族自治区', '海南省',
                     '香港特别行政区', '澳门特别行政区'}
QINGZANG = '青藏'
HIGH_ELEV_M = 3000.0
ECO_ORDER = ['东北', '华北', '西北', '青藏', '西南', '华中', '华东', '华南']

# GPKG cls4 → 六类（4=EBF 5=DBF 6=ENF 7=DNF，f1_pool.py 定义）
CLS4_2_CLS6 = {4: 43, 5: 44, 6: 41, 7: 42}
# 母库 FCS10 原始码 → 六类（f1_pool.py MASTER2CLS6 逐字一致）
CODE10_2_CLS6 = {71: 41, 72: 41, 81: 42, 82: 42, 51: 43, 52: 43,
                 61: 44, 62: 44, 91: 45, 92: 46}
# FCS10 判别效度分区（doc82 / r5_fuse.py 逐字一致）：南方 FCS10 针阔无判别效度
SOUTH_FCS10_VOID = {'华东', '华中', '华南'}
PFT_DEAD = 2013

PREREG = {
    'T1_amp': 'amp 中位 45 < 46 且 46/45 ≥ 2.0；share(amp>500) 45 ≤ 0.20 且 46 ≥ 0.50',
    'T2_sep': '六分类 OOF macro-F1 ≥ legacy_base−0.02 且 45/46 单类 F1 ≥ 0.30',
    'T3_indep_axis': 'v30 与 PFT 都判出针叶轴（ENF/DNF）的混合点上，与候选 45/46 同向率 ≥ legacy_base；可用点 <100 记不可判(None)',
    'T4_size': 'prefer_mixed 口径下 45/46 各 ≥ 500 点',
    'T5_eco': '≥6/8 生态区该档 ≥30 点',
    'decision': 'T1&T2 硬判据；T4 可用性门槛；T3/T5 排序',
    'note': '本节随脚本落盘，判据先写死后跑',
}

T0 = time.time()


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def wilson(k, n, z=Z95):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, cen - half), 4), round(min(1.0, cen + half), 4)]


def jd(o):
    if isinstance(o, dict):
        return {str(k): jd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jd(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if math.isnan(o) else round(o, 6)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def dist_tbl(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'n': 0}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {'n': int(len(x)), 'p10': round(float(q[0]), 1), 'p25': round(float(q[1]), 1),
            'median': round(float(q[2]), 1), 'p75': round(float(q[3]), 1),
            'p90': round(float(q[4]), 1), 'mean': round(float(x.mean()), 1),
            'share_gt500': round(float((x > 500).mean()), 4),
            'share_lt350': round(float((x < 350).mean()), 4)}


def entropy(sh):
    s = sh.sum(axis=1, keepdims=True)
    q = np.divide(sh, np.maximum(s, 1e-9))
    return -np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0).sum(axis=1)


# ================================================================ 30 m 植被类型码表
def read_vat():
    b = open(VAT, 'rb').read()
    nrec, hlen, rlen = struct.unpack('<IHH', b[4:12])
    off, fields = 32, []
    while b[off] != 0x0d:
        fd = b[off:off + 32]; off += 32
        fields.append((fd[:11].split(b'\x00')[0].decode('utf-8', 'ignore'), fd[16]))
    out = {}
    for r in range(nrec):
        rec = b[hlen + r * rlen:hlen + (r + 1) * rlen]
        pos, d = 1, {}
        for nm, ln in fields:
            d[nm] = rec[pos:pos + ln].decode('utf-8', 'ignore').strip(); pos += ln
        try:
            out[int(d['Value'])] = (d.get('类型组', ''), d.get('植被型', ''), d.get('群系', ''))
        except Exception:
            pass
    return out


def veg30_habit(vat):
    """30 m 植被类型 → 生活型 4 类。
    类型组首 token：1 针叶林 / 2 针阔叶混 / 3 阔叶林（首 token 精确匹配，避免 '11 栽培植'/'10 高山植' 误判）。
    针叶林内：群系含"落叶松"→DNF（中国植被图的落叶针叶即落叶松属），否则 ENF。
    阔叶林内：植被型 (8)(9)(10)→DBF；(12)(13)(14)(15)(16)→EBF；(11)常绿落叶混交 / (17)竹林 → 不计票。
    """
    def h(v):
        g, vt, qun = vat.get(v, ('', '', ''))
        tok = g.split()[0] if g.split() else ''
        if tok == '1':
            return H_DNF if '落叶松' in qun else H_ENF
        if tok == '3':
            try:
                mi = int(vt.split(')')[0].strip('('))
            except Exception:
                return 0
            if mi in (8, 9, 10):
                return H_DBF
            if mi in (12, 13, 14, 15, 16):
                return H_EBF
            return 0
        if tok == '2':
            return H_MIX
        return 0
    return {k: h(k) for k in vat}


LUT_H2L = np.zeros(256, np.int8)
LUT_H2L[[H_ENF, H_DNF]] = LEAF_NEEDLE
LUT_H2L[[H_EBF, H_DBF]] = LEAF_BROAD
LUT_H2L[H_MIX] = LEAF_MIXED
LUT_H2AX = np.zeros(256, np.int8)
LUT_H2AX[H_ENF] = AX_ENF
LUT_H2AX[H_DNF] = AX_DNF


# ================================================================ PFT 2013
PFT_FILES = [('ENF_boreal', H_ENF), ('ENF_temperate', H_ENF), ('DNF', H_DNF),
             ('EBF_temperate', H_EBF), ('EBF_tropical', H_EBF),
             ('DBF_boreal', H_DBF), ('DBF_temperate', H_DBF), ('DBF_tropical', H_DBF)]


def build_pft_votes(lon, lat):
    """8 个 1 km 二值 PFT 图层逐点采样 + 3×3 邻域计数。
    每层独立网格/偏移（ENF/DNF/DBF 3580×4254；EBF 3946×3537），不做重采样。
    """
    n = len(lon)
    habit_pt = np.zeros(n, np.int8)
    n_needle_pt = np.zeros(n, np.int8)
    n_broad_pt = np.zeros(n, np.int8)
    n_needle3 = np.zeros(n, np.int16)
    n_broad3 = np.zeros(n, np.int16)
    ev3 = np.zeros(n, np.int16)     # 常绿针叶 3×3 像元数（ENF_boreal + ENF_temperate）
    dec3 = np.zeros(n, np.int16)    # 落叶针叶 3×3 像元数（DNF）
    info = {}
    for fn, code in PFT_FILES:
        pth = os.path.join(PFT_DIR, 'China-%s_1km_2013_v1.0.tif' % fn)
        if not os.path.exists(pth):
            emit('  !! 缺 PFT 文件 %s' % pth)
            continue
        with rasterio.open(pth) as s:
            H, W = s.height, s.width
            tr = Transformer.from_crs('EPSG:4326', CRS.from_wkt(s.crs.to_wkt()), always_xy=True)
            x, y = tr.transform(lon, lat)
            cols, rows = (~s.transform) * (x, y)
            col = np.floor(cols).astype(np.int64)
            row = np.floor(rows).astype(np.int64)
            ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            a = np.asarray(s.read(1))
        idx = np.where(ok)[0]
        r0 = np.clip(row[idx], 0, H - 1); c0 = np.clip(col[idx], 0, W - 1)
        v = a[r0, c0] > 0
        ii = idx[v]
        if code in (H_ENF, H_DNF):
            n_needle_pt[ii] = np.minimum(n_needle_pt[ii] + 1, 120)
        else:
            n_broad_pt[ii] = np.minimum(n_broad_pt[ii] + 1, 120)
        first = habit_pt[ii] == 0
        if first.any():
            habit_pt[ii[first]] = code
        f = uniform_filter(a.astype(np.float32), size=3, mode='constant') * 9.0
        cnt = np.rint(f[r0, c0]).astype(np.int16)
        if code in (H_ENF, H_DNF):
            n_needle3[idx] = np.minimum(n_needle3[idx] + cnt, 9)
        else:
            n_broad3[idx] = np.minimum(n_broad3[idx] + cnt, 9)
        if code == H_ENF:
            ev3[idx] = np.minimum(ev3[idx] + cnt, 9)
        elif code == H_DNF:
            dec3[idx] = np.minimum(dec3[idx] + cnt, 9)
        info[fn] = {'code': int(code), 'grid': [int(H), int(W)], 'hits': int(v.sum())}
        emit('  PFT %-14s %dx%d 命中 %6d' % (fn, H, W, int(v.sum())))
        del a, f
    pft_leaf_pt = np.where((n_needle_pt > 0) & (n_broad_pt > 0), LEAF_MIXED,
                           np.where(n_needle_pt > 0, LEAF_NEEDLE,
                                    np.where(n_broad_pt > 0, LEAF_BROAD, LEAF_NA))).astype(np.int8)
    pft_leaf_3 = np.where((n_needle3 > 0) & (n_broad3 > 0), LEAF_MIXED,
                          np.where(n_needle3 > 0, LEAF_NEEDLE,
                                   np.where(n_broad3 > 0, LEAF_BROAD, LEAF_NA))).astype(np.int8)
    habit4 = np.full(n, 0, np.int8)
    only_one = ((n_needle_pt + n_broad_pt) == 1)
    habit4[only_one] = habit_pt[only_one]
    # PFT 3×3 多数票（针叶像元数 vs 阔叶像元数）；打平或缺失 → NA。
    # **不用"两类同时出现→mixed"**：3.4 km 窗口内几乎总有两者（实测 190,994/472,307=40%），
    # 会把 mixed 变成默认类，L2 直接被稀释（首版踩此坑）。
    pft_leaf_3maj = np.where(n_needle3 > n_broad3, LEAF_NEEDLE,
                             np.where(n_broad3 > n_needle3, LEAF_BROAD, LEAF_NA)).astype(np.int8)
    # 针叶轴：ENF=1 / DNF=2 / 0=不可判（只有一侧有像元才算判出）
    pft_axis = np.where((ev3 > 0) & (dec3 == 0), AX_ENF,
                        np.where((dec3 > 0) & (ev3 == 0), AX_DNF, 0)).astype(np.int8)
    return (pft_leaf_pt, pft_leaf_3, pft_leaf_3maj, habit4, n_needle_pt, n_broad_pt,
            n_needle3, n_broad3, ev3, dec3, pft_axis, info)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rf-trees', type=int, default=300)
    ap.add_argument('--pure-cap', type=int, default=6000)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--skip-rf', action='store_true')
    ap.add_argument('--force-votes', action='store_true')
    args = ap.parse_args()

    # ------------------------------------------------ 读池
    emit('读 forest_pool_v1.parquet')
    pc = ['uid', 'lon', 'lat', 'prov', 'eco', 'cls6', 'cls4_gpkg', 'code_master', 'layer', 'tier',
          'src_primary', 'amp2023', 'djf2023', 'elev_m', 'slope_deg', 'aspect_deg',
          'bio01', 'bio06'] + AEF
    p = pd.read_parquet(POOL, columns=pc)
    n = len(p)
    uid_arr = p.uid.values.copy()
    emit('池 %d 点；cls6 %s' % (n, p.cls6.value_counts().sort_index().to_dict()))

    # 统一码：GPKG 点用 cls4_gpkg 映射，母库点用 code_master 映射（两者互补，零重叠）
    c4 = p.cls4_gpkg.values.astype('float64')
    cm = p.code_master.values.astype('float64')
    from_gpkg = np.isfinite(c4)
    src_code = np.where(from_gpkg,
                        pd.Series(c4).map(CLS4_2_CLS6).fillna(0).values,
                        pd.Series(cm).map(CODE10_2_CLS6).fillna(0).values).astype(np.int16)
    p['src_code'] = src_code
    p['src_code_origin'] = np.where(from_gpkg, 'gpkg_cls4', 'master_code10')
    emit('统一源码 src_code %s（gpkg %d / master %d）' % (
        pd.Series(src_code).value_counts().sort_index().to_dict(),
        int(from_gpkg.sum()), int((~from_gpkg).sum())))

    emit('读 mixed_candidates.parquet')
    cc = ['uid', 'cls6_orig', 'derived_as', 'center_code', 'p_needle', 'p_forest',
          'needle_evergreen_frac', 'frac_needle', 'derived_gate_strict', 'fold10km',
          'v30_value', 'v30_group', 'n_needle_win', 'n_broad_win', 'n_mixedcode_win']
    c = pd.read_parquet(CAND, columns=cc)
    assert (c.uid.values == uid_arr).all(), 'uid 顺序不一致'
    for k in cc[1:]:
        p[k] = c[k].values
    del c
    p['cls6'] = p['cls6'].astype(np.int8)
    p['cls6_orig'] = p['cls6_orig'].astype(np.int8)

    ph = pd.read_parquet(PHENO)
    ph = ph.rename(columns={'amp': 'amp_f2a', 'djf': 'djf_f2a'})[['uid', 'amp_f2a', 'djf_f2a',
                                                                  'n_obs_gs', 'n_obs_djf']]
    p = p.merge(ph, on='uid', how='left')
    assert len(p) == n and (p.uid.values == uid_arr).all(), 'merge 改变了行序/行数'
    miss_before = int(p.amp2023.isna().sum())
    p['amp2023'] = p.amp2023.fillna(p.amp_f2a)
    p['djf2023'] = p.djf2023.fillna(p.djf_f2a)
    emit('物候补采：池内 amp 缺失 %d → 补后仍缺 %d（补入 %d）' % (
        miss_before, int(p.amp2023.isna().sum()), n - int(p.amp2023.isna().sum())))

    # ------------------------------------------------ 证据缓存
    if os.path.exists(VOTES) and not args.force_votes:
        vv = pd.read_parquet(VOTES)
        if len(vv) == n and (vv.uid.values == uid_arr).all() and 'pft_leaf_3maj' in vv.columns:
            emit('命中证据缓存 %s' % VOTES)
            for k in vv.columns:
                if k != 'uid':
                    p[k] = vv[k].values
        else:
            args.force_votes = True
    if (not os.path.exists(VOTES)) or args.force_votes:
        emit('30 m 植被码表 → 生活型')
        vat = read_vat()
        h30map = veg30_habit(vat)
        hv = p.v30_value.values.astype(np.int64)
        uniq = {int(v): h30map.get(int(v), 0) for v in np.unique(hv)}
        habit30 = np.zeros(n, np.int8)
        okv = hv != 65535
        habit30[okv] = np.array([uniq[int(v)] for v in hv[okv]], np.int8)
        p['habit30'] = habit30
        emit('  habit30 %s' % pd.Series(habit30).value_counts().sort_index().to_dict())
        emit('PFT 2013 点采样（8 类 1 km）')
        (lpt, l3, l3m, h4, nnp, nbp, nn3, nb3, ev3, dec3, pax, pinfo) = build_pft_votes(
            p.lon.values.astype('float64'), p.lat.values.astype('float64'))
        p['pft_leaf_pt'] = lpt; p['pft_leaf_3'] = l3; p['pft_leaf_3maj'] = l3m
        p['pft_habit4'] = h4
        p['pft_n_needle_pt'] = nnp; p['pft_n_broad_pt'] = nbp
        p['pft_needle_3x3'] = nn3; p['pft_broad_3x3'] = nb3
        p['pft_ev_3x3'] = ev3; p['pft_dec_3x3'] = dec3; p['pft_axis'] = pax
        keep = ['uid', 'habit30', 'pft_leaf_pt', 'pft_leaf_3', 'pft_leaf_3maj', 'pft_habit4',
                'pft_n_needle_pt', 'pft_n_broad_pt', 'pft_needle_3x3', 'pft_broad_3x3',
                'pft_ev_3x3', 'pft_dec_3x3', 'pft_axis']
        p[keep].to_parquet(VOTES, index=False)
        emit('证据缓存已写 %s（%d 列）' % (VOTES, len(keep)))
    else:
        pinfo = {'cached': True}

    # ------------------------------------------------ 叶型票 / 针叶轴票
    p['v30_leaf'] = LUT_H2L[np.clip(p.habit30.values.astype(np.int32), 0, 255)]
    p['v30_axis'] = LUT_H2AX[np.clip(p.habit30.values.astype(np.int32), 0, 255)]
    p['fcs10_leaf'] = np.select(
        [p.cls6.isin([41, 42]).values, p.cls6.isin([43, 44]).values, p.cls6.isin([45, 46]).values],
        [LEAF_NEEDLE, LEAF_BROAD, LEAF_MIXED], default=LEAF_NA).astype(np.int8)
    ctr = p.center_code.values.astype(np.int32)
    is_ctr_mix = np.isin(ctr, [91, 92])
    is_ctr_needle = np.isin(ctr, [71, 72, 81, 82])
    is_ctr_broad = np.isin(ctr, [51, 52, 61, 62])
    p['fcs10_leaf_mixcode'] = np.select(
        [is_ctr_mix, p.n_mixedcode_win.values >= 3, is_ctr_needle, is_ctr_broad],
        [LEAF_MIXED, LEAF_MIXED, LEAF_NEEDLE, LEAF_BROAD], default=LEAF_NA).astype(np.int8)
    p['fcs10_leaf_win'] = np.where((p.n_needle_win.values >= 3) & (p.n_broad_win.values >= 3),
                                   LEAF_MIXED, LEAF_NA).astype(np.int8)

    # ------------------------------------------------ 三档
    der = p.derived_as.values.astype(np.int8)
    still_mixed = p.needle_evergreen_frac.values      # 用于 legacy 备选方案 B
    hard = (p.p_needle.values >= HARD_LO) & (p.p_needle.values <= HARD_HI)
    strict = p.derived_gate_strict.values.astype(bool)
    legacy = p.cls6_orig.isin(MIXED).values
    T = {'S1': (der > 0) & is_ctr_mix,
         'S2': strict & (der > 0),
         'S3': (der > 0) & (is_ctr_mix | hard)}
    for k in ('S1', 'S2', 'S3'):
        emit('%s: %6d（45 %5d / 46 %5d；legacy 来源 %d）' % (
            k, T[k].sum(), int((T[k] & (der == 45)).sum()), int((T[k] & (der == 46)).sum()),
            int((T[k] & legacy).sum())))
    rel = {f'{k}_n': int(T[k].sum()) for k in T}
    for k in T:
        rel[f'{k}_45'] = int((T[k] & (der == 45)).sum())
        rel[f'{k}_46'] = int((T[k] & (der == 46)).sum())
    rel.update({
        'S1_not_S3': int((T['S1'] & ~T['S3']).sum()),
        'S3_not_S2': int((T['S3'] & ~T['S2']).sum()),
        'S2_not_S3': int((T['S2'] & ~T['S3']).sum()),
        'S1_not_S2': int((T['S1'] & ~T['S2']).sum()),
        'S2_and_S3': int((T['S2'] & T['S3']).sum()),
        'derived_total': int((der > 0).sum()),
        'legacy_derived': int((legacy & (der > 0)).sum()),
        'pure_labeled_derived': int(((der > 0) & p.cls6_orig.isin(PURE).values).sum()),
        'legacy_strict': int((legacy & strict & (der > 0)).sum()),
    })
    emit('档间关系 %s' % json.dumps(rel, ensure_ascii=False))

    # ------------------------------------------------ 参考集（四类纯林固定子样本）
    X = np.ascontiguousarray(p[AEF].values, dtype=np.float32)
    assert np.isfinite(X).all(), 'AEF64 含非有限值'
    fold = p.fold10km.values.astype(np.int8)
    rng = np.random.default_rng(BLOCK_SEED)
    pure_idx = np.sort(np.concatenate([
        rng.choice(np.where(p.cls6.values == cl)[0],
                   size=min(args.pure_cap, int((p.cls6.values == cl).sum())), replace=False)
        for cl in PURE]))
    pure_cls = p.cls6.values[pure_idx]
    tree = cKDTree(X[pure_idx])
    emit('纯林参考子样本 %d（每类 ≤%d）' % (len(pure_idx), args.pure_cap))

    def nn_profile(q_idx, k=10, exclude_self=False):
        kk = k + 1 if exclude_self else k
        d, ii = tree.query(X[q_idx], k=kk, workers=args.n_jobs)
        nb = pure_cls[ii]; dd = d
        if exclude_self:
            nb = nb[:, 1:]; dd = d[:, 1:]
        sh = np.zeros((len(q_idx), 4), np.float32)
        for j, cl in enumerate(PURE):
            sh[:, j] = (nb == cl).mean(axis=1)
        return sh, dd.mean(axis=1)

    pure_ctrl = {}
    for cl in PURE:
        sel = np.where(pure_cls == cl)[0][:2000]
        sh, dd = nn_profile(pure_idx[sel], exclude_self=True)
        mx = sh.max(axis=1)
        pure_ctrl[int(cl)] = {'n': int(len(sel)),
                              'mean_nn_share': {str(c_): round(float(sh[:, j].mean()), 4)
                                                for j, c_ in enumerate(PURE)},
                              'mean_max_share': round(float(mx.mean()), 4),
                              'share_max_ge_0.8': round(float((mx >= 0.8).mean()), 4),
                              'mean_entropy': round(float(entropy(sh).mean()), 4),
                              'mean_nn_dist': round(float(dd.mean()), 3)}
    emit('纯类 NN 基线（留一）%s' % json.dumps(pure_ctrl, ensure_ascii=False))

    def eval_variant(der_lab, class_cap=None):
        """der_lab>0 的点用档标签（45/46），其余用原纯林标签 41-44。
        **注意**：不能写 np.where(isin(cls6,PURE), cls6, der_lab) —— 推导点的原标签本来就是纯林，
        那样写会把 45/46 标签整批丢掉（首版曾踩此坑：f1_45 恒为 0）。
        class_cap: 若给定，则每类最多抽 class_cap 点（类别均衡），使 macro-F1 可跨档比较。
        """
        mix_idx = np.where(der_lab > 0)[0]
        idx = np.concatenate([pure_idx, mix_idx])
        y = p.cls6.values[idx].astype(np.int16).copy()
        y[len(pure_idx):] = der_lab[mix_idx]
        assert len(pure_idx) + len(mix_idx) == len(y)
        if class_cap:
            rng2 = np.random.default_rng(BLOCK_SEED + 7)
            keep = []
            for cl in CLS:
                w = np.where(y == cl)[0]
                if len(w) == 0:
                    continue
                keep.append(rng2.choice(w, size=min(class_cap, len(w)), replace=False))
            keep = np.sort(np.concatenate(keep))
            idx = idx[keep]; y = y[keep]
        Xs = X[idx]; fs = fold[idx]
        r = {'n': int(len(idx)), 'n_mix': int(len(mix_idx)),
             'n_by_cls': {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
             'amp': {}, 'eco': {}, 'v30_leaf': {}, 'pft_leaf': {}, 'nn': {}, 'axis': {}, 'v30_qun': {}}
        for cl in MIXED:
            sel = mix_idx[der_lab[mix_idx] == cl]
            if len(sel) == 0:
                continue
            r['amp'][int(cl)] = dist_tbl(p.amp2023.values[sel])
            r['eco'][int(cl)] = {k: int(v) for k, v in pd.Series(p.eco.values[sel]).value_counts().items()}
            r['v30_leaf'][int(cl)] = {int(k): int(v) for k, v in pd.Series(
                p.v30_leaf.values[sel]).value_counts().sort_index().items()}
            r['pft_leaf'][int(cl)] = {int(k): int(v) for k, v in pd.Series(
                p.pft_leaf_3maj.values[sel]).value_counts().sort_index().items()}
            qun = pd.Series(p.v30_value.values[sel]).map(
                lambda v: vat_global.get(int(v), ('', '', ''))[2] if int(v) != 65535 else '')
            r['v30_qun'][int(cl)] = {'n': int(len(qun)),
                                     'share_hongsong': round(float(qun.str.contains('红松').mean()), 4),
                                     'share_luoyesong': round(float(qun.str.contains('落叶松').mean()), 4),
                                     'share_mengguli': round(float(qun.str.contains('蒙古栎').mean()), 4),
                                     'share_maweisc': round(float(qun.str.contains('马尾松').mean()), 4)}
            sh, dd = nn_profile(sel)
            mx = sh.max(axis=1)
            r['nn'][int(cl)] = {
                'mean_nn_share': {str(c_): round(float(sh[:, j].mean()), 4) for j, c_ in enumerate(PURE)},
                'needle_share': round(float(sh[:, [0, 1]].sum(axis=1).mean()), 4),
                'broad_share': round(float(sh[:, [2, 3]].sum(axis=1).mean()), 4),
                'mean_max_share': round(float(mx.mean()), 4),
                'share_max_ge_0.8': round(float((mx >= 0.8).mean()), 4),
                'mean_entropy': round(float(entropy(sh).mean()), 4),
                'mean_nn_dist': round(float(dd.mean()), 3)}
            a = p.v30_axis.values[sel].astype(np.int16)
            b = p.pft_axis.values[sel].astype(np.int16)
            want = AX_ENF if cl == 45 else AX_DNF
            av = (a > 0) & (b > 0)
            ag = av & (a == want) & (b == want)
            r['axis'][int(cl)] = {'n_avail': int(av.sum()), 'agree': int(ag.sum()),
                                  'rate': round(float(ag.sum() / max(1, av.sum())), 4),
                                  'ci': wilson(int(ag.sum()), int(av.sum())),
                                  'v30_same_dir': round(float((a[av] == want).mean()), 4) if av.sum() else None,
                                  'pft_same_dir': round(float((b[av] == want).mean()), 4) if av.sum() else None}
        if not args.skip_rf:
            oof = np.full(len(y), -1, np.int16); oofp = np.zeros(len(y), np.float32)
            ff1 = []
            for f in range(5):
                trn = fs != f; te = fs == f
                if te.sum() == 0 or len(np.unique(y[trn])) < 2:
                    continue
                rf = RandomForestClassifier(n_estimators=args.rf_trees, n_jobs=args.n_jobs,
                                            random_state=SEED_MODEL + f)
                rf.fit(Xs[trn], y[trn])
                oof[te] = rf.predict(Xs[te]); oofp[te] = rf.predict_proba(Xs[te]).max(axis=1)
                ff1.append(float(f1_score(y[te], oof[te], labels=CLS, average='macro', zero_division=0)))
            okm = oof >= 0
            pcs, rcs, f1s, sup = precision_recall_fscore_support(y[okm], oof[okm], labels=CLS,
                                                                 zero_division=0)
            r['rf'] = {'trees': args.rf_trees, 'n_eval': int(okm.sum()),
                       'macro_f1': round(float(f1_score(y[okm], oof[okm], labels=CLS,
                                                        average='macro', zero_division=0)), 4),
                       'fold_macro_f1': [round(v, 4) for v in ff1],
                       'fold_macro_f1_mean': round(float(np.mean(ff1)), 4) if ff1 else None,
                       'fold_macro_f1_sd': round(float(np.std(ff1)), 4) if ff1 else None,
                       'oa': round(float((y[okm] == oof[okm]).mean()), 4),
                       'per_class': {int(cl): {'precision': round(float(pcs[j]), 4),
                                               'recall': round(float(rcs[j]), 4),
                                               'f1': round(float(f1s[j]), 4), 'support': int(sup[j])}
                                     for j, cl in enumerate(CLS)},
                       'mix_pred_dist': {}, 'mix_maxprob_median': {},
                       'pure_classes_as_mixed_rate': {}}
            for cl in MIXED:
                sm = der_lab[idx] == cl
                if sm.sum() == 0:
                    continue
                r['rf']['mix_pred_dist'][int(cl)] = {int(k): int(v) for k, v in
                                                     pd.Series(oof[sm]).value_counts().sort_index().items()}
                r['rf']['mix_maxprob_median'][int(cl)] = round(float(np.median(oofp[sm])), 4)
                r['rf']['pure_classes_as_mixed_rate'][int(cl)] = round(
                    float(np.isin(oof[(der_lab[idx] == 0) & np.isin(y, PURE)], [45, 46]).mean()), 4)
        return r

    vat_global = read_vat()
    variants = {'S1': np.where(T['S1'], der, 0).astype(np.int8),
                'S2': np.where(T['S2'], der, 0).astype(np.int8),
                'S3': np.where(T['S3'], der, 0).astype(np.int8),
                'legacy_base': np.where(legacy, p.cls6_orig.values, 0).astype(np.int8)}
    compare = {}
    for k in ('S1', 'S2', 'S3', 'legacy_base'):
        t = time.time()
        compare[k] = eval_variant(variants[k])
        emit('  档 %-11s n=%6d 混交=%5d %.1fs macroF1=%s f1_45=%s f1_46=%s' % (
            k, compare[k]['n'], compare[k]['n_mix'], time.time() - t,
            compare[k].get('rf', {}).get('macro_f1'),
            compare[k].get('rf', {}).get('per_class', {}).get(45, {}).get('f1'),
            compare[k].get('rf', {}).get('per_class', {}).get(46, {}).get('f1')))
    # 类别均衡口径：每类最多 2,000 点，消除"档内 45/46 占比不同"对 macro-F1 的混淆
    BAL_CAP = 2000
    compare_bal = {}
    for k in ('S1', 'S2', 'S3', 'legacy_base'):
        t = time.time()
        compare_bal[k] = eval_variant(variants[k], class_cap=BAL_CAP)
        emit('  均衡 %-11s n=%6d macroF1=%s f1_45=%s f1_46=%s %.1fs' % (
            k, compare_bal[k]['n'], compare_bal[k].get('rf', {}).get('macro_f1'),
            compare_bal[k].get('rf', {}).get('per_class', {}).get(45, {}).get('f1'),
            compare_bal[k].get('rf', {}).get('per_class', {}).get(46, {}).get('f1'),
            time.time() - t))
    # 公平对比用均衡口径；T1/T3/T4/T5 不受类别比例影响，用全量口径
    base_f1 = compare_bal['legacy_base'].get('rf', {}).get('macro_f1')
    emit('T2 基线（均衡口径，每类≤%d）= %s' % (BAL_CAP, base_f1))

    # ------------------------------------------------ 预注册裁决
    base_f1 = compare['legacy_base'].get('rf', {}).get('macro_f1')

    def axis_rate(k):
        ax = compare[k]['axis']
        na = sum(x['n_avail'] for x in ax.values()); ag = sum(x['agree'] for x in ax.values())
        return (round(ag / na, 4) if na >= 100 else None), int(na)

    base_ax, base_ax_n = axis_rate('legacy_base')

    def verdict(k):
        r = compare[k]; v = {}
        a45, a46 = r['amp'].get(45, {}), r['amp'].get(46, {})
        if a45.get('median') and a46.get('median'):
            ratio = a46['median'] / max(1e-6, a45['median'])
            v['T1_ratio'] = round(ratio, 3)
            v['T1'] = bool(a45['median'] < a46['median'] and ratio >= 2.0
                           and a45.get('share_gt500', 1) <= 0.20 and a46.get('share_gt500', 0) >= 0.50)
        else:
            v['T1'] = None; v['T1_ratio'] = None
        rfb = compare_bal[k].get('rf')
        if rfb:
            f45 = rfb['per_class'].get(45, {}).get('f1', 0)
            f46 = rfb['per_class'].get(46, {}).get('f1', 0)
            v.update({'T2_macroF1': rfb['macro_f1'], 'T2_base': base_f1, 'T2_f1_45': f45,
                      'T2_f1_46': f46, 'T2_n': compare_bal[k]['n'],
                      'T2': bool(base_f1 is not None and rfb['macro_f1'] >= base_f1 - 0.02
                                 and f45 >= 0.30 and f46 >= 0.30)})
        else:
            v['T2'] = None
        ar, an = axis_rate(k)
        v['T3_rate'] = ar; v['T3_n'] = an; v['T3_base_rate'] = base_ax
        v['T3'] = None if (ar is None or base_ax is None) else bool(ar >= base_ax)
        v['T4_45'] = int((variants[k] == 45).sum()); v['T4_46'] = int((variants[k] == 46).sum())
        v['T4'] = bool(v['T4_45'] >= 500 and v['T4_46'] >= 500)
        ecos = {e for cl in MIXED for e, c_ in r['eco'].get(int(cl), {}).items() if c_ >= 30}
        v['T5_n_eco'] = len(ecos); v['T5'] = bool(len(ecos) >= 6)
        return v

    verdicts = {k: verdict(k) for k in ('S1', 'S2', 'S3', 'legacy_base')}
    # T3 在 legacy 上仅 282 个可判点，按定义 base==base 恒真；标注其规模不足
    verdicts['legacy_base']['T3_note'] = ('legacy 可判点仅 %d（<300），T3 用它在定义上是 base==base，'
                                          '不构成对 legacy 的支持证据' % base_ax_n)
    emit('裁决 %s' % json.dumps(verdicts, ensure_ascii=False))

    tiers = ('S1', 'S2', 'S3')
    hard_ok = [k for k in tiers if verdicts[k]['T1'] and verdicts[k]['T2']]
    if hard_ok:
        pick = sorted(hard_ok, key=lambda k: (verdicts[k]['T4'], verdicts[k]['T3_rate'],
                                              compare[k]['n_mix']), reverse=True)
        chosen = pick[0]
        reason = ('通过 T1&T2 硬判据的档：%s；按 T4(45/46 各≥500) → T3(独立针叶轴同向率) → 规模 '
                  '排序取 %s（%d 点；45 %d / 46 %d）' % (
                      pick, chosen, compare[chosen]['n_mix'],
                      verdicts[chosen]['T4_45'], verdicts[chosen]['T4_46']))
    else:
        soft = [k for k in tiers if verdicts[k]['T1'] or verdicts[k]['T2']]
        chosen = sorted(soft or list(tiers),
                        key=lambda k: (verdicts[k]['T4'], compare[k]['n_mix']), reverse=True)[0]
        reason = '无档同时通过 T1&T2；按可用规模退让取 %s，须 L3 影像验证后才能进生产' % chosen
    emit('推荐档 = %s（%s）' % (chosen, reason))

    # ------------------------------------------------ 融合
    lab_c = variants[chosen]
    mixed_mask = lab_c > 0
    orig = p.cls6_orig.values.astype(np.int8)
    cls_final = orig.copy()                       # keep_pure：纯林标签不被覆盖
    cls_final[mixed_mask & legacy] = lab_c[mixed_mask & legacy]
    cls_final[~mixed_mask & legacy] = 0           # 无法拆分的混交点 → 待定
    cls_train = orig.copy()                       # prefer_mixed：档内点一律 45/46
    cls_train[mixed_mask] = lab_c[mixed_mask]
    cls_train[~mixed_mask & legacy] = 0

    tier_mask = np.array([';'.join([k for k in ('S1', 'S2', 'S3') if T[k][i]]) for i in range(n)],
                         dtype=object)
    p['cls6_final'] = cls_final.astype(np.int8)
    p['cls6_train'] = cls_train.astype(np.int8)
    p['tier_mask'] = tier_mask
    p['mixed_tier'] = np.where(T[chosen], chosen, np.where(legacy, 'legacy', 'none')).astype(object)
    # 冲突标记（两条口径，见 fusion.conflict_rule）
    #   also_mixed_window：**最终生效**的冲突——原标签是纯林 41-44，但在本档下会被改成 45/46
    #                       → 下游训练 41-44 时可据此剔除（keep_pure 口径下这些点仍是 41-44）
    #   also_mixed_any   ：只要被 derived 规则推过就标（宽口径，含未进本档的 4.8 万点）
    p['also_mixed_window'] = (mixed_mask & p.cls6_orig.isin(PURE).values).astype(np.int8)
    p['also_mixed_any'] = ((der > 0) & p.cls6_orig.isin(PURE).values).astype(np.int8)
    p['legacy_mixed'] = legacy.astype(np.int8)
    p['relabeled'] = ((cls_train != cls_final) & (cls_train > 0)).astype(np.int8)

    # legacy 三方案
    ev = p.needle_evergreen_frac.values.astype('float64')
    leg_der = legacy & (der > 0)
    leg_ok = legacy & leg_der & np.isfinite(ev)
    leg_re = np.where(ev >= EV_DOM, 45, np.where(ev <= 1 - EV_DOM, 46, 0)).astype(np.int8)
    legacy_variants = {
        'A_chosen_tier_then_zero': {
            'n_45': int((legacy & (lab_c == 45)).sum()),
            'n_46': int((legacy & (lab_c == 46)).sum()),
            'n_zero': int((legacy & ~mixed_mask).sum())},
        'B_reclass_by_window_needle_evergreen_frac': {
            'n_45': int((leg_ok & (leg_re == 45)).sum()),
            'n_46': int((leg_ok & (leg_re == 46)).sum()),
            'n_zero': int((legacy & ~(leg_ok & (leg_re > 0))).sum())},
        'C_keep_legacy_45_46_as_is': {
            'n_45': int((orig[legacy] == 45).sum()), 'n_46': int((orig[legacy] == 46).sum()),
            'n_zero': 0},
        'n_legacy': int(legacy.sum()),
        'n_legacy_with_window_evidence': int(leg_der.sum()),
        'n_legacy_in_chosen_tier': int((legacy & T[chosen]).sum()),
        'n_legacy_splittable_by_ev_frac': int((leg_ok & (leg_re > 0)).sum()),
        'n_legacy_no_needle_px_in_window': int((legacy & (p.n_needle_win.values == 0)).sum()),
        'n_legacy_no_broad_px_in_window': int((legacy & (p.n_broad_win.values == 0)).sum()),
        'n_legacy_center_in_9192': int((legacy & is_ctr_mix).sum()),
        'chosen_rule': ('A：legacy 点只有同时满足 %s 档 + derived 时才改标为对应 45/46；'
                        '不满足者 cls6_final=0（待定），但保留 legacy_mixed=1 与 cls6_orig=91/92 可追溯。'
                        'B/C 供下游按需切换。' % chosen),
        'why_not_C': ('F2a 实测：原 91（名义"常绿混交"）amp 中位 697.3，紧贴落叶阔叶 743.3；'
                      '按 doc84 §4.2 原规则只有 11/1913 点像常绿 → 奇偶不携带常绿/落叶信息，'
                      'C 不能作 45/46 真值（实测 45/46 的 amp 比值 0.926 < 1，方向都是落叶）。'),
        'why_not_B_first': ('B 用窗口 needle_evergreen_frac 拆分，但可用者仅 %d/%d 点'
                            '（其余窗口内无针叶像元），且该判据与 45/46 定义同源，属循环；'
                            '故只作备选。' % (int((leg_ok & (leg_re > 0)).sum()), int(legacy.sum()))),
    }
    emit('legacy 方案 %s' % json.dumps(legacy_variants, ensure_ascii=False))

    # ------------------------------------------------ 生态合理性过滤
    bio01 = p.bio01.values.astype('float64') / T_SCALE
    bio06 = p.bio06.values.astype('float64') / T_SCALE
    clim_miss = ~np.isfinite(bio01) | ~np.isfinite(bio06)
    m43 = cls_final == 43; m42 = cls_final == 42
    v1 = m43 & (bio06 < -20.0)
    v2 = m43 & (bio01 < 0.0) & ~v1
    v3 = m42 & (bio01 > 22.0)
    viol = np.array([''] * n, dtype=object)
    viol[v1] = '43_bio06_lt_-20C'; viol[v2] = '43_bio01_lt_0C'; viol[v3] = '42_bio01_gt_22C'
    viol[clim_miss & (viol == '')] = 'clim_missing'
    p['eco_ok'] = (~(v1 | v2 | v3)).astype(np.int8)
    p['eco_violation'] = viol
    p.loc[v1 | v2 | v3, 'cls6_final'] = 0
    p.loc[v1 | v2 | v3, 'cls6_train'] = 0
    # 剔除后重算标记（标记只对仍带标签的点有意义）
    p['high_elev_enf'] = (((p.cls6_final.values == 41) & (p.eco.values == QINGZANG)
                           & (p.elev_m.values >= HIGH_ELEV_M))).astype(np.int8)
    p['south_needle_flag'] = ((np.isin(p.cls6_final.values, [41, 42])
                               & p.prov.isin(SOUTH_NEEDLE_PROV).values)).astype(np.int8)

    listed = p.loc[v1, ['uid', 'lon', 'lat', 'prov', 'eco', 'layer', 'src_primary',
                        'cls6_orig', 'bio01', 'bio06', 'elev_m']].copy()
    listed['bio01_C'] = (listed.bio01 / T_SCALE).round(2)
    listed['bio06_C'] = (listed.bio06 / T_SCALE).round(2)
    eco_filter = {
        'unit_note': 'bio01/bio06 存储为 0.1 ℃（×10）；判据用 doc84 §4.4 原值（−20 ℃ / 0 ℃ / 22 ℃）',
        'unit_proof': 'rule_1 命中 41 点 = 任务给出的已知越界清单（川 22 / 甘 10 / 青 9，全 layer=clean）',
        'rule_1_43_bio06_lt_-20C': int(v1.sum()),
        'rule_1_by_prov': {k: int(x) for k, x in p.loc[v1, 'prov'].value_counts().items()},
        'rule_1_by_layer': {k: int(x) for k, x in p.loc[v1, 'layer'].value_counts().items()},
        'rule_2_43_bio01_lt_0C': int(v2.sum()),
        'rule_2_by_prov': {k: int(x) for k, x in p.loc[v2, 'prov'].value_counts().items()},
        'rule_3_42_bio01_gt_22C': int(v3.sum()),
        'rule_3_by_prov': {k: int(x) for k, x in p.loc[v3, 'prov'].value_counts().items()},
        'total_removed': int((v1 | v2 | v3).sum()),
        'removed_by_cls': {int(k): int(x) for k, x in pd.Series(cls_final[v1 | v2 | v3]).value_counts().items()},
        'clim_missing_n': int(clim_miss.sum()),
        'clim_missing_policy': 'bio01/bio06 缺失 → 无法判定，eco_ok=1 但 eco_violation=clim_missing（不剔除）',
        'flag_high_elev_enf': int(p.high_elev_enf.sum()),
        'flag_high_elev_enf_by_prov': {k: int(x) for k, x in
                                       p.loc[p.high_elev_enf == 1, 'prov'].value_counts().items()},
        'flag_south_needle': int(p.south_needle_flag.sum()),
        'flag_south_needle_by_prov': {k: int(x) for k, x in
                                      p.loc[p.south_needle_flag == 1, 'prov'].value_counts().items()},
        'flag_south_needle_by_cls': {int(k): int(x) for k, x in
                                     p.loc[p.south_needle_flag == 1, 'cls6_final'].value_counts().items()},
        'known_41_outliers_n': int(len(listed)),
        'known_41_outliers_all': listed.to_dict('records'),
    }
    emit('生态过滤：43_bio06<-20C %d / 43_bio01<0C %d / 42_bio01>22C %d → 共剔 %d；'
         'high_elev_enf %d；south_needle %d' % (
             v1.sum(), v2.sum(), v3.sum(), (v1 | v2 | v3).sum(),
             p.high_elev_enf.sum(), p.south_needle_flag.sum()))

    # ------------------------------------------------ L1 构造性自洽
    def l1_check(lab_col, allow_relabel):
        lab = p[lab_col].values
        sc = p.src_code.values
        out = {}
        for cl in PURE:
            m = lab == cl
            bad = m & (sc != cl)
            out['%d_src_code_eq_cls' % cl] = {
                'n': int(m.sum()), 'violations': int(bad.sum()),
                'violation_src_codes': {int(k): int(v) for k, v in
                                        pd.Series(sc[bad]).value_counts().head(5).items()}}
        ev2 = p.needle_evergreen_frac.values.astype('float64')
        for cl, okf, desc in ((45, lambda x: x >= EV_DOM, 'needle_evergreen_frac>=0.60'),
                              (46, lambda x: x <= 1 - EV_DOM, 'needle_evergreen_frac<=0.40')):
            m = lab == cl
            nanm = m & ~np.isfinite(ev2)
            bad = m & np.isfinite(ev2) & ~okf(np.nan_to_num(ev2, nan=0.5))
            out['%d_%s' % (cl, desc)] = {
                'n': int(m.sum()), 'unverifiable_nan': int(nanm.sum()), 'violations': int(bad.sum()),
                'min': (round(float(np.nanmin(ev2[m])), 4) if (m & np.isfinite(ev2)).any() else None),
                'max': (round(float(np.nanmax(ev2[m])), 4) if (m & np.isfinite(ev2)).any() else None),
                'min_incl_nan_as_violation_possible': int(nanm.sum())}
        # 45/46 定义互斥性 + 与 v30 群系交叉（非平凡检验）
        out['45_46_mutually_exclusive'] = 0
        v30_axis = p.v30_axis.values
        for cl in MIXED:
            m = lab == cl
            av = m & (v30_axis > 0)
            want = AX_ENF if cl == 45 else AX_DNF
            out['diag_%d_v30_axis_same_dir' % cl] = (
                round(float((v30_axis[av] == want).mean()), 4) if av.sum() else None)
            out['diag_%d_v30_axis_n' % cl] = int(av.sum())
        out['pure_label_overwritten'] = int(((lab > 0) & np.isin(lab, PURE) & (lab != orig)).sum())
        out['mixed_label_without_tier'] = int((np.isin(lab, MIXED)
                                               & (p.mixed_tier.values == 'none')).sum())
        out['labeled_without_full_AEF64'] = int(((lab > 0) & ~np.isfinite(X).all(axis=1)).sum())
        out['counts'] = {int(k): int(v) for k, v in pd.Series(lab).value_counts().sort_index().items()}
        vk = [k for k in out if isinstance(out[k], dict) and 'violations' in out[k]]
        nv = sum(out[k]['violations'] for k in vk) + out['labeled_without_full_AEF64']
        if not allow_relabel:
            nv += (out['pure_label_overwritten'] + out['mixed_label_without_tier'])
        else:
            nv += 0
        out['n_violations_total'] = int(nv)
        out['PASS'] = bool(nv == 0)
        out['note'] = ('纯林 41-44 的 src_code 检查是**构造性**的（cls6 本就由源码映射），'
                       '只能证明映射表未被污染；45/46 的 needle_evergreen_frac 与 v30 群系交叉才是非平凡检验。'
                       'mixed_label_without_tier 在 cls6_train 口径下不适用（该口径本身就是"无档不加混交标签"）。')
        return out

    L1 = {'cls6_final': l1_check('cls6_final', allow_relabel=False),
          'cls6_train': l1_check('cls6_train', allow_relabel=True)}
    emit('L1 final PASS=%s / train PASS=%s' % (L1['cls6_final']['PASS'], L1['cls6_train']['PASS']))

    # ------------------------------------------------ L2 源间一致率
    def l2_block(lab_col, sources, label, circular=False):
        lab = p[lab_col].values
        tgt = np.where(np.isin(lab, [41, 42]), LEAF_NEEDLE,
                       np.where(np.isin(lab, [43, 44]), LEAF_BROAD,
                                np.where(np.isin(lab, MIXED), LEAF_MIXED, LEAF_NA))).astype(np.int8)
        V = np.stack([p[k].values.astype(np.int8) for k in sources], axis=1)
        av = (V != LEAF_NA) & (tgt[:, None] != LEAF_NA)
        eq = (V == tgt[:, None]) & av
        n_av = av.sum(axis=1); n_eq = eq.sum(axis=1)
        valid = n_av >= min(2, len(sources))
        unan = valid & (n_eq == n_av)
        maj = valid & (n_eq * 2 > n_av)
        res = {'label': label, 'sources': list(sources), 'circular': circular,
               'overall': {'n_labeled': int((tgt != LEAF_NA).sum()), 'n_eval': int(valid.sum()),
                           'unanimous': int(unan.sum()),
                           'unanimous_rate': round(float(unan.sum() / max(1, valid.sum())), 4),
                           'unanimous_ci': wilson(int(unan.sum()), int(valid.sum())),
                           'majority': int(maj.sum()),
                           'majority_rate': round(float(maj.sum() / max(1, valid.sum())), 4),
                           'majority_ci': wilson(int(maj.sum()), int(valid.sum()))},
               'by_class': {}, 'by_eco': {}, 'by_prov': {}}
        for cl in CLS:
            m = (lab == cl); mm = m & valid
            k = int(unan[mm].sum()); nn = int(mm.sum()); k2 = int(maj[mm].sum())
            res['by_class'][int(cl)] = {
                'n': int(m.sum()), 'n_eval': nn, 'na_rate': round(1 - nn / max(1, int(m.sum())), 4),
                'unanimous': k, 'unanimous_rate': round(k / max(1, nn), 4), 'ci': wilson(k, nn),
                'majority': k2, 'majority_rate': round(k2 / max(1, nn), 4), 'majority_ci': wilson(k2, nn)}
        tot = sum(res['by_class'][cl]['n'] for cl in res['by_class'])
        res['weighted_overall'] = round(sum(res['by_class'][cl]['unanimous_rate'] * res['by_class'][cl]['n']
                                            for cl in res['by_class']) / max(1, tot), 4)
        for gk in ('eco', 'prov'):
            gi = {}
            for name, sub in p.groupby(gk, sort=False).indices.items():
                m = np.zeros(n, bool); m[sub] = True
                mm = m & valid
                if mm.sum() < 20:
                    continue
                k = int(unan[mm].sum()); nn = int(mm.sum())
                pc = {}
                for cl in CLS:
                    mc = mm & (lab == cl)
                    if mc.sum() >= 20:
                        pc[str(cl)] = {'n': int(mc.sum()), 'rate': round(float(unan[mc].mean()), 4),
                                       'ci': wilson(int(unan[mc].sum()), int(mc.sum()))}
                gi[str(name)] = {'n_eval': nn, 'unanimous_rate': round(k / nn, 4),
                                 'ci': wilson(k, nn), 'per_class': pc}
            res['by_%s' % gk] = gi
        return res

    def l2_axis(lab_col):
        """常绿/落叶针叶轴的源间/源-标签一致率（L2 叶型轴无法覆盖的第三轴，单列）。
        编码：AX_NA=0 / AX_ENF=1 / AX_DNF=2；标签 45 期望 ENF、46 期望 DNF。"""
        lab = p[lab_col].values
        a = p.v30_axis.values.astype(np.int16); b = p.pft_axis.values.astype(np.int16)
        av = (a > 0) & (b > 0)
        res = {'label': '针叶轴 45/46：30m植被(落叶松→DNF) × PFT(ENF vs DNF) 同向 + 与标签同向',
               'encoding': {'0': 'NA', '1': 'ENF(常绿)', '2': 'DNF(落叶)'},
               'n_both_avail': int(av.sum()),
               'two_src_agree_rate': round(float((a[av] == b[av]).mean()), 4),
               'two_src_agree_ci': wilson(int((a[av] == b[av]).sum()), int(av.sum())),
               'by_class': {}}
        for cl in CLS:
            m = (lab == cl) & av
            if m.sum() == 0:
                continue
            want = AX_ENF if cl == 45 else (AX_DNF if cl == 46 else None)
            rec = {'n': int(m.sum())}
            if want is not None:
                rec.update({'v30_same': round(float((a[m] == want).mean()), 4),
                            'pft_same': round(float((b[m] == want).mean()), 4),
                            'both_same': round(float(((a[m] == want) & (b[m] == want)).mean()), 4),
                            'both_same_ci': wilson(int(((a[m] == want) & (b[m] == want)).sum()), int(m.sum()))})
            else:
                # 纯阔叶/纯针叶类不属此轴，仅记源-源一致率作对照
                rec.update({'two_src_agree': round(float((a[m] == b[m]).mean()), 4)})
            res['by_class'][int(cl)] = rec
        return res

    # FCS10 × 30m 的实测一致率（用户 brief 把 0.9837 归给这一对，需澄清）
    m_fc30 = p.fcs10_leaf.values != LEAF_NA
    p_av = m_fc30 & (p.v30_leaf.values != LEAF_NA)
    L2 = {
        'A_3src_pixel': l2_block('cls6_final', ['fcs10_leaf', 'v30_leaf', 'pft_leaf_3maj'],
                                 'A 三方：FCS10(中心像元) × 30m植被 × PFT(3×3 多数票)'),
        'B_2src_fcs10_pft': l2_block('cls6_final', ['fcs10_leaf', 'pft_leaf_3maj'],
                                     'B 剔除同源后的两方：FCS10 × PFT(3×3 多数票)'),
        'C_2src_independent': l2_block('cls6_final', ['v30_leaf', 'pft_leaf_3maj'],
                                       'C 两个独立于 FCS10 的源：30m植被 × PFT(3×3 多数票)'),
        'D_3src_mixcode': l2_block('cls6_final', ['fcs10_leaf_mixcode', 'v30_leaf', 'pft_leaf_3maj'],
                                   'D 三方（FCS10 用 91/92 混交码证据票）× 30m × PFT(多数票)'),
        'E_3src_window_circular': l2_block('cls6_final', ['fcs10_leaf_win', 'v30_leaf', 'pft_leaf_3maj'],
                                           'E 三方（FCS10 用窗口 needle+broad 票）——与 45/46 定义循环，仅参考',
                                           circular=True),
        'F_v30_pft_pointwise': l2_block('cls6_final', ['v30_leaf', 'pft_leaf_pt'],
                                        'F 30m植被 × PFT(点值)'),
        'F2_3src_anyboth_OLD': l2_block('cls6_final', ['fcs10_leaf', 'v30_leaf', 'pft_leaf_3'],
                                        'F2 旧口径对照：PFT 用"两类并存→mixed"（会把 mixed 变默认类，'
                                        '190,994/472,307=40% 成 mixed），**不采用**'),
        'H_axis_2src': l2_axis('cls6_final'),
        'same_source_check': {
            'fcs10_x_v30_leaf_agree': round(float((p.fcs10_leaf.values[p_av] == p.v30_leaf.values[p_av]).mean()), 4),
            'fcs10_x_v30_n': int(p_av.sum()),
            'doc84_fact': 'doc84 §2.4：30 m 植被类型 = 100 万植被图的栅格版（同源，一致率 0.9837）；'
                          '该 0.9837 不是 FCS10×30m 的一致率',
            'brief_correction': '用户 brief 把 0.9837 记为"FCS10 × 30m 同源"，与 doc84 不符；'
                                '实测 FCS10×30m 叶型一致率见 fcs10_x_v30_leaf_agree。'
                                '无论哪种口径，本轮都只把 30 m 栅格计一票、不把矢量植被图另计一票，'
                                '故不存在同源重复计票；偏乐观主要来自"年代错配 + 轴不可判"。',
        },
    }
    # 生活型 4 类口径（仅 41-44）
    hab = np.where(np.isin(p.cls6_final.values, PURE), p.cls6_final.values, 0).astype(np.int8)
    a4 = p.habit30.values.astype(np.int8); b4 = p.pft_habit4.values.astype(np.int8)
    m4 = (hab > 0) & (a4 > 0) & (b4 > 0)
    bc4 = {}
    for cl in PURE:
        mm = m4 & (hab == cl); k = int(mm.sum())
        bc4[int(cl)] = {'n': k,
                        'fcs10_v30': round(float((hab[mm] == a4[mm]).mean()), 4) if k else None,
                        'fcs10_pft': round(float((hab[mm] == b4[mm]).mean()), 4) if k else None,
                        'v30_pft': round(float((a4[mm] == b4[mm]).mean()), 4) if k else None,
                        'all3': round(float(((hab[mm] == a4[mm]) & (hab[mm] == b4[mm])).mean()), 4) if k else None,
                        'all3_ci': wilson(int(((hab[mm] == a4[mm]) & (hab[mm] == b4[mm])).sum()), k)}
    L2['G_habit4_pure'] = {
        'label': '生活型 4 类口径（FCS10 × 30m植被[落叶松→DNF] × PFT），仅 41-44',
        'n_eval': int(m4.sum()),
        'fcs10_v30': round(float((hab[m4] == a4[m4]).mean()), 4),
        'fcs10_pft': round(float((hab[m4] == b4[m4]).mean()), 4),
        'v30_pft': round(float((a4[m4] == b4[m4]).mean()), 4),
        'all3': round(float(((hab[m4] == a4[m4]) & (hab[m4] == b4[m4])).mean()), 4),
        'all3_ci': wilson(int(((hab[m4] == a4[m4]) & (hab[m4] == b4[m4])).sum()), int(m4.sum())),
        'by_class': bc4}
    L2['headline'] = {
        'A_3src_all6': L2['A_3src_pixel']['overall']['unanimous_rate'],
        'B_2src_fcs10_pft': L2['B_2src_fcs10_pft']['overall']['unanimous_rate'],
        'C_independent_2src': L2['C_2src_independent']['overall']['unanimous_rate'],
        'D_3src_mixcode': L2['D_3src_mixcode']['overall']['unanimous_rate'],
        'G_habit4_all3_pure': L2['G_habit4_pure']['all3'],
        'H_axis_2src': L2['H_axis_2src']['two_src_agree_rate'],
        'A_3src_pure_only': round(float(np.mean([
            L2['A_3src_pixel']['by_class'][cl]['unanimous_rate'] for cl in PURE if
            L2['A_3src_pixel']['by_class'][cl]['n_eval'] > 0])), 4),
        'A_3src_mixed_only': round(float(np.mean([
            L2['A_3src_pixel']['by_class'][cl]['unanimous_rate'] for cl in MIXED if
            L2['A_3src_pixel']['by_class'][cl]['n_eval'] > 0])), 4),
        'recommended_primary': 'A（六类统一口径，主口径）；B（剔除 FCS10 后的两方）；'
                               'G（生活型 4 类口径，仅 41-44）；H（常绿/落叶针叶轴）；'
                               '45/46 另看 D 与 H',
    }
    L2['caveat'] = (
        '同源关系（doc84 §2.4）：**30 m 植被类型 = 100 万植被图的栅格版（一致率 0.9837）**，与 FCS10 不同源；'
        '本轮只把栅格版计一票 → 无重复计票。L2 仍系统性偏乐观/偏保守的原因：'
        '(i) 植被图为 1980s–90s 地面调查，与 2023 现状年代错配；'
        '(ii) **PFT 断于 2013**，与 2023 差 10 年；'
        '(iii) **30m 与 PFT 只能投叶型轴（needle/broad），不能判常绿/落叶** → 45 vs 46 这一轴在 A/B/C 口径下'
        '结构上不可验证（只能看 H 与 amp 物候）；'
        '(iv) A 口径下 derived 点中心像元按构造必为纯针叶或纯阔叶（只有 0.3%/2.7% 真 91/92），'
        'FCS10 票对 45/46 机械反对 → “混交类 A 一致率≈0” 是定义性结果，不是数据缺陷；'
        '(v) PFT 3×3 若用”两类并存→mixed”会把 mixed 变成默认类（40% 全池），本轮改**多数票**。'
        '所以 41-44 用 A/G，45/46 用 H+D；真精度仍须 L3。'
        '另：用户 brief 把 0.9837 记为”FCS10×30m 同源”，与 doc84 不符 —— 实测 FCS10×30m 叶型一致率 '
        '见 same_source_check.fcs10_x_v30_leaf_agree。')
    emit('L2 A=%.4f B=%.4f C=%.4f D=%.4f G=%.4f H=%.4f（41-44 子集 A=%.4f）' % (
        L2['headline']['A_3src_all6'], L2['headline']['B_2src_fcs10_pft'],
        L2['headline']['C_independent_2src'], L2['headline']['D_3src_mixcode'],
        L2['headline']['G_habit4_all3_pure'], L2['headline']['H_axis_2src'],
        L2['headline']['A_3src_pure_only']))

    # ------------------------------------------------ 分布核对（doc84 §4.4）
    req = {'东北': [41, 42, 44, 45, 46], '华北': [41, 42, 44, 45, 46], '西北': [41, 42, 44],
           '青藏': [41, 42], '西南': [41, 42, 43, 44, 45], '华中': [41, 43, 44, 45, 46],
           '华东': [41, 43, 44, 45], '华南': [43, 44, 45, 46]}
    distc = {}
    for e, cll in req.items():
        cnt = {int(cl): int(((p.eco.values == e) & (p.cls6_final.values == cl)).sum()) for cl in CLS}
        distc[e] = {'expect_classes': cll, 'counts': cnt,
                    'present': int(sum(1 for cl in cll if cnt[cl] > 0)), 'n_expected': len(cll),
                    'total': int(sum(cnt.values())),
                    'below_100': [int(cl) for cl in cll if 0 < cnt[cl] < 100]}

    # ------------------------------------------------ 输出
    keep_cols = (['uid', 'lon', 'lat', 'cls6_final', 'cls6_train', 'cls6_orig', 'mixed_tier', 'tier_mask',
                  'also_mixed_window', 'also_mixed_any', 'legacy_mixed', 'relabeled',
                  'high_elev_enf', 'south_needle_flag', 'eco_ok', 'eco_violation', 'prov', 'eco'] + AEF +
                 ['amp2023', 'djf2023', 'elev_m', 'slope_deg', 'aspect_deg', 'bio01', 'bio06',
                  'p_needle', 'p_forest', 'needle_evergreen_frac', 'frac_needle', 'tier', 'src_primary',
                  'layer', 'code_master', 'src_code', 'derived_as', 'center_code', 'v30_value', 'v30_group',
                  'habit30', 'v30_leaf', 'v30_axis', 'pft_leaf_pt', 'pft_leaf_3', 'pft_habit4',
                  'pft_axis', 'pft_needle_3x3', 'pft_broad_3x3', 'fcs10_leaf_win', 'fcs10_leaf_mixcode'])
    out = p[keep_cols].copy()
    for c_ in ('cls6_final', 'cls6_train', 'cls6_orig'):
        out[c_] = out[c_].astype(np.int8)
    out.to_parquet(OUT_PQ, index=False)
    emit('写 %s：%d × %d（%.1f MB）' % (OUT_PQ, len(out), out.shape[1], os.path.getsize(OUT_PQ) / 1e6))

    # f3_class_compare.csv
    rows = []
    for k in ('S1', 'S2', 'S3', 'legacy_base'):
        r = compare[k]; rf = r.get('rf', {}); rb = compare_bal[k]; rfb = rb.get('rf', {})
        row = {'level': 'tier', 'tier': k, 'cls': 'ALL', 'n': r['n'], 'n_mixed': r['n_mix'],
               'n_balanced': rb['n'],
               'macro_f1': rf.get('macro_f1'), 'fold_f1_mean': rf.get('fold_macro_f1_mean'),
               'fold_f1_sd': rf.get('fold_macro_f1_sd'), 'oa': rf.get('oa'),
               'f1_45': rf.get('per_class', {}).get(45, {}).get('f1'),
               'f1_46': rf.get('per_class', {}).get(46, {}).get('f1'),
               'macro_f1_bal': rfb.get('macro_f1'),
               'f1_45_bal': rfb.get('per_class', {}).get(45, {}).get('f1'),
               'f1_46_bal': rfb.get('per_class', {}).get(46, {}).get('f1'),
               'oa_bal': rfb.get('oa'),
               'amp_med_45': r['amp'].get(45, {}).get('median'),
               'amp_med_46': r['amp'].get(46, {}).get('median'),
               'amp_ratio_46_45': (round(r['amp'][46]['median'] / max(1e-6, r['amp'][45]['median']), 3)
                                   if r['amp'].get(45, {}).get('median') and r['amp'].get(46, {}).get('median') else None),
               'share_gt500_45': r['amp'].get(45, {}).get('share_gt500'),
               'share_gt500_46': r['amp'].get(46, {}).get('share_gt500'),
               'axis2src_rate': axis_rate(k)[0], 'axis2src_n': axis_rate(k)[1],
               'n_eco_covered': len({e for cl in MIXED for e, c_ in r['eco'].get(int(cl), {}).items() if c_ >= 30}),
               'verdict_T1': verdicts[k]['T1'], 'verdict_T2': verdicts[k]['T2'],
               'verdict_T3': verdicts[k]['T3'], 'verdict_T4': verdicts[k]['T4'],
               'verdict_T5': verdicts[k]['T5']}
        for cl in MIXED:
            v30s = r['v30_leaf'].get(int(cl), {}); pfs = r['pft_leaf'].get(int(cl), {})
            nnv = r['nn'].get(int(cl), {}); qv = r['v30_qun'].get(int(cl), {})
            axv = r['axis'].get(int(cl), {})
            t30 = sum(v30s.values()); tpf = sum(pfs.values())
            tag = str(cl)
            row['v30_mixed_share_' + tag] = round(v30s.get(LEAF_MIXED, 0) / max(1, t30), 4)
            row['pft_mixed_share_' + tag] = round(pfs.get(LEAF_MIXED, 0) / max(1, tpf), 4)
            row['nn_max_share_' + tag] = nnv.get('mean_max_share')
            row['nn_share_ge08_' + tag] = nnv.get('share_max_ge_0.8')
            row['nn_entropy_' + tag] = nnv.get('mean_entropy')
            row['nn_needle_share_' + tag] = nnv.get('needle_share')
            for pc_ in PURE:
                row['nn_share_%d_%s' % (pc_, tag)] = nnv.get('mean_nn_share', {}).get(str(pc_))
            row['v30q_hongsong_' + tag] = qv.get('share_hongsong')
            row['v30q_luoyesong_' + tag] = qv.get('share_luoyesong')
            row['v30q_maweisc_' + tag] = qv.get('share_maweisc')
            row['axis_both_same_' + tag] = axv.get('rate')
            row['pred_as_pure_rate_' + tag] = (
                round(1 - sum(v for kk, v in rf.get('mix_pred_dist', {}).get(int(cl), {}).items()
                              if kk in (45, 46)) / max(1, sum(rf.get('mix_pred_dist', {}).get(int(cl), {}).values())), 4)
                if rf.get('mix_pred_dist', {}).get(int(cl)) else None)
        rows.append(row)
        for cl in CLS:
            pcc = rf.get('per_class', {}).get(int(cl), {})
            rows.append({'level': 'class', 'tier': k, 'cls': cl, 'n': r['n_by_cls'].get(int(cl), 0),
                         'precision': pcc.get('precision'), 'recall': pcc.get('recall'),
                         'f1': pcc.get('f1'), 'support': pcc.get('support'),
                         'amp_med': r['amp'].get(int(cl), {}).get('median') if cl in MIXED else None,
                         'eco_dist': (json.dumps(r['eco'].get(int(cl), {}), ensure_ascii=False)
                                      if cl in MIXED else None)})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    emit('写 %s（%d 行）' % (OUT_CSV, len(rows)))

    # 华南 46 诊断（生态上中国落叶针叶=落叶松属，主要在东北/华北/西北/西南山地）
    m46 = p.cls6_final.values == 46
    diag46 = {'n': int(m46.sum()),
              'by_eco': {k: int(v) for k, v in pd.Series(p.eco.values[m46]).value_counts().items()},
              'v30_axis_same_dir_by_eco': {}}
    for e in ECO_ORDER:
        mm = m46 & (p.eco.values == e) & (p.v30_axis.values > 0)
        if mm.sum() >= 20:
            diag46['v30_axis_same_dir_by_eco'][e] = {
                'n': int(mm.sum()), 'same_dir_rate': round(float((p.v30_axis.values[mm] == AX_DNF).mean()), 4)}

    st = {
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'script': 'code/f3_fuse.py',
        'inputs': {'pool': POOL, 'candidates': CAND, 'pheno': PHENO, 'pft_dir': PFT_DIR,
                   'vat': VAT, 'pft_hits': pinfo},
        'units': {'amp2023/djf2023': 'NDVI×1000（池与 f2a 补采同口径：amp 中位差 0.00 / 相关 0.9916）',
                  'bio01/bio06': '0.1 ℃（×10）', 'bio12': 'mm',
                  'leaf_vote': {'0': 'NA', '1': 'needle', '2': 'broad', '3': 'mixed'},
                  'habit': {'41': 'ENF', '42': 'DNF', '43': 'EBF', '44': 'DBF', '45': 'mixed'},
                  'axis': {'0': '不可判', '45': '常绿针叶(ENF)', '46': '落叶针叶(DNF)'}},
        'pre_registration': PREREG,
        'tier_definitions': {
            'S1': 'derived_as>0 & center_code∈{91,92}（中心像元真是混交码）',
            'S2': 'derived_gate_strict=True（derived & center_is_forest & p_forest≥0.5 & v30_group∈{针叶林,阔叶林,针阔混交林}）',
            'S3': 'derived_as>0 & (center_code∈{91,92} | p_needle∈[0.35,0.65])',
            'legacy_base': 'FCS10 原 91/92（不拆分）',
            'derived_def': '5×5@10m 窗口 needle{71,72,81,82}≥3 且 broad{51,52,61,62}≥3、frac_needle∈[0.15,0.85]、'
                           '针叶常绿码>60%→45 / 落叶码>60%→46（doc84 §4.2 修正）'},
        'tier_relations': rel,
        'tier_compare': compare,
        'tier_compare_balanced': compare_bal,
        'tier_compare_balanced_note': (
            'tier_compare 用各档全量（混交类占比随档变化，macro-F1 不可直接跨档比较）；'
            'tier_compare_balanced 每类最多 %d 点（种子 %d），是 T2 与跨档比较的**唯一**口径。'
            % (BAL_CAP, BLOCK_SEED + 7)),
        'tier_verdicts': verdicts,
        'tier_recommendation': {'chosen': chosen, 'reason': reason, 'hard_pass': hard_ok,
                                'tier_mask_distribution': {k: int(v) for k, v in
                                                           pd.Series(tier_mask).value_counts().items()}},
        'pure_nn_baseline': pure_ctrl,
        'fusion': {
            'conflict_rule': 'keep_pure：同一点既是纯林 41-44 又被推为 45/46 → 保留原纯林标签 + also_mixed_window=1，不删除',
            'counts_cls6_final': {int(k): int(v) for k, v in pd.Series(p.cls6_final.values).value_counts().sort_index().items()},
            'counts_cls6_train': {int(k): int(v) for k, v in pd.Series(p.cls6_train.values).value_counts().sort_index().items()},
            'counts_cls6_orig': {int(k): int(v) for k, v in pd.Series(orig).value_counts().sort_index().items()},
            'n_total': int(n),
            'n_labeled_final': int((p.cls6_final.values > 0).sum()),
            'n_labeled_train': int((p.cls6_train.values > 0).sum()),
            'n_pending_final': int((p.cls6_final.values == 0).sum()),
            'tier_label_counts': {k: {'n_45': int((lab == 45).sum()), 'n_46': int((lab == 46).sum()),
                                      'n_from_pure_label': (int(((T[k]) & ~legacy).sum())
                                                            if k in T else int((legacy & (lab > 0) & ~legacy).sum())),
                                      'n_from_legacy': (int(((T[k]) & legacy).sum())
                                                        if k in T else int((legacy & (lab > 0)).sum()))}
                                  for k, lab in variants.items()},
            'also_mixed_window': int(p.also_mixed_window.sum()),
            'also_mixed_window_by_orig': {int(k): int(v) for k, v in
                                          p.loc[p.also_mixed_window == 1, 'cls6_orig'].value_counts().items()},
            'also_mixed_any': int(p.also_mixed_any.sum()),
            'also_mixed_note': ('also_mixed_window = 在本档下会被改成 45/46 的纯林点（下游训 41-44 时可剔除）；'
                                'also_mixed_any = 宽口径（只要 derived 推过就标，含未进本档的 %d 点）'
                                % (int(p.also_mixed_any.sum()) - int(p.also_mixed_window.sum()))),
            'relabeled_n': int(p.relabeled.sum()),
            'legacy_handling': legacy_variants,
            'two_label_columns': 'cls6_final=keep_pure（纯林标签不被覆盖）；cls6_train=prefer_mixed（档内点一律 45/46）；冲突点 relabeled=1',
        },
        'L1': L1,
        'L2': L2,
        'eco_filter': eco_filter,
        'distribution_check': distc,
        'diag_cls46_by_eco': diag46,
        'runtime_min': round((time.time() - T0) / 60, 2),
    }
    json.dump(jd(st), open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('写 %s（%.2f MB）' % (OUT_JSON, os.path.getsize(OUT_JSON) / 1e6))
    emit('总计 %.1f min' % ((time.time() - T0) / 60))
    emit('cls6_final %s' % st['fusion']['counts_cls6_final'])
    emit('cls6_train %s' % st['fusion']['counts_cls6_train'])


if __name__ == '__main__':
    main()
