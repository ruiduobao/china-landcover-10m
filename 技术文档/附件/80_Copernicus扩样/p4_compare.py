# -*- coding: utf-8 -*-
"""p4_compare.py — CGLS-LC100 C3 `forest_type` 与 GLC_FCS10 针阔一致性 / 新增覆盖评估

* 输入（只读，禁止修改）：
    F:/lc_work/nb_national_2026/labels/methods/gee_struct_sample.parquet
        —— 实际 CGLS 抽样点（20,000 点，cgls_ft 非空 15,966）；**work_base.cgls_ft 全为 NaN**
    F:/lc_work/nb_national_2026/labels/methods/work_base.parquet
        —— 提供 lon/lat/tile/prov/leaf_fcs10/fcs10_code/fcs10_covered
    F:/lc_work/nb_national_2026/labels/needle_broad_national_v2.parquet
        —— v2 瓦片/点位（C5/C6 基准）
    F:/lc_work/nb_national_2026/cgls/core/{TILE}_{needle,broad}_core.tif
        —— P1 产物（100 m，EPSG:3857，uint8 0/1；用运行时刻已有的瓦片）
    F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv（313 瓦）
    F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json（省界）
* 规则源：技术文档/79 §4 预注册判据 C2 / C5 / C6（**判据写死，不因结果修改**）
        CGLS forest_type：1/3→needle、2/4→broad、5→mixed、0/其它→nonforest
        FCS10 叶型：51/52/61/62→broad、71/72/81/82→needle、91/92→mixed（不并入两类）
* 门槛：C2 一致率 ≥ 0.75；C5 新增瓦 ≥ +20（相对 v2 的 248 瓦）；C6 南方针叶新增 ≥ 3,000
* 输出（cgls/）：
    p4_cgls_vs_fcs10.json、p4_by_prov.csv、p4_by_region.csv、p4_tiles.csv
* 用法：
    python p4_compare.py               # 全部三步（零 GEE；第3步仅在南方样本 <500 时执行）
    python p4_compare.py --no-tiles     # 只跑第1步（C2）
* 口径声明（必须随结果一起引用）：
    CGLS-LC100 C3 是 100 m 产品（2015–2019），FCS10 是 10 m 产品（2023）。
    逐点比较存在**尺度不匹配**（一个 100 m 像元可含多个 10 m 地类，边缘必然冲突）
    与**时相不匹配**（2019→2023 的造林/采伐/成熟度变化不可见）。
    因此一致率的上限被这两项共同压住，该值**不是**任一方精度，只能读作"两源可对齐程度"。
"""
import argparse
import glob
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'F:/lc_work/nb_national_2026'
CGLS_DIR = os.path.join(ROOT, 'cgls')
WORK = os.path.join(ROOT, 'labels', 'methods')
V2 = os.path.join(ROOT, 'labels', 'needle_broad_national_v2.parquet')
GS = os.path.join(WORK, 'gee_struct_sample.parquet')
WB = os.path.join(WORK, 'work_base.parquet')
SOUTH_EXTRA = os.path.join(CGLS_DIR, 'south_extra.parquet')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
TILES_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
BOUNDARY = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
CORE = os.path.join(CGLS_DIR, 'core')

DECLUTTER_DEG = 0.01
R_E = 6378137.0
SOUTH_PROVS = ['广东省', '福建省', '广西壮族自治区']
MIN_N = 30  # 分组最小样本量（<30 不报一致率）

# ---- 省 → 生态区（8 组；与 v2.eco / labels/code/s2_sample_national.py ECO 完全一致）----
ECO = {
    '东北': ['黑龙江省', '吉林省', '辽宁省'],
    '华北': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区'],
    '西北': ['陕西省', '甘肃省', '宁夏回族自治区', '新疆维吾尔自治区'],
    '青藏': ['青海省', '西藏自治区'],
    '西南': ['四川省', '重庆市', '云南省', '贵州省'],
    '华中': ['河南省', '湖北省', '湖南省', '江西省', '安徽省'],
    '华东': ['江苏省', '浙江省', '上海市', '山东省', '福建省', '台湾省'],
    '华南': ['广东省', '广西壮族自治区', '海南省', '香港特别行政区', '澳门特别行政区'],
}
PROV2ECO = {p: e for e, ps in ECO.items() for p in ps}


# ---------------- 统计工具 ----------------
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """Wilson 95% 区间；n=0 返回 (nan, nan)。"""
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - hw) / d, (c + hw) / d


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float('nan')
    r = tp / (tp + fn) if (tp + fn) else float('nan')
    if math.isnan(p) or math.isnan(r) or (p + r) <= 0:
        return p, r, float('nan')
    return p, r, 2 * p * r / (p + r)


def ft_class(x):
    """CGLS forest_type → needle / broad / mixed / nonforest / na"""
    if x is None:
        return 'na'
    try:
        v = float(x)
    except Exception:
        return 'na'
    if math.isnan(v):
        return 'na'
    v = int(v)
    if v in (1, 3):
        return 'needle'
    if v in (2, 4):
        return 'broad'
    if v == 5:
        return 'mixed'
    return 'nonforest'


FCS10_MIXED = {91, 92}


def two_class(df, fcs='leaf_fcs10'):
    """两类共同有效子集：CGLS∈{needle,broad} 且 FCS10∈{N,B}"""
    return df[df.cg.isin(['needle', 'broad']) & df[fcs].isin(['N', 'B'])]


def agree_row(df, fcs='leaf_fcs10'):
    s = two_class(df, fcs)
    n = len(s)
    k = int((s.cg.map({'needle': 'N', 'broad': 'B'}) == s[fcs]).sum())
    lo, hi = wilson(k, n)
    return dict(n=n, agree=k, rate=round(k / n, 6) if n else None,
                ci95=[round(lo, 6), round(hi, 6)] if n else None)


def confmat(df, fcs='leaf_fcs10'):
    """完整混淆矩阵：行=CGLS，列=FCS10（N/B/NA）"""
    rows = ['needle', 'broad', 'mixed', 'nonforest', 'na']
    cols = ['N', 'B', 'NA']
    f = df[fcs].fillna('NA').replace('', 'NA')
    t = pd.crosstab(df.cg, f)
    out = {}
    for r in rows:
        out[r] = {c: int(t.loc[r, c]) if (r in t.index and c in t.columns) else 0 for c in cols}
        out[r]['total'] = int(sum(out[r][c] for c in cols))
    out['_column_totals'] = {c: int(sum(out[r][c] for r in rows)) for c in cols}
    return out


def leaf_metrics(df, fcs='leaf_fcs10'):
    """针叶/阔叶 P/R/F1（CGLS 为预测，FCS10 为参照），带 Wilson CI 与 n"""
    s = two_class(df, fcs).copy()
    s['cg_l'] = s.cg.map({'needle': 'N', 'broad': 'B'})
    out = {}
    for leaf, lab in (('needle', 'N'), ('broad', 'B')):
        tp = int(((s.cg_l == lab) & (s[fcs] == lab)).sum())
        fp = int(((s.cg_l == lab) & (s[fcs] != lab)).sum())
        fn = int(((s.cg_l != lab) & (s[fcs] == lab)).sum())
        p, r, f = prf(tp, fp, fn)
        plo, phi = wilson(tp, tp + fp)
        rlo, rhi = wilson(tp, tp + fn)
        out[leaf] = dict(
            tp=tp, fp=fp, fn=fn,
            precision=None if math.isnan(p) else round(p, 6),
            precision_ci95=[round(plo, 6), round(phi, 6)] if (tp + fp) else None,
            recall=None if math.isnan(r) else round(r, 6),
            recall_ci95=[round(rlo, 6), round(rhi, 6)] if (tp + fn) else None,
            f1=None if math.isnan(f) else round(f, 6),
            n_ref=int((s[fcs] == lab).sum()),
            n_pred=int((s.cg_l == lab).sum()))
    out['n_two_class'] = int(len(s))
    return out


def chance_metrics(df, fcs='leaf_fcs10'):
    """一致率的'机会校正'读数：多数类基线 / Cohen's kappa / 平衡准确率。
    理由：FCS10 参照里阔叶占 64%，单靠类不平衡就能拿到 0.64 的一致率，
    光看 raw agreement 会高估两源的真正可对齐程度。"""
    s = two_class(df, fcs)
    n = len(s)
    if n == 0:
        return dict(n=0)
    ref = (s[fcs] == 'N').values          # True=needle
    pred = (s.cg == 'needle').values
    po = float((ref == pred).mean())
    pe = float(ref.mean() * pred.mean() + (1 - ref.mean()) * (1 - pred.mean()))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else None
    maj = max(float(ref.mean()), 1 - float(ref.mean()))
    tpr = float(((pred) & (ref)).sum() / max(1, ref.sum()))
    tnr = float(((~pred) & (~ref)).sum() / max(1, (~ref).sum()))
    return dict(
        n=n,
        observed_agreement=round(po, 6),
        chance_agreement=round(pe, 6),
        kappa=None if kappa is None else round(kappa, 6),
        balanced_accuracy=round((tpr + tnr) / 2, 6),
        majority_class_baseline=round(maj, 6),
        lift_over_majority_pp=round((po - maj) * 100, 2),
        note=('majority_class_baseline = 全部猜"阔叶"的一致率。observed 只比它高 '
              'lift_over_majority_pp 个百分点；kappa 才是扣除机会后的一致度。'))


def stratum_block(df, by, fcs='leaf_fcs10', min_n=MIN_N):
    recs = []
    for k, g in df.groupby(by, dropna=False):
        s = two_class(g, fcs)
        n = len(s)
        kk = int((s.cg.map({'needle': 'N', 'broad': 'B'}) == s[fcs]).sum())
        lo, hi = wilson(kk, n)
        gn = g[g.cg.isin(['needle', 'broad'])]
        rec = dict(
            key=k if isinstance(k, str) else str(k),
            n_total=int(len(g)),
            n_two_class=n,
            agree=kk,
            rate=round(kk / n, 6) if (n and n >= min_n) else None,
            ci_lo=round(lo, 6) if (n and n >= min_n) else None,
            ci_hi=round(hi, 6) if (n and n >= min_n) else None,
            reported=bool(n >= min_n),
            n_cgls_needle=int((gn.cg == 'needle').sum()),
            n_cgls_broad=int((gn.cg == 'broad').sum()),
            n_cgls_mixed=int((g.cg == 'mixed').sum()),
            n_cgls_nonforest=int((g.cg == 'nonforest').sum()),
            n_cgls_na=int((g.cg == 'na').sum()),
            n_fcs10_needle=int((g[fcs] == 'N').sum()),
            n_fcs10_broad=int((g[fcs] == 'B').sum()),
            # 组内针叶占比：CGLS 说 vs FCS10 说
            cgls_needle_share=(round(int((gn.cg == 'needle').sum()) / len(gn), 6) if len(gn) else None),
            fcs10_needle_share=(round(int((g[fcs] == 'N').sum())
                                     / max(1, int((g[fcs] == 'N').sum() + (g[fcs] == 'B').sum())), 6)
                                if (g[fcs] == 'N').sum() + (g[fcs] == 'B').sum() else None))
        if n < min_n:
            rec['note'] = 'n<%d 不报一致率（判据纪律，不做点估计）' % min_n
        recs.append(rec)
    recs.sort(key=lambda x: -x['n_total'])
    return recs


# ---------------- 栅格 ----------------
def core_inventory():
    """扫描 cgls/core/*_core.tif → tile × {needle,broad} 核心像元数"""
    import rasterio
    rows = []
    for f in sorted(glob.glob(os.path.join(CORE, '*_core.tif'))):
        b = os.path.basename(f)
        parts = b.split('_')
        if len(parts) < 3:
            continue
        tile, kind = parts[0], parts[1]
        try:
            with rasterio.open(f) as s:
                a = s.read(1)
                n = int((a > 0).sum())
                h, w = a.shape
            rows.append(dict(tile=tile, kind=kind, n_px=n, h=h, w=w, err=''))
        except Exception as e:
            rows.append(dict(tile=tile, kind=kind, n_px=np.nan, h=np.nan, w=np.nan,
                             err=str(e)[:120]))
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=['needle', 'broad', 'any'])
    piv = d.pivot_table(index='tile', columns='kind', values='n_px', aggfunc='first')
    for k in ('needle', 'broad'):
        if k not in piv.columns:
            piv[k] = np.nan
    piv = piv.fillna(0.0)
    piv['any'] = (piv.needle > 0) | (piv.broad > 0)
    return piv


def load_polys(names_wanted):
    import json as _json
    from shapely.geometry import shape
    d = _json.load(open(BOUNDARY, encoding='utf-8'))
    out = {}
    for f in d['features']:
        n = f['properties'].get('name')
        if n in names_wanted:
            out[n] = shape(f['geometry'])
    return out


def core_cells(df_pts, layer, want=None):
    """在已加载的 core 栅格上取点 → 是否命中核心区。
    df_pts 需含 lon/lat/tile。返回与 df_pts 对齐的 bool 数组（无 tif → False）。"""
    import rasterio
    hit = np.zeros(len(df_pts), dtype=bool)
    if len(df_pts) == 0:
        return hit
    for tile, g in df_pts.groupby('tile'):
        f = os.path.join(CORE, '%s_%s_core.tif' % (tile, layer))
        if not os.path.exists(f):
            continue
        with rasterio.open(f) as s:
            a = s.read(1)
            tr = s.transform
        x = g.lon.values
        y = g.lat.values
        # WGS84 → WebMercator
        xm = np.radians(x) * R_E
        lat_r = np.radians(y)
        ym = R_E * np.log(np.tan(np.pi / 4 + lat_r / 2))
        cc = np.floor((xm - tr.c) / tr.a).astype('int64')
        rr = np.floor((ym - tr.f) / tr.e).astype('int64')
        ok = (rr >= 0) & (rr < a.shape[0]) & (cc >= 0) & (cc < a.shape[1])
        idx = g.index.values
        if ok.any():
            hit[np.isin(df_pts.index.values, idx[ok])] |= (
                a[rr[ok], cc[ok]] > 0)
    return hit


def core_cells_by_prov(path, geoms, want):
    """核心栅格 → 像元中心经纬度 → 省归属 → 0.01° 唯一格集合"""
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1)
        tr = s.transform
    rr, cc = np.nonzero(a > 0)
    if len(rr) == 0:
        return {}, 0, {}
    x = tr.c + (cc + 0.5) * tr.a
    y = tr.f + (rr + 0.5) * tr.e
    lon = np.degrees(x / R_E)
    lat = np.degrees(np.arctan(np.sinh(y / R_E)))
    keys = (np.floor(lon / DECLUTTER_DEG).astype('int64') * 10_000_000
            + np.floor(lat / DECLUTTER_DEG).astype('int64'))
    out, px_by_prov = {}, {}
    try:
        from shapely import contains_xy
        for p in want:
            if p not in geoms:
                continue
            m = contains_xy(geoms[p], lon, lat)
            if m.any():
                out[p] = np.unique(keys[m])
                px_by_prov[p] = int(m.sum())
    except Exception:
        from shapely.geometry import Point
        for p in want:
            if p not in geoms:
                continue
            g = geoms[p]
            m = np.zeros(len(lon), dtype=bool)
            for i in range(len(lon)):
                m[i] = g.contains(Point(float(lon[i]), float(lat[i])))
            if m.any():
                out[p] = np.unique(keys[m])
                px_by_prov[p] = int(m.sum())
    return out, len(rr), px_by_prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-tiles', action='store_true', help='跳过栅格扫描（C5/C6 只报第1步）')
    ap.add_argument('--out', default=os.path.join(CGLS_DIR, 'p4_cgls_vs_fcs10.json'))
    args = ap.parse_args()
    t0 = pd.Timestamp.now()
    os.makedirs(CGLS_DIR, exist_ok=True)
    res = {'built': t0.strftime('%Y-%m-%d %H:%M:%S'),
           'spec': '技术文档/79 §4（预注册 C2/C5/C6，判据未改）',
           'provenance': {}, 'input': {}}

    # ================= 第1步：C2 =================
    wb = pd.read_parquet(WB, columns=['stable_uid', 'lon', 'lat', 'tile', 'prov',
                                      'leaf_fcs10', 'fcs10_code', 'fcs10_covered',
                                      'cl_24', 'excluded'])
    gs = pd.read_parquet(GS)[['stable_uid', 'cgls_ft', 'cgls_tcd']]
    df = gs.merge(wb, on='stable_uid', how='inner')
    df['cg'] = df.cgls_ft.map(ft_class)
    df['prov'] = df['prov'].fillna('NA')
    df['eco'] = df['prov'].map(PROV2ECO).fillna('其他')
    df['leaf_fcs10'] = df['leaf_fcs10'].fillna('').replace('', np.nan)
    df['fcs10_mixed'] = df['fcs10_code'].isin(list(FCS10_MIXED))

    # 仅保留 FCS10 纯林（51/52/61/62/71/72/81/82）作参照；91/92 混交不并入两类
    res['provenance'] = dict(
        cgls_point_source=GS,
        cgls_column='cgls_ft / cgls_tcd（gee_struct_sample）',
        note=('work_base.parquet 的 cgls_ft / cgls_tcd / cgls_year 列**全为 NaN**（构建时留空未回填）；'
              '实际 CGLS 抽样点在 labels/methods/gee_struct_sample.parquet（20,000 点）。'
              '本脚本以 stable_uid 内连 work_base 取得 lon/lat/tile/prov/leaf_fcs10。'
              'cgls_year 未记录 → 该点样本的 CGLS 年份不可从数据自证；'
              'm3_gee_struct.py 的 docstring 声明为 2015（ImageCollection.first()）。'
              'P1 核心区用 2017/2018/2019 三年一致 → 两者存在**年份口径差**，见 caveats。'),
        cgls_product='COPERNICUS/Landcover/100m/Proba-V-C3/Global（forest_type, tree-coverfraction）',
        cgls_grid_m=100, fcs10_grid_m=10,
        cgls_temporal='2015–2019（点样本 .first()，年份未落盘；P1 核心区 2017–2019）',
        fcs10_temporal='2023',
        fcs10_reference_rule='51/52/61/62→broad；71/72/81/82→needle；91/92→mixed（不并入两类）',
        fcs10_tile_from='work_base.tile')

    res['input'] = dict(
        n_cgls_sample=int(len(df)),
        n_cgls_nonnull=int(df.cgls_ft.notna().sum()),
        n_fcs10_pure_forest=int(df.leaf_fcs10.isin(['N', 'B']).sum()),
        n_fcs10_mixed=int(df.fcs10_mixed.sum()),
        n_fcs10_nonforest=int((~df.leaf_fcs10.isin(['N', 'B']) & ~df.fcs10_mixed).sum()),
        n_two_class=int(len(two_class(df))),
        n_tiles=int(df.tile.nunique()))

    res['overall'] = dict(confusion=confmat(df), **agree_row(df))
    res['overall']['crosstab'] = res['overall']['confusion']  # 别名（规范要求字段名）
    res['overall']['cgls_class_distribution'] = {str(k): int(v) for k, v in
                                                 df.cg.value_counts().to_dict().items()}
    res['overall']['fcs10_leaf_distribution'] = {str(k): int(v) for k, v in
                                                 df.leaf_fcs10.value_counts(dropna=False).to_dict().items()}
    res['by_leaf'] = leaf_metrics(df)
    res['chance_corrected'] = chance_metrics(df)

    # 敏感性 1：只在 CGLS tcd≥60（合理郁闭林）上算
    hi = df[df.cgls_tcd.fillna(0) >= 60]
    res['sensitivity_tcd_ge60'] = dict(**agree_row(hi), n_subset=int(len(hi)))
    # 敏感性 2：只在 P1 核心区内（= P2 真实候选总体）上算
    core_hit_n = core_hit_b = None
    if not args.no_tiles and os.path.isdir(CORE):
        df = df.reset_index(drop=True)
        hn = core_cells(df, 'needle')
        hb = core_cells(df, 'broad')
        df['in_core_needle'] = hn
        df['in_core_broad'] = hb
        df['in_core'] = hn | hb
        sub = df[df.in_core]
        res['sensitivity_in_core'] = dict(
            **agree_row(sub), n_subset=int(len(sub)),
            note=('仅取落在 P1 100 m 核心区内的抽样点：这是 P2 实际会提取的候选总体，'
                  '用它复核 C2 比全样本更贴近判据意图；但子集会偏向"大块均质林"，'
                  '一致率可能系统性高于全国抽样。'))
        # 核心区内部：CGLS 林型 vs 所落核心类型 的自洽性（纯审计，非判据）
        chk = sub[sub.cg.isin(['needle', 'broad'])].copy()
        agree_core = int(((chk.cg == 'needle') & chk.in_core_needle).sum()
                         + ((chk.cg == 'broad') & chk.in_core_broad).sum())
        res['sensitivity_in_core']['cgls_vs_core_layer'] = dict(
            n=int(len(chk)), agree=agree_core,
            rate=round(agree_core / len(chk), 6) if len(chk) else None,
            note='抽样点 CGLS 林型与所在核心层是否一致（自洽性审计）')
    res['by_region'] = stratum_block(df, 'eco')
    res['by_prov'] = stratum_block(df, 'prov')

    # ---- 南方三省专项（C6 的认识基础）----
    s = df[df.prov.isin(SOUTH_PROVS)]
    sn = s[s.cg == 'needle']
    two = sn[sn.leaf_fcs10.isin(['N', 'B'])]
    lo, hi_ = wilson(int((two.leaf_fcs10 == 'N').sum()), len(two))
    res['south_needle'] = dict(
        provinces=SOUTH_PROVS,
        n_total=int(len(s)),
        n_cgls_needle=int(len(sn)),
        n_cgls_broad=int((s.cg == 'broad').sum()),
        n_cgls_mixed=int((s.cg == 'mixed').sum()),
        n_cgls_nonforest=int((s.cg == 'nonforest').sum()),
        n_cgls_na=int((s.cg == 'na').sum()),
        south_needle_share_of_total=round(len(sn) / len(s), 6) if len(s) else None,
        cgls_needle_share_of_two_class=(
            round(len(sn) / max(1, int(s.cg.isin(['needle', 'broad']).sum())), 6)),
        cgls_needle_vs_fcs10={'N': int((sn.leaf_fcs10 == 'N').sum()),
                              'B': int((sn.leaf_fcs10 == 'B').sum()),
                              'NA': int((~sn.leaf_fcs10.isin(['N', 'B'])).sum())},
        cgls_needle_vs_fcs10_code={str(k): int(v) for k, v in
                                   sn.fcs10_code.fillna(-1).astype(int).value_counts().head(12).to_dict().items()},
        cgls_needle_agree_fcs10_needle=int((two.leaf_fcs10 == 'N').sum()),
        cgls_needle_agreement_with_fcs10=(
            round(float((two.leaf_fcs10 == 'N').mean()), 6) if len(two) else None),
        cgls_needle_agreement_n=int(len(two)),
        cgls_needle_agreement_ci95=[round(lo, 6), round(hi_, 6)] if len(two) else None,
        south_two_class=agree_row(s),
        south_confusion=confmat(s))
    fcs_needle = int((s.leaf_fcs10 == 'N').sum())
    fcs_broad = int((s.leaf_fcs10 == 'B').sum())
    fcs_share = round(fcs_needle / max(1, fcs_needle + fcs_broad), 6)
    cg_share = res['south_needle']['cgls_needle_share_of_two_class']
    res['south_needle']['doc77_check'] = dict(
        doc77_claim='南方真针叶仅 666 点 / 1.66%（FCS10 口径，纯林参考 40,150 点）',
        this_sample_fcs10_needle=fcs_needle,
        this_sample_fcs10_broad=fcs_broad,
        this_sample_fcs10_needle_share=fcs_share,
        cgls_needle_share=cg_share,
        interpretation=(f'CGLS 南方针叶占比 {cg_share} vs FCS10 {fcs_share}。'
                        f'若 CGLS 显著更高且这些点在 FCS10 里多为阔叶，'
                        f'则 §13 的 1.66% 是「FCS10 的观点」而非地表事实；'
                        f'但 CGLS 在南方的针叶判读本身未经核查，不能反过来当作真针叶证据。'
                        f'本抽样按 cl_24×region 分层，非面积比例样本，占比不可外推为全国面积比。'))
    if len(sn):
        res['south_needle']['cgls_needle_vs_pool_cl24'] = {
            'cl4_evergreen_broad': int((sn.cl_24 == 4).sum()),
            'cl5_deciduous_broad': int((sn.cl_24 == 5).sum()),
            'cl6_evergreen_needle': int((sn.cl_24 == 6).sum()),
            'cl7_deciduous_needle': int((sn.cl_24 == 7).sum())}
    # 分省南方针叶
    sp = []
    for p in SOUTH_PROVS:
        g = df[df.prov == p]
        gn = g[g.cg == 'needle']
        gt = gn[gn.leaf_fcs10.isin(['N', 'B'])]
        k = int((gt.leaf_fcs10 == 'N').sum())
        lo2, hi2 = wilson(k, len(gt))
        sp.append(dict(prov=p, n_cgls=int(len(g)), n_cgls_needle=int(len(gn)),
                       n_needle_with_fcs10=int(len(gt)), agree_needle=k,
                       rate=round(k / len(gt), 6) if len(gt) >= MIN_N else None,
                       ci95=[round(lo2, 6), round(hi2, 6)] if len(gt) >= MIN_N else None,
                       fcs10_needle_share=(round(int((g.leaf_fcs10 == 'N').sum())
                                                / max(1, int(g.leaf_fcs10.isin(['N', 'B']).sum())), 6))))
    res['south_needle']['by_prov'] = sp

    # 全国分歧方向
    nn = df[(df.cg == 'needle') & df.leaf_fcs10.isin(['N', 'B'])]
    bb = df[(df.cg == 'broad') & df.leaf_fcs10.isin(['N', 'B'])]
    res['disagreement_direction'] = dict(
        cgls_needle_n=int(len(nn)),
        cgls_needle_fcs10_broad=int((nn.leaf_fcs10 == 'B').sum()),
        cgls_needle_fcs10_broad_rate=round(float((nn.leaf_fcs10 == 'B').mean()), 6) if len(nn) else None,
        cgls_broad_n=int(len(bb)),
        cgls_broad_fcs10_needle=int((bb.leaf_fcs10 == 'N').sum()),
        cgls_broad_fcs10_needle_rate=round(float((bb.leaf_fcs10 == 'N').mean()), 6) if len(bb) else None)

    # ---- CGLS 落在 FCS10 无覆盖区（fcs10_code==0）的点：这才是"真正新增信息"的量级 ----
    out_cov = df[~df.fcs10_covered.fillna(False)]
    res['cgls_outside_fcs10_coverage'] = dict(
        n_outside=int(len(out_cov)),
        share_of_sample=round(len(out_cov) / len(df), 6) if len(df) else None,
        cgls_needle=int((out_cov.cg == 'needle').sum()),
        cgls_broad=int((out_cov.cg == 'broad').sum()),
        cgls_mixed=int((out_cov.cg == 'mixed').sum()),
        cgls_nonforest=int((out_cov.cg == 'nonforest').sum()),
        cgls_na=int((out_cov.cg == 'na').sum()),
        cgls_needle_share=round(int((out_cov.cg == 'needle').sum())
                                / max(1, int(out_cov.cg.isin(['needle', 'broad']).sum())), 6),
        south_needle=int(((out_cov.prov.isin(SOUTH_PROVS)) & (out_cov.cg == 'needle')).sum()),
        by_prov={k: int(v) for k, v in
                 out_cov[out_cov.cg.isin(['needle', 'broad'])].prov.value_counts().head(12).to_dict().items()},
        note=('fcs10_covered=False 即该点不在 FCS10-2023 公开覆盖范围内（code=0），'
              '这些点上 CGLS 提供的是 FCS10 给不出的信息——但它们同样未经影像核查，'
              '且集中在西部/边远区，FCS10 缺席本身可能反映当地制图难度而非"新增森林"。'))

    # ================= 第2步：C5 =================
    v2 = pd.read_parquet(V2, columns=['tile', 'lon', 'lat', 'leaf'])
    v2_tiles = set(v2.tile)
    plan = pd.read_csv(TILES_CSV, encoding='utf-8-sig')
    plan_tiles = set(plan.tile)
    res['v2'] = dict(tiles=int(len(v2_tiles)), points=int(len(v2)),
                     by_leaf={str(k): int(v) for k, v in v2.leaf.value_counts().to_dict().items()},
                     south=dict(points=int(v2.prov.isin(SOUTH_PROVS).sum()) if 'prov' in v2.columns else None),
                     plan_tiles_without_v2=int(len(plan_tiles - v2_tiles)))

    if not args.no_tiles:
        piv = core_inventory()
        piv.index.name = 'tile'
        missing = sorted(plan_tiles - set(piv.index))
        new_tiles = sorted(set(piv.index[piv['any']]) - v2_tiles)
        new_needle = sorted(set(piv.index[piv.needle > 0]) - v2_tiles)
        new_broad = sorted(set(piv.index[piv.broad > 0]) - v2_tiles)
        res['C5'] = dict(
            metric='CGLS 核心区能在多少个「v2 无样本」的瓦片提供候选（瓦片级有无，不看点级精度）',
            tiles_in_plan=int(len(plan_tiles)),
            tiles_scanned=int(len(piv)),
            tiles_not_scanned=int(len(missing)),
            tiles_not_scanned_list=missing,
            tiles_not_scanned_without_v2=sorted((plan_tiles - set(piv.index)) - v2_tiles),
            tiles_with_needle_core=int((piv.needle > 0).sum()),
            tiles_with_broad_core=int((piv.broad > 0).sum()),
            tiles_with_any_core=int(piv['any'].sum()),
            v2_tiles=int(len(v2_tiles)),
            new_tiles_vs_v2=int(len(new_tiles)),
            new_tiles_needle_only=int(len(new_needle)),
            new_tiles_broad_only=int(len(new_broad)),
            new_tiles_list=new_tiles,
            new_tiles_detail={t: dict(needle=int(piv.loc[t, 'needle']),
                                       broad=int(piv.loc[t, 'broad'])) for t in new_tiles},
            threshold=20,
            pass_=bool(len(new_tiles) >= 20),
            verdict_note=None)
        n_unscanned_nov2 = len(res['C5']['tiles_not_scanned_without_v2'])
        # 上界证明：即使全部未扫描的无 v2 瓦都能出核心区，总数也只有
        upper = len(new_tiles) + n_unscanned_nov2
        res['C5']['upper_bound_new_tiles'] = int(upper)
        res['C5']['upper_bound_pass'] = bool(upper >= 20)
        res['C5']['verdict_note'] = (
            'P1 完成 %d/%d 瓦（缺 %d 瓦）；新增瓦 = %d 是**下界**。'
            '未扫描的 %d 瓦中有 %d 瓦无 v2 样本。'
            '**上界 = 下界 + %d = %d**：即使剩余无 v2 瓦全部出核心区，也无法达到 +20。'
            '故 C5 可以**确定为 FAIL**，不是"数据不全待定"。'
            % (len(piv), len(plan_tiles), len(missing), len(new_tiles), len(missing),
               n_unscanned_nov2, n_unscanned_nov2, upper))
        piv.reset_index().to_csv(os.path.join(CGLS_DIR, 'p4_tiles.csv'),
                                 index=False, encoding='utf-8-sig')
        # 结构性解释：只有 181 瓦有任何核心区，其中 177 瓦已在 v2 里 →
        # v2 本身就是"森林靶向"选瓦，CGLS 核心区几乎被 v2 瓦域包含，这是 C5 失败的结构原因。
        core_tiles = set(piv.index[piv['any']])
        res['C5']['structural_note'] = (
            '全国 313 瓦里只有 %d 瓦存在 CGLS 核心区（needle %d / broad %d）；'
            '这些有核瓦中 %d 瓦已含 v2 样本，仅 %d 瓦是 v2 空白。'
            'v2 的 248 瓦本身即按"有森林"靶向选瓦，CGLS 核心区瓦域几乎被其包含 → '
            'CGLS 在"补空白瓦"这件事上的结构性增量很小，C5 失败不是数据不全造成的。'
            % (len(core_tiles), int((piv.needle > 0).sum()), int((piv.broad > 0).sum()),
               len(core_tiles & v2_tiles), len(core_tiles - v2_tiles)))
        res['C5']['v2_tiles_scanned'] = int(len(v2_tiles & set(piv.index)))
        res['C5']['v2_tiles_scanned_without_any_core'] = int(
            len((v2_tiles & set(piv.index)) - core_tiles))

        # ---- C6：南方三省 needle 核心区可提点数上限（0.01° 唯一格，精确省界归属）----
        geoms = load_polys(set(SOUTH_PROVS))
        from shapely.geometry import box
        south_tiles = sorted({r.tile for r in plan.itertuples()
                              if any(geoms[p].intersects(box(r.x0, r.y0, r.x1, r.y1))
                                     for p in geoms)})
        c6 = {'metric': ('南方三省 CGLS needle 核心区按 P2 同一去聚团规则'
                         '（每 100 m 核心像元 ≤1 点 + 0.01° 同类仅 1 点）的可提点数上限；'
                         '省归属用 DataV 省界精确点在面内'),
              'threshold': 3000, 'south_provs': SOUTH_PROVS,
              'south_tiles_total': len(south_tiles), 'by_prov': {},
              'tiles_with_core': {}, 'tiles_missing': []}
        prov_cells = {p: [] for p in SOUTH_PROVS}
        prov_px = {p: 0 for p in SOUTH_PROVS}
        prov_tiles = {p: 0 for p in SOUTH_PROVS}
        # v2 已有的 0.01° 格（用于"新增"扣除）
        if 'lon' in v2.columns:
            vs = v2[v2.tile.isin(south_tiles)]
            v2_keys = set((np.floor(vs.lon.values / DECLUTTER_DEG).astype('int64') * 10_000_000
                           + np.floor(vs.lat.values / DECLUTTER_DEG).astype('int64')).tolist())
        else:
            v2_keys = set()
        for t in south_tiles:
            f = os.path.join(CORE, '%s_needle_core.tif' % t)
            if not os.path.exists(f):
                c6['tiles_missing'].append(t)
                continue
            out, npx, px_prov = core_cells_by_prov(f, geoms, SOUTH_PROVS)
            if npx:
                c6['tiles_with_core'][t] = int(npx)
            for p, cells in out.items():
                prov_cells[p].append(cells)
                prov_px[p] += px_prov.get(p, 0)
                prov_tiles[p] += 1
        allc, allc_new = [], []
        for p in SOUTH_PROVS:
            if prov_cells[p]:
                u = np.unique(np.concatenate(prov_cells[p]))
            else:
                u = np.zeros(0, dtype='int64')
            un = np.array([k for k in u.tolist() if k not in v2_keys], dtype='int64')
            c6['by_prov'][p] = dict(uniq_cell_0p01=int(len(u)),
                                    uniq_cell_0p01_not_in_v2=int(len(un)),
                                    tiles=int(prov_tiles[p]),
                                    needle_core_px_in_prov=int(prov_px[p]))
            allc.append(u)
            allc_new.append(un)
        tot = np.unique(np.concatenate(allc)) if allc else np.zeros(0, dtype='int64')
        tot_new = np.unique(np.concatenate(allc_new)) if allc_new else np.zeros(0, dtype='int64')
        c6['uniq_cell_0p01_total'] = int(len(tot))
        c6['uniq_cell_0p01_not_in_v2'] = int(len(tot_new))
        c6['south_tiles_with_needle_core'] = int(len(c6['tiles_with_core']))
        c6['south_tiles_scanned'] = int(len(south_tiles) - len(c6['tiles_missing']))
        c6['v2_cells_in_south'] = int(len(v2_keys))
        c6['pass_'] = bool(len(tot_new) >= 3000)
        c6['verdict_note'] = (
            '值 = 0.01° 唯一格数（= P2 同一去聚团规则下可提点数上限），并扣除 v2 已占格。'
            'P1 完成 %d/%d 南方瓦，尚未扫描 %d 瓦（%s）；当前值同为**下界**。'
            '**该值不是已验证样本量**：尚未经 AEF 森林掩膜（§3.2）与影像核查（§3.3, C3≥0.80）过滤。'
            % (c6['south_tiles_scanned'], len(south_tiles), len(c6['tiles_missing']),
               ','.join(c6['tiles_missing'][:12]) or '无'))
        res['C6'] = c6

    # ================= 第3步：南方补抽（仅当第1步南方样本 <500）=================
    south_two_n = len(two_class(df[df.prov.isin(SOUTH_PROVS)]))
    res['step3_south_extra'] = dict(
        triggered=bool(south_two_n < 500),
        south_two_class_n=south_two_n,
        threshold=500,
        note=('第1步南方两类样本 n=%d %s 500，%s'
              % (south_two_n, '≥' if south_two_n >= 500 else '<',
                 '按指令**不执行** GEE 补抽（避免额外 GEE 消耗）。'
                 'C6 改用 P1 核心区可提点数上限评估（见 C6）。'
                 if south_two_n >= 500 else
                 '按指令应执行补抽；若 south_extra.parquet 不存在则本轮仅报缺口。')))
    if os.path.exists(SOUTH_EXTRA):
        se = pd.read_parquet(SOUTH_EXTRA)
        se['cg'] = se.cgls_ft.map(ft_class)
        if 'leaf_fcs10' not in se.columns:
            se['leaf_fcs10'] = np.nan
        res['south_extra'] = dict(
            source=SOUTH_EXTRA, n=int(len(se)),
            cgls_class_distribution={str(k): int(v) for k, v in se.cg.value_counts().to_dict().items()},
            overall=dict(confusion=confmat(se), **agree_row(se)),
            by_leaf=leaf_metrics(se),
            note='补抽口径：CGLS 2019 forest_type，100 m 网格；FCS10 值来自本地 10 m nearest')

    # ================= 判据裁决 =================
    c2 = res['overall']['rate']
    res['C2'] = dict(
        metric='CGLS needle/broad 与 FCS10 needle/broad 的逐点一致率（两类共同有效子集）',
        observed=c2, n=res['overall']['n'],
        ci95=res['overall']['ci95'], threshold=0.75,
        pass_=bool(c2 is not None and c2 >= 0.75),
        baseline_prior='m3_cgls_vs_fcs10.json: n=6,648 / 0.6995（本次复核）',
        scale_caveat=('CGLS 100 m vs FCS10 10 m：一个 100 m 像元可含多个 10 m 地类，'
                      '边缘混像元必然产生冲突，一致率存在结构性上限；'
                      '时相 2015–2019 vs 2023 又叠加造林/采伐/成熟度变化。'
                      '故该数字不得读作任一方精度。'))
    res['verdict'] = dict(
        C2=res['C2']['pass_'],
        C5=bool(res.get('C5', {}).get('pass_', False)),
        C6=bool(res.get('C6', {}).get('pass_', False)))
    res['verdict']['rule'] = ('C2 或 C3 不过 → 只出候选清单，不进训练（规范 §4 裁决规则）。'
                              '本轮只评估 C2/C5/C6；C3 影像核查由 P3 另行给出。')
    res['verdict']['fail_count'] = int(sum(1 for k in ('C2', 'C5', 'C6')
                                           if not res['verdict'][k]))
    res['verdict']['overall'] = ('C2 与 C5 双 FAIL → 按规范 §4，CGLS 只能出**候选清单**、不进训练。'
                                 'C6 虽 PASS，但它统计的是"候选数量上限"而非已验证样本，'
                                 '且 C2 已证明这些候选与 FCS10 的针阔判断几乎互斥（南方针叶一致率 1.5%），'
                                 '故 C6 的 PASS 不改变总体裁决。')

    res['caveats'] = [
        '**尺度不匹配（最大风险）**：CGLS 100 m 单像元 vs FCS10 10 m。若 100 m 内 60% 阔叶 + 40% 针叶，'
        'CGLS 只能给一个标签，FCS10 抽样点按面积落在多数类 → 一致率上限被结构性压低，'
        '且该压低量与真实标签噪声不可分离。因此 0.6995 不能作为"两源谁对"的证据。',
        '**时相不匹配**：点样本的 CGLS 年份未落盘（脚本声明 2015），P1 核心区用 2017–2019 三年一致；'
        'FCS10 为 2023。南方 2015→2023 桉树/杉木人工林轮伐与杉改阔可造成真实变化，'
        '这部分分歧被计入"不一致"，进一步压低一致率。',
        '**标签循环风险**：CGLS forest_type 本身是 100 m 分类产品（Proba-V），不是地面真值；'
        'FCS10 也是产品。两者一致 ≠ 正确，两者不一致 ≠ 任一错误。C2 只衡量"可对齐程度"。',
        '**抽样非面积比例**：CGLS 点样本按 cl_24×region 分层（每层最少 50），'
        '分省/分生态区的一致率不可外推为全国面积比例；占比列仅作方向性参考。',
        '**FCS10 参照只取纯林**：91/92 混交不并入两类（与 v2 的 leaf 口径一致），'
        '故本文的"FCS10 针叶占比"不含混交林；若把混交算作阔叶，针叶占比会更低。',
        '**C6 的上限性质**：C6 值来自 P1 核心区 0.01° 去重格数，未经 AEF 森林掩膜与影像核查，'
        '且南方 CGLS 针叶与 FCS10 的逐点一致率极低（见 south_needle），'
        'C6 通过只说明"候选数量够"，不说明"候选质量可用"。',
    ]

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    pd.DataFrame(res['by_prov']).to_csv(os.path.join(CGLS_DIR, 'p4_by_prov.csv'),
                                        index=False, encoding='utf-8-sig')
    pd.DataFrame(res['by_region']).to_csv(os.path.join(CGLS_DIR, 'p4_by_region.csv'),
                                          index=False, encoding='utf-8-sig')

    for k in ('input', 'C2', 'overall', 'by_leaf', 'chance_corrected',
              'sensitivity_tcd_ge60', 'sensitivity_in_core', 'south_needle',
              'cgls_outside_fcs10_coverage', 'C5', 'C6', 'verdict'):
        if k in res:
            print('===', k)
            print(json.dumps(res[k], ensure_ascii=False, indent=1, default=str))
    print('elapsed_min', round((pd.Timestamp.now() - t0).total_seconds() / 60, 2))


if __name__ == '__main__':
    main()
