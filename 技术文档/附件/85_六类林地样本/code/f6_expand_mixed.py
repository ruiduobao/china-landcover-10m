# -*- coding: utf-8 -*-
"""f6_expand_mixed.py — 从 FCS10 原生栅格撒点扫描扩大 45/46 混交候选池

* 输入
    Z:/nb_national_2026/fcs10_dl/tiles/{TILE}_fcs10.tif    FCS10 2023 全国瓦片（10 m, uint8, 0=nodata）
    F:/lc_work/nb_national_2026/forest6/forest6_v1.parquet 现有 472,307 点（去重用 lon/lat）
    F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif  (EPSG:3857) 类型组 / 群系
    F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json  省界（逐 feature）
    .../China_Forest_Dataset_v1.0/3_Forest_Change_Gain/...StartYear_1km_v1.0.tif  recent_gain 标记
* 规则源
    技术文档/84_全国六类林地样本汇总.md §4.2（2026-10-25 已修正：用邻域针叶码定义 45/46，不用 amp/djf）
      needle={71,72,81,82}  broad={51,52,61,62}  91/92 不计入分母、单独统计 n_mixedcode_win
      混交候选：n_needle>=3 且 n_broad>=3 且 frac_needle∈[0.15,0.85]
      45 = needle_eg/(needle_eg+needle_dc) > 0.60 ；46 = < 0.40 ；其余 uncertain 不入库
      附加：中心像元必须是森林码；须在中国境内；recent_gain 只标记不剔除
    撒点均匀性：100 m 格(10×10 像元)每格最多 1 点（取格心）；0.01° 内同类唯一；
              0.005° 内与现有池重合则跳过；每瓦每类 ≤300；全国每类 ≤35000
* 门槛
    单进程；逐瓦行带 4096 行；单瓦读后即释放；幂等续跑（tile 级 checkpoint）
* 输出
    F:/lc_work/nb_national_2026/forest6/mixed_expanded.parquet
    F:/lc_work/nb_national_2026/forest6/f6_expand_stats.json
    F:/lc_work/nb_national_2026/forest6/_f6_ckpt/  逐瓦 checkpoint（npz）
* 用法
    python f6_expand_mixed.py --scan            # 阶段1：扫全部瓦片，落 tile checkpoint
    python f6_expand_mixed.py --scan --tiles T1506 T2007
    python f6_expand_mixed.py --assemble        # 阶段2：合并 + 全国配额 + 属性提取 + 统计
    python f6_expand_mixed.py --scan --assemble
"""
import argparse
import gc
import json
import math
import os
import struct
import sys
import time
import warnings
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ------------------------------------------------------------------ 常量
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORK = r'F:/lc_work/nb_national_2026'
FOREST6 = os.path.join(WORK, 'forest6')
TILEDIR = r'Z:/nb_national_2026/fcs10_dl/tiles'   # 重产物在 Z 盘（F 盘只有占位记录）
PLAN_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
POOL = os.path.join(FOREST6, 'forest6_v1.parquet')
VEG30 = os.path.join(WORK, 'veg30', '中国植被类型分布_30m.tif')
VAT = VEG30 + '.vat.dbf'
DATAV = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
GAIN = os.path.join(PROJ, '数据', '外部数据', 'China_Forest_Dataset_v1.0',
                    '3_Forest_Change_Gain', 'China-ForestChange_Gain_StartYear_1km_v1.0.tif')
CKPT = os.path.join(FOREST6, '_f6_ckpt')
OUT_PQ = os.path.join(FOREST6, 'mixed_expanded.parquet')
OUT_STATS = os.path.join(FOREST6, 'f6_expand_stats.json')

# FCS10 码表
NEEDLE = (71, 72, 81, 82)
BROAD = (51, 52, 61, 62)
NEEDLE_EG = (71, 72)          # 常绿针叶
NEEDLE_DC = (81, 82)          # 落叶针叶
MIXCODE = (91, 92)
FOREST = NEEDLE + BROAD + MIXCODE

# 预注册阈值（doc84 §4.2）
MIN_PX = 3                    # n_needle / n_broad 各自 >=3
FRAC_LO, FRAC_HI = 0.15, 0.85
EG_DOM, DC_DOM = 0.60, 0.40   # 45: >0.60  /  46: <0.40

# 撒点约束
LATTICE = 10                  # 100 m 格 = 10 m × 10 像元
PER_TILE_CAP = 300            # 每瓦每类上限
NATIONAL_CAP = 35000          # 全国每类上限
GRID_001 = 0.01               # 0.01° 内同类唯一
EXCL_0005 = 0.005             # 与现有池的重合判定

# 生态区（与 f1_pool.py 逐字一致）
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
ECO_ORDER = ['东北', '华北', '西北', '青藏', '西南', '华中', '华东', '华南']

BAND_ROWS = 4096
T0 = time.time()

FUN_KEYS = ('valid_px', 'forest_px', 'l2_both_ge3_px', 'l3_frac_px',
            'l4_lattice_px', 'l5_center_forest_px', 'l5_cls45', 'l5_cls46',
            'l5_uncertain', 'l6_kept_45', 'l6_kept_46')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def lut(codes):
    a = np.zeros(256, np.uint8)
    for c in codes:
        a[c] = 1
    return a


L_N, L_B, L_E, L_D, L_M = (lut(NEEDLE), lut(BROAD), lut(NEEDLE_EG),
                          lut(NEEDLE_DC), lut(MIXCODE))
L_F = lut(FOREST)


# ------------------------------------------------------------------ 工具
def load_tiles():
    """返回 [(tile, x0, y0, x1, y1)]，按 tiles.csv 顺序，只保留有 FCS10 文件的。"""
    d = pd.read_csv(PLAN_CSV, encoding='utf-8-sig')
    out = []
    for r in d.itertuples():
        if os.path.exists(os.path.join(TILEDIR, r.tile + '_fcs10.tif')):
            out.append((r.tile, float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    return out


def read_vat():
    """读 30m 植被图 VAT（dBASE III），返回 {Value: (类型组, 群系)}。"""
    b = open(VAT, 'rb').read()
    nrec, hlen, rlen = struct.unpack('<IHH', b[4:12])
    off, fields = 32, []
    while b[off] != 0x0d:
        fd = b[off:off + 32]
        off += 32
        fields.append((fd[:11].split(b'\x00')[0].decode('utf-8', 'ignore'), fd[16]))
    out = {}
    for r in range(nrec):
        rec = b[hlen + r * rlen:hlen + (r + 1) * rlen]
        pos, d = 1, {}
        for nm, ln in fields:
            d[nm] = rec[pos:pos + ln].decode('utf-8', 'ignore').strip()
            pos += ln
        try:
            out[int(d['Value'])] = (d.get('类型组', ''), d.get('群系', ''))
        except Exception:
            pass
    return out


def canon(g):
    """类型组 -> 四组。首 token 必须精确等于 1/2/3（见 f2b_fix_v30.py）。"""
    tok = g.split()[0] if g and g.split() else ''
    if tok == '1':
        return '针叶林'
    if tok == '2':
        return '针阔混交林'
    if tok == '3':
        return '阔叶林'
    return '其他'


def veg30_sample(lon, lat):
    """30 m 植被类型点采样（栅格 EPSG:3857，先投影到球面墨卡托）。"""
    R = 6378137.0
    x = np.asarray(lon, float) * math.pi / 180.0 * R
    y = np.log(np.tan((90.0 + np.asarray(lat, float)) * math.pi / 360.0)) * R
    vals = np.empty(len(x), np.int64)
    with rasterio.open(VEG30) as s:
        chunk = 40000
        for i in range(0, len(x), chunk):
            j = min(i + chunk, len(x))
            vals[i:j] = np.fromiter((p[0] for p in s.sample(zip(x[i:j], y[i:j]))),
                                    dtype=np.int64, count=j - i)
    return vals


def gain_sample(lon, lat):
    """2014 年后取得林（StartYear>2014）标记；1 km Albers 栅格，须显式重投影。"""
    from pyproj import Transformer
    out = np.zeros(len(lon), np.uint8)
    with rasterio.open(GAIN) as s:
        tr = Transformer.from_crs('EPSG:4326', s.crs, always_xy=True)
        x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
        for i, (xx, yy) in enumerate(zip(x, y)):
            try:
                v = next(s.sample([(float(xx), float(yy))]))[0]
            except Exception:
                v = 0
            if v and int(v) > 2014:
                out[i] = 1
    return out


# ------------------------------------------------------------------ 阶段1：逐瓦扫描
def scan_tile(tile, band_rows=BAND_ROWS):
    """逐瓦行带扫描；返回 (候选 dict, 漏斗计数 dict)。

    候选只保留 100 m 格（10×10 像元）格心、且中心像元为森林码的位置。
    瓦内每类超过 PER_TILE_CAP 时按确定性随机抽样截断（格点已保证 100 m 间距）。
    """
    path = os.path.join(TILEDIR, tile + '_fcs10.tif')
    fun = {k: 0 for k in FUN_KEYS}
    acc = defaultdict(list)
    import zlib
    rng = np.random.RandomState(zlib.crc32(tile.encode()) % (2 ** 31))  # 确定性，跨进程稳定

    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        tf = ds.transform
        cc = np.arange(LATTICE // 2, W, LATTICE)      # 格心列（全局对齐）
        for r0 in range(0, H, band_rows):
            r1 = min(H, r0 + band_rows)
            pr0 = max(0, r0 - 2)
            pr1 = min(H, r1 + 2)
            buf = ds.read(1, window=Window(0, pr0, W, pr1 - pr0))
            hh = pr1 - pr0
            top = r0 - pr0
            nr = r1 - r0

            nL = cv2.boxFilter(L_N[buf], cv2.CV_8U, (5, 5), normalize=False,
                              borderType=cv2.BORDER_REPLICATE)
            nB = cv2.boxFilter(L_B[buf], cv2.CV_8U, (5, 5), normalize=False,
                              borderType=cv2.BORDER_REPLICATE)
            bL = nL[top:top + nr]
            bB = nB[top:top + nr]
            body = buf[top:top + nr]

            fun['valid_px'] += int(np.count_nonzero(body))
            fun['forest_px'] += int(L_F[body].sum())

            # ---- 漏斗 L2/L3：分块整数比较，避免 91M 元素的临时大数组 ----
            for s in range(0, nr, 512):
                e = min(nr, s + 512)
                a = bL[s:e].astype(np.int16)
                b = bB[s:e].astype(np.int16)
                m2 = (a >= MIN_PX) & (b >= MIN_PX)
                n2 = int(np.count_nonzero(m2))
                if n2:
                    fun['l2_both_ge3_px'] += n2
                    # frac_needle∈[0.15,0.85]  <=>  3a>=0.15*... 整数化：17a>=3b 且 3a<=17b
                    m3 = m2 & (17 * a >= 3 * b) & (3 * a <= 17 * b)
                    fun['l3_frac_px'] += int(np.count_nonzero(m3))
                    del m3
                del a, b, m2

            # ---- 100 m 格心抽样 ----
            rl = np.arange((LATTICE // 2 - r0) % LATTICE, nr, LATTICE)
            if len(rl) and len(cc):
                sel = bL[np.ix_(rl, cc)]
                sel2 = bB[np.ix_(rl, cc)]
                ok = (sel >= MIN_PX) & (sel2 >= MIN_PX) & \
                     (17 * sel.astype(np.int16) >= 3 * sel2.astype(np.int16)) & \
                     (3 * sel.astype(np.int16) <= 17 * sel2.astype(np.int16))
                n_ok = int(np.count_nonzero(ok))
                fun['l4_lattice_px'] += n_ok
                if n_ok:
                    ri, ci = np.nonzero(ok)
                    rr = rl[ri]
                    ccs = cc[ci]
                    ctr = body[rr, ccs]
                    keep = L_F[ctr].astype(bool)          # 中心像元必须是森林码
                    fun['l5_center_forest_px'] += int(np.count_nonzero(keep))
                    if keep.any():
                        rr = rr[keep]
                        ccs = ccs[keep]
                        ctr = ctr[keep]
                        rs = np.clip((rr + top - 2)[:, None] + np.arange(5)[None, :], 0, hh - 1)
                        cs = np.clip((ccs - 2)[:, None] + np.arange(5)[None, :], 0, W - 1)
                        patch = buf[rs[:, :, None], cs[:, None, :]]
                        nE = L_E[patch].sum((1, 2)).astype(np.int16)
                        nD = L_D[patch].sum((1, 2)).astype(np.int16)
                        nM = L_M[patch].sum((1, 2)).astype(np.int16)
                        del patch, rs, cs
                        nLv = sel[ri[keep], ci[keep]].astype(np.int16)
                        nBv = sel2[ri[keep], ci[keep]].astype(np.int16)
                        frac = (nLv / (nLv + nBv).astype(np.float32)).astype(np.float32)
                        nd = nE + nD
                        evf = np.where(nd > 0, nE / np.maximum(nd, 1), np.nan).astype(np.float32)
                        cls = np.where(evf > EG_DOM, 45,
                                       np.where(evf < DC_DOM, 46, 0)).astype(np.int8)
                        fun['l5_cls45'] += int((cls == 45).sum())
                        fun['l5_cls46'] += int((cls == 46).sum())
                        fun['l5_uncertain'] += int((cls == 0).sum())
                        acc['row'].append((rr + r0).astype(np.int32))
                        acc['col'].append(ccs.astype(np.int32))
                        acc['cls'].append(cls)
                        acc['frac'].append(frac)
                        acc['nL'].append(nLv)
                        acc['nB'].append(nBv)
                        acc['nE'].append(nE)
                        acc['nD'].append(nD)
                        acc['nM'].append(nM)
                        acc['evf'].append(evf)
                        acc['ctr'].append(ctr.astype(np.uint8))
                        del nE, nD, nM, nLv, nBv, frac, evf, nd, cls
                del sel, sel2, ok
            del buf, nL, nB, bL, bB, body

    # ---- 瓦内每类 300 上限（确定性随机抽样）----
    if acc['cls']:
        cls = np.concatenate(acc['cls'])
        keep_idx = []
        for c, cap in ((45, PER_TILE_CAP), (46, PER_TILE_CAP)):
            ii = np.flatnonzero(cls == c)
            if len(ii) > cap:
                ii = np.sort(rng.choice(ii, cap, replace=False))
            keep_idx.append(ii)
            fun['l6_kept_%d' % c] = int(len(ii))
        keep_idx = np.sort(np.concatenate(keep_idx))    # uncertain(0) 不入库
        cat = {k: np.concatenate(v)[keep_idx] for k, v in acc.items()}
    else:
        cat = {k: np.zeros(0, dtype=np.int32) for k in
               ('row', 'col', 'cls', 'frac', 'nL', 'nB', 'nE', 'nD', 'nM', 'evf', 'ctr')}
    fun['width'] = W
    fun['height'] = H
    fun['transform'] = [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]
    return cat, fun


def ckpt_path(tile):
    return os.path.join(CKPT, tile + '.npz')


def save_ckpt(tile, cand, fun):
    os.makedirs(CKPT, exist_ok=True)
    np.savez_compressed(ckpt_path(tile), fun=json.dumps(fun, ensure_ascii=False),
                        **cand)


def load_ckpt(tile):
    p = ckpt_path(tile)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    fun = json.loads(str(z['fun']))
    cand = {k: z[k] for k in z.files if k != 'fun'}
    return cand, fun


def frac_needle_500m(df):
    """对最终点算 500 m（101×101 @10m）窗口针叶占比 —— 纯林回吸诊断。

    只需处理最终入选点（≤70k），逐点 boundless 读约 3 ms/点。
    接近 0/1 → 500 m 尺度已是纯林，50 m 内的针阔同现只是纯林边缘锯齿；
    接近 0.5 → 真异质混交景观。
    """
    out = np.full(len(df), np.nan, np.float32)
    for tile in df.tile.unique():
        path = os.path.join(TILEDIR, tile + '_fcs10.tif')
        if not os.path.exists(path):
            continue
        ii = np.flatnonzero((df.tile == tile).values)
        with rasterio.open(path) as ds:
            inv = ~ds.transform
            xx, yy = inv * (df.lon.values[ii], df.lat.values[ii])
            r = np.floor(yy).astype(np.int64)
            c = np.floor(xx).astype(np.int64)
            for k, (ri, ci) in enumerate(zip(r, c)):
                try:
                    a = ds.read(1, window=Window(int(ci) - 50, int(ri) - 50, 101, 101),
                                boundless=True, fill_value=0)
                except Exception:
                    continue
                nl = int(L_N[a].sum())
                nb = int(L_B[a].sum())
                out[ii[k]] = nl / max(nl + nb, 1)
    return out


# ------------------------------------------------------------------ 阶段2：合并 + 配额 + 属性
def assemble(tiles):
    emit('=== 阶段2：合并 checkpoint ===')
    frames = []
    funnel = Counter()
    tile_rows = {}
    missing = []
    for tile, x0, y0, x1, y1 in tiles:
        got = load_ckpt(tile)
        if got is None:
            missing.append(tile)
            continue
        cand, fun = got
        for k in FUN_KEYS:
            funnel[k] += int(fun.get(k, 0))
        n = len(cand['row'])
        tile_rows[tile] = n
        if n == 0:
            continue
        a, b, c_, d_, e_, f_ = fun['transform']
        lon = a * (cand['col'] + 0.5) + b * (cand['row'] + 0.5) + c_
        lat = d_ * (cand['col'] + 0.5) + e_ * (cand['row'] + 0.5) + f_
        frames.append(pd.DataFrame({
            'lon': lon.astype(np.float64),
            'lat': lat.astype(np.float64),
            'tile': tile,
            'cls': cand['cls'],
            'frac_needle': cand['frac'],
            'n_needle_win': cand['nL'],
            'n_broad_win': cand['nB'],
            'n_eg_win': cand['nE'],
            'n_dc_win': cand['nD'],
            'n_mixedcode_win': cand['nM'],
            'needle_evergreen_frac': cand['evf'],
            'center_code': cand['ctr'],
        }))
    if missing:
        emit('  !! 缺 checkpoint 的瓦片 %d 个：%s' % (len(missing), missing[:12]))
    if not frames:
        raise SystemExit('没有任何 checkpoint，先跑 --scan')
    df = pd.concat(frames, ignore_index=True)
    emit('合并瓦内上限后候选 %d 点（%d 瓦有候选 / %d 瓦已扫）'
         % (len(df), sum(1 for v in tile_rows.values() if v > 0), len(tile_rows)))
    del frames
    gc.collect()
    n_after_tile_cap = len(df)

    # ---- 0.01° 同类唯一 ----
    emit('0.01° 同类唯一')
    key = (np.floor(df.lon.values / GRID_001).astype(np.int64) * 1000003
           + np.floor(df.lat.values / GRID_001).astype(np.int64) * 7
           + df.cls.values.astype(np.int64))
    order = np.lexsort((-df.n_mixedcode_win.values,
                        -(df.n_eg_win.values.astype(np.int32) + df.n_dc_win.values),
                        key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    df = df.iloc[np.sort(order[first])].reset_index(drop=True)
    emit('  → %d 点（去重 %d）' % (len(df), int(len(order) - first.sum())))
    n_after_001 = len(df)

    # ---- 与现有池 0.005° 重合剔除 ----
    emit('与现有池 0.005° 重合剔除')
    pool = pd.read_parquet(POOL, columns=['lon', 'lat'])
    pg = (np.floor(pool.lon.values / EXCL_0005).astype(np.int64) * 1000003
          + np.floor(pool.lat.values / EXCL_0005).astype(np.int64) * 7)
    qg = (np.floor(df.lon.values / EXCL_0005).astype(np.int64) * 1000003
          + np.floor(df.lat.values / EXCL_0005).astype(np.int64) * 7)
    dup = np.isin(qg, pg)
    n_dup = int(dup.sum())
    emit('  与现有池同 0.005° 格 %d 点（%.2f%%）→ 剔除'
         % (n_dup, dup.mean() * 100 if len(df) else 0.0))
    df = df.loc[~dup].reset_index(drop=True)
    del pg, qg, dup
    gc.collect()
    n_after_pool = len(df)

    # ---- 境外剔除（DataV 逐 feature，不做 unary_union）----
    emit('境外剔除（DataV 逐 feature）')
    import geopandas as gpd
    from shapely.geometry import Point
    gj = json.load(open(DATAV, encoding='utf-8'))
    feats = [f for f in gj['features']
             if f['properties'].get('name') and f['properties'].get('level') == 'province']
    gdf = gpd.GeoDataFrame.from_features(feats, crs='EPSG:4326')[['name', 'geometry']]
    inside = np.zeros(len(df), bool)
    prov = np.array([None] * len(df), dtype=object)
    pts = gpd.GeoDataFrame(df[['lon', 'lat']].copy(),
                           geometry=gpd.points_from_xy(df.lon.values, df.lat.values),
                           crs='EPSG:4326')
    for _, row in gdf.iterrows():
        if inside.all():
            break
        sub = pts.loc[~inside]
        try:
            sj = gpd.sjoin(sub[['geometry']],
                           gpd.GeoDataFrame([row], crs='EPSG:4326'),
                           how='inner', predicate='intersects')
        except Exception as ex:
            emit('  feature %s sjoin 失败：%s' % (row['name'], ex))
            continue
        ii = np.unique(sj.index.values)
        inside[ii] = True
        prov[ii] = row['name']
    n_out = int((~inside).sum())
    emit('  境外/未落省 %d 点（%.2f%%）→ 剔除'
         % (n_out, n_out / max(len(df), 1) * 100))
    df = df.loc[inside].reset_index(drop=True)
    df['prov'] = prov[inside]
    n_after_cn = len(df)
    del pts, gdf
    gc.collect()

    # ---- 全国每类 35000 上限（按省轮转抽样，保持空间均匀）----
    emit('全国每类 %d 上限' % NATIONAL_CAP)
    keep_parts = []
    for cls in (45, 46):
        sub = df[df.cls == cls]
        if len(sub) > NATIONAL_CAP:
            sub = sub.assign(_r=np.random.RandomState(45 if cls == 45 else 46).rand(len(sub)))
            sub = (sub.sort_values(['prov', '_r']).groupby('prov', sort=False).head(10 ** 9)
                   .sort_values('_r').iloc[:NATIONAL_CAP].drop(columns=['_r']))
        keep_parts.append(sub)
    df = pd.concat(keep_parts, ignore_index=True)
    emit('  → %d 点（45=%d, 46=%d）'
         % (len(df), int((df.cls == 45).sum()), int((df.cls == 46).sum())))

    # ---- 属性提取 ----
    emit('提取 v30 类型组/群系、recent_gain')
    vat = read_vat()
    vals = veg30_sample(df.lon.values, df.lat.values)
    gmap = {k: canon(v[0]) for k, v in vat.items()}
    df['v30_value'] = vals.astype(np.int32)
    df['v30_group'] = np.array(
        ['NA' if int(v) == 65535 else gmap.get(int(v), '其他') for v in vals], dtype=object)
    df['v30_群系'] = np.array(
        [vat.get(int(v), ('', ''))[1] if int(v) != 65535 else '' for v in vals], dtype=object)
    df['recent_gain'] = gain_sample(df.lon.values, df.lat.values)
    emit('500 m 窗口针叶占比（纯林回吸诊断）')
    df['frac_needle_500m'] = frac_needle_500m(df)
    df['cls6_expand'] = df['cls'].astype(np.int8)
    df['uid'] = -np.arange(1, len(df) + 1, dtype=np.int64)   # 负 uid，与现有池区分


    # ---- 统计 ----
    emit('统计')
    stats = build_stats(df, tiles, funnel, tile_rows, {
        'n_after_tile_cap': n_after_tile_cap,
        'n_after_001deg': n_after_001,
        'n_pool_overlap_removed': n_dup,
        'n_after_pool_excl': n_after_pool,
        'n_outside_china_removed': n_out,
        'n_after_china': n_after_cn,
        'n_missing_ckpt': len(missing),
        'missing_ckpt': missing,
    })
    cols = ['uid', 'lon', 'lat', 'tile', 'cls6_expand', 'frac_needle',
            'n_needle_win', 'n_broad_win', 'n_mixedcode_win',
            'needle_evergreen_frac', 'center_code', 'v30_group', 'v30_群系',
            'prov', 'recent_gain', 'n_eg_win', 'n_dc_win', 'v30_value',
            'frac_needle_500m']
    df[cols].to_parquet(OUT_PQ, index=False)
    with open(OUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    emit('写出 %s（%d 行）' % (OUT_PQ, len(df)))
    emit('写出 %s' % OUT_STATS)
    return stats


def build_stats(df, tiles, funnel, tile_rows, stage):
    """四级漏斗 + 分类计数 + 覆盖 + 分布 + 与现有池距离。"""
    from scipy.spatial import cKDTree

    n45 = int((df.cls6_expand == 45).sum())
    n46 = int((df.cls6_expand == 46).sum())
    eco_of = df.prov.map(PROV2ECO)
    s = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': os.path.abspath(__file__),
        'rule_source': '技术文档/84_全国六类林地样本汇总.md §4.2（2026-10-25 修正版：邻域针叶码定义 45/46）',
        'rule': {
            'window': '5x5 @10m (50m x 50m)',
            'needle_codes': list(NEEDLE), 'broad_codes': list(BROAD),
            'needle_eg_codes': list(NEEDLE_EG), 'needle_dc_codes': list(NEEDLE_DC),
            'mixcode_codes': list(MIXCODE),
            'min_px_each': MIN_PX, 'frac_needle': [FRAC_LO, FRAC_HI],
            'cls45': 'needle_evergreen_frac > %.2f' % EG_DOM,
            'cls46': 'needle_evergreen_frac < %.2f' % DC_DOM,
            'note_91_92': '91/92 不计入 needle/broad 分母，仅单独统计 n_mixedcode_win',
            'lattice': '100 m 格（10x10 像元）每格最多 1 点，取格心',
            'grid_001_uniques': '0.01° 内同类唯一',
            'pool_overlap_excl': '0.005° 内与现有 472,307 点重合则剔除',
            'per_tile_cap_per_class': PER_TILE_CAP,
            'national_cap_per_class': NATIONAL_CAP,
            'recent_gain': 'StartYear>2014 标记 recent_gain=1，不剔除',
        },
        'funnel': {
            'L0_fcs10_valid_px': int(funnel['valid_px']),
            'L1_forest_code_px': int(funnel['forest_px']),
            'L2_window_needle_and_broad_ge3_px': int(funnel['l2_both_ge3_px']),
            'L3_frac_needle_015_085_px': int(funnel['l3_frac_px']),
            'L4_100m_lattice_heart_px': int(funnel['l4_lattice_px']),
            'L5_center_is_forest_px': int(funnel['l5_center_forest_px']),
        },
        'scatter_stage': {
            'note': 'L5 之后进入撒点：瓦内每类≤300 → 0.01°同类唯一 → 0.005° 与现有池不重合 → 境内 → 全国每类≤35000',
            'after_tile_cap': int(stage['n_after_tile_cap']),
            'after_001deg_unique': int(stage['n_after_001deg']),
            'removed_pool_0005_overlap': int(stage['n_pool_overlap_removed']),
            'after_pool_exclusion': int(stage['n_after_pool_excl']),
            'removed_outside_china': int(stage['n_outside_china_removed']),
            'after_china_boundary': int(stage['n_after_china']),
            'final': int(len(df)),
        },
        'counts': {
            'total': int(len(df)),
            'cls45': n45, 'cls46': n46,
            'uncertain_px_upper_bound_at_L5': int(funnel['l5_uncertain']),
            'uncertain_note': '窗口内针叶常绿/落叶各半（needle_evergreen_frac 0.40–0.60）→ 不入库；'
                              '此数为 L5 处像素级计数（未去重）',
        },
        'coverage': {
            'tiles_total_in_plan': int(len(tiles)),
            'tiles_scanned': int(len(tile_rows)),
            'tiles_with_candidates': int(sum(1 for v in tile_rows.values() if v > 0)),
            'tiles_missing_ckpt': int(stage['n_missing_ckpt']),
            'tiles_missing_ckpt_list': stage['missing_ckpt'][:20],
            'provinces': int(df.prov.nunique()),
            'provinces_list': sorted([str(x) for x in df.prov.dropna().unique()]),
        },
        'by_eco': {e: {'45': int(((df.cls6_expand == 45) & (eco_of == e)).sum()),
                       '46': int(((df.cls6_expand == 46) & (eco_of == e)).sum()),
                       'total': int((eco_of == e).sum())}
                   for e in ECO_ORDER},
        'by_prov': {str(p): {'45': int(((df.cls6_expand == 45) & (df.prov == p)).sum()),
                             '46': int(((df.cls6_expand == 46) & (df.prov == p)).sum()),
                             'total': int((df.prov == p).sum())}
                    for p in sorted(df.prov.dropna().unique())},
        'v30_group_crosstab': {str(g): {'45': int(((df.cls6_expand == 45) & (df.v30_group == g)).sum()),
                                        '46': int(((df.cls6_expand == 46) & (df.v30_group == g)).sum()),
                                        'total': int((df.v30_group == g).sum())}
                               for g in df.v30_group.value_counts().index},
        'v30_top_biome': {str(g): [str(x) for x in df.loc[df.v30_group == g, 'v30_群系']
                                   .value_counts().head(8).index]
                          for g in df.v30_group.value_counts().index},
        'center_code_dist': {str(k): int(v) for k, v in df.center_code.value_counts().items()},
        'recent_gain': {'n': int(df.recent_gain.sum()),
                        'pct': round(float(df.recent_gain.mean() * 100), 3),
                        'by_cls': {str(c): int(df.loc[df.cls6_expand == c, 'recent_gain'].sum())
                                   for c in (45, 46)}},
        'frac_needle_hist': {},
        'needle_evergreen_frac_hist': {},
        'tile_yield': {'mean_per_tile_45': round(n45 / max(1, sum(1 for v in tile_rows.values() if v > 0)), 2),
                       'mean_per_tile_46': round(n46 / max(1, sum(1 for v in tile_rows.values() if v > 0)), 2)},
    }
    for c in (45, 46):
        sub = df[df.cls6_expand == c]
        def hist(v, edges):
            cnt = pd.cut(v, edges).value_counts().sort_index()
            return {'%.2f' % iv.left: int(n) for iv, n in cnt.items()}
        s['frac_needle_hist'][str(c)] = hist(sub.frac_needle, np.arange(0, 1.01, 0.1))
        s['needle_evergreen_frac_hist'][str(c)] = hist(sub.needle_evergreen_frac,
                                                       np.arange(0, 1.01, 0.1))
        s.setdefault('per_class_detail', {})[str(c)] = {
            'n': int(len(sub)),
            'frac_needle_median': round(float(sub.frac_needle.median()), 4),
            'needle_evergreen_frac_median': round(float(sub.needle_evergreen_frac.median()), 4),
            'n_needle_win_median': float(sub.n_needle_win.median()),
            'n_broad_win_median': float(sub.n_broad_win.median()),
            'n_mixedcode_win_median': float(sub.n_mixedcode_win.median()),
            'n_mixedcode_win_gt0_pct': round(float((sub.n_mixedcode_win > 0).mean() * 100), 2),
            'v30_mixed_group_pct': round(float((sub.v30_group == '针阔混交林').mean() * 100), 2),
            'n_prov': int(sub.prov.nunique()),
            'n_tile': int(sub.tile.nunique()),
        }

    # ---- 纯林回吸诊断：500 m 窗口针叶占比 ----
    s['pure_forest_diagnostic'] = {
        'question': '扩出的点是否又落在纯林区（50m 内混交、但 500m 尺度是纯林）',
        'metric': 'frac_needle_500m = 500m×500m（101×101 @10m）窗口内 needle/(needle+broad)',
        'interpretation': '接近 0 或 1 → 500 m 尺度已是纯林，该点只是纯林边缘的锯齿噪声，'
                          '不是真混交；接近 0.5 → 真异质混交景观',
    }
    for c in (45, 46):
        sub = df[df.cls6_expand == c]
        if not len(sub):
            continue
        f5 = sub.frac_needle_500m.astype(float).values
        s['pure_forest_diagnostic'][str(c)] = {
            'n': int(len(sub)),
            'median': round(float(np.median(f5)), 4),
            'p05': round(float(np.percentile(f5, 5)), 4),
            'p25': round(float(np.percentile(f5, 25)), 4),
            'p75': round(float(np.percentile(f5, 75)), 4),
            'p95': round(float(np.percentile(f5, 95)), 4),
            'frac_lt_0.10_pct': round(float((f5 < 0.10).mean() * 100), 2),
            'frac_gt_0.90_pct': round(float((f5 > 0.90).mean() * 100), 2),
            'frac_in_0.25_0.75_pct': round(float(((f5 >= 0.25) & (f5 <= 0.75)).mean() * 100), 2),
            'pure_core_pct(<0.10 or >0.90)': round(
                float(((f5 < 0.10) | (f5 > 0.90)).mean() * 100), 2),
        }
    s['pure_forest_diagnostic']['overall'] = {
        'n': int(len(df)),
        'pure_core_pct(<0.10 or >0.90)': round(
            float(((df.frac_needle_500m < 0.10) | (df.frac_needle_500m > 0.90)).mean() * 100), 2),
        'in_0.25_0.75_pct': round(
            float(((df.frac_needle_500m >= 0.25) & (df.frac_needle_500m <= 0.75)).mean() * 100), 2),
    }

    # ---- 与现有池的距离 ----
    emit('  计算与现有 472,307 点的最近距离')
    pool = pd.read_parquet(POOL, columns=['lon', 'lat'])
    tree = cKDTree(np.column_stack([pool.lon.values, pool.lat.values]),
                   compact_nodes=False, balanced_tree=False)
    dd, _ = tree.query(np.column_stack([df.lon.values, df.lat.values]), k=1, p=np.inf)
    km = dd * 111.0
    pg = set((np.floor(pool.lon.values / GRID_001).astype(np.int64) * 1000003
              + np.floor(pool.lat.values / GRID_001).astype(np.int64) * 7).tolist())
    qg = (np.floor(df.lon.values / GRID_001).astype(np.int64) * 1000003
          + np.floor(df.lat.values / GRID_001).astype(np.int64) * 7)
    s['distance_to_existing_pool'] = {
        'metric': 'Chebyshev（Δlon/Δlat 取 max）×111 km —— 保守上界，真实欧氏距离更小',
        'n_pool': int(len(pool)),
        'km_median': round(float(np.median(km)), 3),
        'km_p25': round(float(np.percentile(km, 25)), 3),
        'km_p75': round(float(np.percentile(km, 75)), 3),
        'km_p95': round(float(np.percentile(km, 95)), 3),
        'km_max': round(float(km.max()), 3),
        'lt_10km_pct': round(float((km < 10).mean() * 100), 2),
        'lt_1km_pct': round(float((km < 1).mean() * 100), 2),
        'lt_0.5km_pct': round(float((km < 0.5).mean() * 100), 2),
        'km_median_by_cls': {str(c): round(float(np.median(km[df.cls6_expand.values == c])), 3)
                             for c in (45, 46) if (df.cls6_expand.values == c).any()},
        'overlap_001deg_pct': round(float(np.mean([x in pg for x in qg]) * 100), 3),
        'overlap_001deg_n': int(sum(1 for x in qg if x in pg)),
    }
    return s


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--tiles', nargs='*', default=None)
    ap.add_argument('--band-rows', type=int, default=BAND_ROWS)
    ap.add_argument('--force', action='store_true', help='忽略已有 checkpoint')
    args = ap.parse_args()
    if not args.scan and not args.assemble:
        ap.error('至少给 --scan 或 --assemble')

    tiles = load_tiles()
    emit('瓦片 %d 个（%s）' % (len(tiles), TILEDIR))
    if args.tiles:
        want = set(args.tiles)
        tiles = [t for t in tiles if t[0] in want]
        emit('  限定 %d 个：%s' % (len(tiles), [t[0] for t in tiles]))

    if args.scan:
        os.makedirs(CKPT, exist_ok=True)
        done = n_skip = n_fail = 0
        for tile, x0, y0, x1, y1 in tiles:
            p = ckpt_path(tile)
            if os.path.exists(p) and not args.force and os.path.getsize(p) > 0:
                n_skip += 1
                continue
            t = time.time()
            try:
                cand, fun = scan_tile(tile, args.band_rows)
                save_ckpt(tile, cand, fun)
                emit('  %s %5.1fs  L5=%d  45=%d 46=%d unc=%d (累计 %d/%d, 跳过 %d, %.0fs)'
                     % (tile, time.time() - t, fun['l5_center_forest_px'],
                        fun['l6_kept_45'], fun['l6_kept_46'], fun['l5_uncertain'],
                        done + n_skip + n_fail + 1, len(tiles), n_skip, time.time() - T0))
                del cand, fun
            except Exception as ex:
                n_fail += 1
                emit('  !! %s 失败：%r' % (tile, ex))
                import traceback
                traceback.print_exc()
            done += 1
            gc.collect()
        emit('扫描完成：处理 %d，跳过 %d，失败 %d，总 %.1f 分钟'
             % (done, n_skip, n_fail, (time.time() - T0) / 60))

    if args.assemble:
        assemble(tiles)


if __name__ == '__main__':
    main()
