# -*- coding: utf-8 -*-
"""f1_pool.py — 六类林地样本 F1 阶段：四类纯林 GPKG + 母库林地码 → 统一去重林地池（带齐 AEF64/物候/地形/气候）

* 输入（只读，绝不修改）：
    A. 四类纯林 GPKG（含 AEF64，本轮主力）
       F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/
           林地_常绿针叶林/..._2017-2024并集_WGS84.gpkg   134,717
           林地_落叶针叶林/..._2017-2024并集_WGS84.gpkg    59,785
           林地_常绿阔叶林/..._2017-2024并集_WGS84.gpkg   112,083
           林地_落叶阔叶林/..._2017-2024并集_WGS84.gpkg   124,304
       layer=strict 推荐层 / clean 可用层；cls4 为类内码（4=EBF 5=DBF 6=ENF 7=DNF）
    B. 母库逐年子集（混交 91/92 的唯一来源）
       F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet（2,118,721 点，取 10 个林地码 466,315）
    C. 特征源（只 join，不并入主池）
       F:/lc_work/nb_national_2026/labels/methods/work_base.parquet              物候/气候/cz/地形/省
       F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet 地形/物候/省/生态区（兜底）
       E:/DEM/NASA_DEM/NASADEM+UD20250222/NASADEM_30m_RD2020_UD20250221+COG.tif 地形（末位兜底，自算 60m）

* 规则源：技术文档/84_全国六类林地样本汇总.md §2（资产盘点）§4.1（六类码表）§4.4（8 生态区）§6（坑）
* 坐标键：lon6=round(lon,6)、lat6=round(lat,6)，键 = f'{lon6:.6f}_{lat6:.6f}'
* 去重优先级：GPKG(layer=strict) > GPKG(layer=clean) > 母库
* 矛盾：同一坐标键在 GPKG 内落入不同类 → conflict_4class=1，cls6=0（保留不删）
* 门槛（QC）：越界 / lon-lat NaN / AEF64 全 NaN → 剔除计数；去重后重复键必须 0；AEF64 逐列值域体检
* 输出（F:/lc_work/nb_national_2026/forest6/）：
    forest_pool_v1.parquet / f1_stats.json / f1_by_eco.csv / f1_by_prov.csv
* 用法：python f1_pool.py [--no-terrain-fill]     单进程，峰值内存 < 3 GB
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------- 路径常量 ----------------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
GPKG_DIR = os.path.join(PROJ, '数据', '单类样本的单独处理_2026.10.12')
MASTER = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
WORK_BASE = r'F:/lc_work/nb_national_2026/labels/methods/work_base.parquet'
V2_HABIT = r'F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet'
DEM = r'E:/DEM/NASA_DEM/NASADEM+UD20250222/NASADEM_30m_RD2020_UD20250221+COG.tif'
DATAV = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
TERRAIN_LIB = r'F:/lc_work/nb_sep_2026/aspect'
OUT_DIR = r'F:/lc_work/nb_national_2026/forest6'

AEF = ['A%02d' % i for i in range(64)]

GPKG_FILES = [('林地_常绿针叶林', 41), ('林地_落叶针叶林', 42),
              ('林地_常绿阔叶林', 43), ('林地_落叶阔叶林', 44)]
MASTER2CLS6 = {71: 41, 72: 41, 81: 42, 82: 42, 51: 43, 52: 43,
               61: 44, 62: 44, 91: 45, 92: 46}
MASTER_FOREST_CODES = sorted(MASTER2CLS6)
CLS6_NAME = {0: '未定(四类冲突)', 41: '常绿针叶林', 42: '落叶针叶林', 43: '常绿阔叶林',
             44: '落叶阔叶林', 45: '常绿针阔混交林', 46: '落叶针阔混交林'}

# 8 生态区（doc84 §4.4；与 nb_national_2026/labels/code/s2_sample_national.py 的 ECO/PROV2ECO 逐字一致）
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

# 地形自算参数（与 s2_sample_national.py 的 DEM_PAD/DEM_AGG 逐值一致：NASADEM 30m → 60m 2×2 均值）
DEM_PAD, DEM_AGG, BAND = 0.02, 2, 2000

T0 = time.time()
def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def add_key(df):
    lon6 = np.round(df['lon'].to_numpy(float), 6)
    lat6 = np.round(df['lat'].to_numpy(float), 6)
    df['lon6'] = lon6
    df['lat6'] = lat6
    df['key'] = np.char.add(np.char.add(np.char.mod('%.6f', lon6), '_'),
                            np.char.mod('%.6f', lat6))
    return df


# ---------------- 源 A：四类纯林 GPKG（逐个读，读完即释放） ----------------
def load_gpkg():
    import pyogrio
    cols = ['lon', 'lat', 'prov', 'cls4', 'layer', 'tier', 'sug', 'src',
            'ct_prov', 'ct_code', 'ct_km'] + AEF
    parts, raw = [], {}
    for d, cls6 in GPKG_FILES:
        p = os.path.join(GPKG_DIR, d, d + '_2017-2024并集_WGS84.gpkg')
        g = pyogrio.read_dataframe(p, columns=cols, read_geometry=False)
        raw[d] = {'rows': int(len(g)), 'cls6': cls6,
                  'layer': {str(k): int(v) for k, v in g['layer'].value_counts().items()},
                  'cls4_values': sorted(int(x) for x in g['cls4'].unique()),
                  'tier': {str(k): int(v) for k, v in g['tier'].value_counts().items()},
                  'src': {str(k): int(v) for k, v in g['src'].value_counts().items()},
                  'aef_all_nan': int(g[AEF].isna().all(axis=1).sum())}
        g['cls6_gpkg'] = np.int16(cls6)
        g['cls4_gpkg'] = g['cls4'].astype(np.int16)
        g = g.drop(columns=['cls4'])
        for c in AEF:
            g[c] = g[c].astype(np.float32)
        g = add_key(g)
        parts.append(g)
        emit('GPKG %s → %d 行（strict %d / clean %d，cls4=%s）' % (
            d, len(g), raw[d]['layer'].get('strict', 0), raw[d]['layer'].get('clean', 0),
            raw[d]['cls4_values']))
        del g
    G = pd.concat(parts, ignore_index=True)
    del parts
    return G, raw


# ---------------- 源 B：母库林地码 ----------------
def load_master():
    cols = ['row_id', 'lon', 'lat', 'class_new', 'tier', 'src', 'src_conf',
            'train_weight', 'qc_status'] + AEF
    m = pq.read_table(MASTER, columns=cols,
                      filters=[('class_new', 'in', MASTER_FOREST_CODES)]).to_pandas()
    for c in AEF:
        m[c] = m[c].astype(np.float32)
    m = m.rename(columns={'class_new': 'code_master'})
    m['cls6_master'] = m['code_master'].map(MASTER2CLS6).astype(np.int16)
    raw = {'rows': int(len(m)),
           'by_code': {str(k): int(v) for k, v in m['code_master'].value_counts().sort_index().items()},
           'tier': {str(k): int(v) for k, v in m['tier'].value_counts().items()},
           'src': {str(k): int(v) for k, v in m['src'].value_counts().items()},
           'aef_all_nan': int(m[AEF].isna().all(axis=1).sum())}
    m = add_key(m)
    emit('母库林地码 → %d 行（91 混交 %d / 92 混交 %d）' % (
        len(m), raw['by_code'].get('91', 0), raw['by_code'].get('92', 0)))
    return m, raw


# ---------------- 合并 + 去重 + 矛盾 ----------------
def merge_pool(G, M):
    n_g, n_m = len(G), len(M)
    G = G.rename(columns={'src': 'src_orig'})
    G['src_rank'] = np.where(G['layer'] == 'strict', 1, 2).astype(np.int8)
    G['code_master'] = np.int16(-1)
    G['master_row_id'] = np.int64(-1)
    G['cls6_master'] = np.int16(-1)
    M = M.rename(columns={'src': 'src_orig'})
    M['src_rank'] = np.int8(3)
    M['layer'] = 'master'
    M['cls6_gpkg'] = np.int16(-1)
    M['cls4_gpkg'] = np.int16(-1)
    M['prov'] = None
    M['ct_prov'] = None
    M['ct_km'] = np.nan
    M['sug'] = np.nan
    M['master_row_id'] = M['row_id'].astype(np.int64)
    M = M.drop(columns=['row_id'])

    keep = ['key', 'lon', 'lat', 'lon6', 'lat6', 'src_rank', 'layer', 'tier', 'src_orig',
            'prov', 'ct_prov', 'ct_km', 'sug', 'cls4_gpkg', 'cls6_gpkg', 'cls6_master',
            'code_master', 'master_row_id'] + AEF
    allr = pd.concat([G[keep], M[keep]], ignore_index=True)
    del G, M
    emit('合并行数 %d（GPKG %d + 母库 %d）' % (len(allr), n_g, n_m))

    # 源 token：GPKG 用 'G<cls6>.<layer>'，母库用 'M<code>'，共 ≤18 种
    isg = (allr['src_rank'].to_numpy() < 3)
    tok = np.empty(len(allr), dtype=object)
    tok[isg] = ('G' + allr.loc[isg, 'cls6_gpkg'].astype(str).to_numpy()
                + '.' + allr.loc[isg, 'layer'].astype(str).to_numpy())
    tok[~isg] = 'M' + allr.loc[~isg, 'code_master'].astype(str).to_numpy()
    allr['token'] = tok
    toks = sorted(set(tok))
    tmap = {t: i for i, t in enumerate(toks)}
    tb = np.fromiter((tmap[t] for t in tok), dtype=np.int64, count=len(tok))
    codes, ukeys = pd.factorize(allr['key'])
    nk = len(ukeys)
    bit = np.zeros((nk, len(toks)), dtype=bool)
    for i in range(len(toks)):
        bit[codes[tb == i], i] = True
    emit('坐标键 %d 个，源 token %s' % (nk, toks))

    # GPKG 内部四类矛盾（同一键出现在两份 GPKG）→ 记录涉及的类码
    gc = allr.loc[isg, ['key', 'cls6_gpkg']].drop_duplicates()
    ncls = gc.groupby('key', sort=False).size().reindex(ukeys).fillna(0).to_numpy(np.int16)
    cls_by_key = gc.groupby('key', sort=False)['cls6_gpkg'].agg(lambda s: sorted(set(s.tolist())))
    del gc
    conflict = ncls > 1
    emit('GPKG 内四类矛盾点 %d' % int(conflict.sum()))

    # 每键源清单字符串（bit 矩阵 → 拼接，toks 已排序）
    bl = [np.where(bit[:, i], toks[i], '') for i in range(len(toks))]
    src_list = bl[0].astype(object)
    for b in bl[1:]:
        src_list = np.where(b == '', src_list, np.where(src_list == '', b,
                                                         src_list.astype(str) + ';' + b))
    src_n = bit.sum(axis=1).astype(np.int8)

    # 取每键 rank 最小的一行（rank 相同保留首个）
    n_before = len(allr)
    allr = allr.sort_values(['key', 'src_rank'], kind='stable')
    win = allr.drop_duplicates('key', keep='first').copy()
    del allr
    emit('去重后 %d 点（删除 %d 行）' % (len(win), n_before - len(win)))
    assert len(win) == nk

    # 按 key 顺序对齐（factorize 顺序 = 首次出现顺序，与 win 排序不同，需重排）
    order = pd.Index(ukeys).get_indexer(win['key'])
    win['src_list'] = src_list[order]
    win['src_n'] = src_n[order]
    win['n_gpkg_cls'] = ncls[order]
    conf_w = conflict[order]
    win['conflict_4class'] = conf_w.astype(np.int8)
    win['conflict_classes'] = ''
    if conf_w.any():
        ks = win['key'].to_numpy()[conf_w]
        win.loc[conf_w, 'conflict_classes'] = ['|'.join(str(x) for x in cls_by_key[k]) for k in ks]
    win['src_primary'] = win['token']
    win['cls6'] = np.where(win['src_rank'].to_numpy() < 3,
                           win['cls6_gpkg'].to_numpy(), win['cls6_master'].to_numpy()).astype(np.int16)
    win.loc[conf_w, 'cls6'] = 0
    return win.reset_index(drop=True), {'gpkg_rows': n_g, 'master_rows': n_m,
                                        'raw_rows_total': n_g + n_m,
                                        'dedup_removed': n_before - nk,
                                        'conflict_4class': int(conf_w.sum()),
                                        'tokens': toks}


# ---------------- 特征：work_base / v2 habit ----------------
def load_wb():
    c = ['lon', 'lat', 'prov', 'prov_cta', 'amp2019', 'amp2023', 'djf2019', 'djf2023',
         'v2019', 'v2023', 'cz', 'bio01', 'bio06', 'bio11', 'bio12',
         'dem_nasadem', 'slope', 'aspect', 'cgls_ft']
    d = pq.read_table(WORK_BASE, columns=c).to_pandas()
    d = add_key(d)
    for c0 in ['v2019', 'v2023']:
        d[c0] = d[c0].replace('', np.nan)
    d = d.rename(columns={'dem_nasadem': 'elev_wb', 'slope': 'slope_wb', 'aspect': 'aspect_wb',
                          'prov': 'prov_wb'})
    # 去掉本源的坐标列与 key（lon6/lat6 会与主表同名冲突，导致主表 lon6/lat6 被覆盖）
    return d.drop(columns=['lon', 'lat', 'lon6', 'lat6']).drop_duplicates('key')


def load_v2():
    c = ['lon', 'lat', 'prov', 'eco', 'elev_m', 'slope_deg', 'aspect_deg',
         'amp2019', 'amp2023', 'djf2019', 'djf2023', 'v2019', 'v2023']
    d = pq.read_table(V2_HABIT, columns=c).to_pandas()
    d = add_key(d)
    for c0 in ['v2019', 'v2023']:
        d[c0] = d[c0].replace('', np.nan)
    d = d.rename(columns={'prov': 'prov_v2', 'eco': 'eco_v2', 'elev_m': 'elev_v2',
                          'slope_deg': 'slope_v2', 'aspect_deg': 'aspect_v2',
                          'amp2019': 'amp2019_v2', 'amp2023': 'amp2023_v2',
                          'djf2019': 'djf2019_v2', 'djf2023': 'djf2023_v2',
                          'v2019': 'v2019_v2', 'v2023': 'v2023_v2'})
    return d.drop(columns=['lon', 'lat', 'lon6', 'lat6']).drop_duplicates('key')


# ---------------- 地形自算（末位兜底） ----------------
sys.path.insert(0, TERRAIN_LIB)
import terrain_lib as TL
import rasterio
from rasterio.windows import Window


def terrain_block(lon0, lat0, lon1, lat1):
    """NASADEM 窗口（外扩 pad）→ 2×2 均值 60m → Horn 坡度/坡向（ESRI 约定）。
    与 nb_national_2026/labels/code/s2_sample_national.py:terrain_tile 同法同参。"""
    with rasterio.open(DEM) as s:
        w = s.window(lon0 - DEM_PAD, lat0 - DEM_PAD, lon1 + DEM_PAD, lat1 + DEM_PAD)
        w = w.round_offsets().round_lengths()
        dem = np.array(s.read(1, window=w), dtype=np.float32, copy=True)
        res30 = abs(s.transform[0])
        x_off = float(s.transform.c + w.col_off * s.transform.a)
        y_off = float(s.transform.f + w.row_off * s.transform.e)
    dem = np.where(dem <= -32000, np.nan, dem)
    if np.isnan(dem).any():
        from scipy.ndimage import uniform_filter
        msk = np.isnan(dem)
        v = np.where(msk, 0.0, dem).astype(np.float32)
        cnt = uniform_filter((~msk).astype(np.float32), size=3, mode='nearest')
        sm = uniform_filter(v, size=3, mode='nearest')
        dem[msk] = (sm / np.maximum(cnt, 1e-6))[msk]
    H, W = dem.shape
    H2, W2 = (H // DEM_AGG) * DEM_AGG, (W // DEM_AGG) * DEM_AGG
    elev = dem[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2).mean(axis=(1, 3)).astype(np.float32)
    del dem
    res60 = res30 * DEM_AGG
    hs, ws = elev.shape
    asp = np.full((hs, ws), -1.0, dtype=np.float32)
    slp = np.zeros((hs, ws), dtype=np.float32)
    lat_c = float((lat0 + lat1) / 2)
    for r0 in range(0, hs, BAND):
        r1 = min(r0 + BAND, hs)
        a, b = max(0, r0 - 1), min(hs, r1 + 1)
        A, S = TL.aspect_slope(elev[a:b].astype(np.float64), res60, lat_c)
        ta, tb = max(r0, a + 1), min(r1, b - 1)
        if tb <= ta:
            continue
        asp[ta:tb, 1:-1] = A[ta - (a + 1):tb - (a + 1)]
        slp[ta:tb, 1:-1] = S[ta - (a + 1):tb - (a + 1)]
    xs = x_off + (np.arange(ws) + 0.5) * res60
    ys = y_off - (np.arange(hs) + 0.5) * res60
    return asp, slp, elev, xs, ys


def sample_grid(A, xs, ys, lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    ci = np.searchsorted(xs, lon); ci = np.clip(ci, 1, len(xs) - 1)
    ci = np.where(np.abs(xs[ci] - lon) < np.abs(xs[ci - 1] - lon), ci, ci - 1)
    inv = -ys
    ri = np.searchsorted(inv, -lat); ri = np.clip(ri, 1, len(inv) - 1)
    ri = np.where(np.abs(inv[ri] + lat) < np.abs(inv[ri - 1] + lat), ri, ri - 1)
    return A[ri, ci]


def terrain_at(lon, lat):
    e = np.full(len(lon), np.nan, np.float32)
    s = np.full(len(lon), np.nan, np.float32)
    a = np.full(len(lon), np.nan, np.float32)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    tlon = np.floor(lon / 2) * 2; tlat = np.floor(lat / 2) * 2
    kk = pd.Series(tlon.astype(str) + '_' + tlat.astype(str))
    n_tile = 0
    for k, idx in kk.groupby(kk, sort=False).groups.items():
        j = np.asarray(idx)
        x0, y0 = tlon[j[0]], tlat[j[0]]
        asp, slp, elev, xs, ys = terrain_block(x0, y0, x0 + 2, y0 + 2)
        e[j] = sample_grid(elev, xs, ys, lon[j], lat[j])
        s[j] = sample_grid(slp, xs, ys, lon[j], lat[j])
        a[j] = sample_grid(asp, xs, ys, lon[j], lat[j])
        n_tile += 1
    return e, s, a, n_tile


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-terrain-fill', action='store_true')
    ap.add_argument('--check-n', type=int, default=600)
    ap.add_argument('--smoke', type=int, default=0, help='调试：各源随机抽样 N 行后跑通全流程')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': os.path.abspath(__file__),
             'inputs': {'gpkg_dir': GPKG_DIR, 'master': MASTER, 'work_base': WORK_BASE,
                        'v2_habit': V2_HABIT, 'dem': DEM}}

    G, raw_gpkg = load_gpkg()
    M, raw_master = load_master()
    if args.smoke:
        # 调试模式：各源抽样，只用于验证流程（结果不可当产出）
        rng = np.random.default_rng(7)
        G = G.iloc[rng.choice(len(G), size=min(args.smoke, len(G)), replace=False)].reset_index(drop=True)
        M = M.iloc[rng.choice(len(M), size=min(args.smoke // 2, len(M)), replace=False)].reset_index(drop=True)
        emit('SMOKE 模式：GPKG %d 行 / 母库 %d 行' % (len(G), len(M)))
    stats['sources'] = {'gpkg': raw_gpkg, 'master': raw_master}

    pool, mstat = merge_pool(G, M)
    del G, M
    stats['merge'] = mstat
    stats['merge']['unique_keys'] = int(len(pool))
    stats['merge']['src_primary_counts'] = {str(k): int(v) for k, v in pool['src_primary'].value_counts().items()}
    stats['merge']['src_n_counts'] = {str(k): int(v) for k, v in pool['src_n'].value_counts().sort_index().items()}

    # ---- 特征 join ----
    wb = load_wb(); v2 = load_v2()
    pool = pool.merge(wb, on='key', how='left')
    pool = pool.merge(v2, on='key', how='left')
    del wb, v2
    stats['join_coverage'] = {
        'work_base_key_hit': int(pool['elev_wb'].notna().sum()),
        'work_base_key_hit_pct': round(100 * float(pool['elev_wb'].notna().mean()), 3),
        'v2_key_hit': int(pool['elev_v2'].notna().sum()),
        'v2_key_hit_pct': round(100 * float(pool['elev_v2'].notna().mean()), 3),
        'either_hit': int((pool['elev_wb'].notna() | pool['elev_v2'].notna()).sum()),
        'either_hit_pct': round(100 * float((pool['elev_wb'].notna() | pool['elev_v2'].notna()).mean()), 3),
    }
    emit('join：work_base %.2f%%，v2 %.2f%%' % (stats['join_coverage']['work_base_key_hit_pct'],
                                               stats['join_coverage']['v2_key_hit_pct']))

    def pick(a, b):
        return np.where(pd.notna(pool[a]), pool[a], pool[b])

    pool['amp2019'] = pick('amp2019', 'amp2019_v2')
    pool['amp2023'] = pick('amp2023', 'amp2023_v2')
    pool['djf2019'] = pick('djf2019', 'djf2019_v2')
    pool['djf2023'] = pick('djf2023', 'djf2023_v2')
    pool['v2019'] = pick('v2019', 'v2019_v2')
    pool['v2023'] = pick('v2023', 'v2023_v2')
    pool['elev_m'] = pick('elev_wb', 'elev_v2')
    pool['slope_deg'] = pick('slope_wb', 'slope_v2')
    pool['aspect_deg'] = pick('aspect_wb', 'aspect_v2')

    ph = np.where(pool['amp2023'].notna(),
                  np.where(pool['amp2023_v2'].isna(), 'wb', 'wb|v2'), 'none')
    te = np.where(pool['elev_wb'].notna(), 'wb', np.where(pool['elev_v2'].notna(), 'v2', 'none'))
    cl = np.where(pool['bio01'].notna(), 'wb', 'none')
    pool['feat_src'] = pd.Series('pheno:' + pd.Series(ph).astype(str) + ';terrain:' + pd.Series(te).astype(str)
                                 + ';climate:' + pd.Series(cl).astype(str)).to_numpy()
    pool['terrain_src'] = te
    pool = pool.drop(columns=['amp2019_v2', 'amp2023_v2', 'djf2019_v2', 'djf2023_v2', 'v2019_v2', 'v2023_v2',
                              'elev_wb', 'slope_wb', 'aspect_wb', 'elev_v2', 'slope_v2', 'aspect_v2'])

    # ---- 地形自算兜底（work_base 与 v2 都缺的点，主要是混交 91/92） ----
    stats['terrain_selfcheck'] = None
    miss_t = pool['elev_m'].isna().to_numpy()
    stats['terrain_fill'] = {'n_missing_before': int(miss_t.sum())}
    if miss_t.any() and not args.no_terrain_fill:
        # 方法自检：在已有地形的点上抽样自算，比较一致度
        have = np.flatnonzero(~miss_t)
        rng = np.random.default_rng(20261025)
        sidx = rng.choice(have, size=int(min(args.check_n, len(have))), replace=False)
        e_t, s_t, a_t, nt = terrain_at(pool['lon'].to_numpy()[sidx], pool['lat'].to_numpy()[sidx])
        base_e = pool['elev_m'].to_numpy()[sidx]; base_s = pool['slope_deg'].to_numpy()[sidx]
        base_a = pool['aspect_deg'].to_numpy()[sidx]
        de = np.abs(e_t - base_e); ds = np.abs(s_t - base_s)
        da = np.abs((a_t - base_a + 180) % 360 - 180)
        stats['terrain_selfcheck'] = {
            'n_sample': int(len(sidx)), 'n_tile': int(nt),
            'elev_absdiff_p50': round(float(np.nanmedian(de)), 3),
            'elev_absdiff_p99': round(float(np.nanquantile(de, .99)), 3),
            'slope_absdiff_p50': round(float(np.nanmedian(ds)), 3),
            'slope_absdiff_p99': round(float(np.nanquantile(ds, .99)), 3),
            'aspect_circdiff_p50': round(float(np.nanmedian(da)), 3),
            'aspect_circdiff_p99': round(float(np.nanquantile(da, .99)), 3),
            'note': 'existing terrain = work_base(NASADEM 60m) 或 v2(NASADEM_30m_agg60m)，本自算用 NASADEM 30m→60m 同法；差异来自采样网格约定与 DEM 版本',
        }
        emit('地形自检 %d 点 / %d 瓦：%s' % (len(sidx), nt, json.dumps(stats['terrain_selfcheck'], ensure_ascii=False)))
        del e_t, s_t, a_t, base_e, base_s, base_a, de, ds, da

        mi = np.flatnonzero(miss_t)
        e_f, s_f, a_f, nt2 = terrain_at(pool['lon'].to_numpy()[mi], pool['lat'].to_numpy()[mi])
        pool.loc[pool.index[mi], 'elev_m'] = e_f
        pool.loc[pool.index[mi], 'slope_deg'] = s_f
        pool.loc[pool.index[mi], 'aspect_deg'] = a_f
        stats['terrain_fill'].update({'n_filled': int(np.isfinite(e_f).sum()), 'n_tile': int(nt2),
                                      'still_missing': int((~np.isfinite(e_f)).sum())})
        emit('地形自算补齐 %d 点 / %d 瓦（仍缺 %d）' % (
            stats['terrain_fill']['n_filled'], nt2, stats['terrain_fill']['still_missing']))
        pool.loc[pool.index[mi], 'terrain_src'] = 'dem60'
        pool['feat_src'] = pool['feat_src'].str.replace('terrain:none', 'terrain:dem60', regex=False)
        del e_f, s_f, a_f
    for c in ['elev_m', 'slope_deg', 'aspect_deg']:
        pool[c] = pd.to_numeric(pool[c], errors='coerce').astype(np.float32)

    # 气候缺口：只记最近邻距离（诚实标记，不外推估值）
    pool['clim_nn_dist_km'] = np.float32(0.0)
    miss_c = pool['bio01'].isna().to_numpy()
    if miss_c.any():
        from scipy.spatial import cKDTree
        ok = pool.loc[~miss_c, ['lon', 'lat']].to_numpy(float)
        t = cKDTree(ok)
        d, _ = t.query(pool.loc[miss_c, ['lon', 'lat']].to_numpy(float), k=1)
        pool.loc[miss_c, 'clim_nn_dist_km'] = (d * 111.0).astype(np.float32)
        km = d * 111.0
        stats['climate_gap_nn_km'] = {'n_missing': int(miss_c.sum()),
                                      'p50': round(float(np.median(km)), 2),
                                      'p90': round(float(np.quantile(km, .9)), 2),
                                      'max': round(float(km.max()), 2),
                                      'note': '气候仅从 work_base 取，缺口处不外推；供下游判定是否补采'}
        emit('气候缺失 %d 点，其最近 work_base 点 p50=%.1f km / p90=%.1f km' % (
            int(miss_c.sum()), stats['climate_gap_nn_km']['p50'], stats['climate_gap_nn_km']['p90']))

    # ---- 省与生态区 ----
    prov = pool['prov'].astype(object).copy()
    prov_src = pd.Series('gpkg', index=pool.index, dtype=object)
    for col, tag in [('ct_prov', 'gpkg_ct_prov'), ('prov_wb', 'work_base'),
                     ('prov_cta', 'work_base_cta'), ('prov_v2', 'v2')]:
        mm = prov.isna() & pool[col].notna()
        prov[mm] = pool.loc[mm, col]; prov_src[mm] = tag
    # 末位兜底：DataV 省界点面相交（权威边界，非插值；用于 work_base 未覆盖的 91/92 混交点）
    n_miss0 = int(prov.isna().sum())
    if n_miss0 and os.path.exists(DATAV):
        import geopandas as gpd
        from shapely.geometry import Point
        d = json.load(open(DATAV, encoding='utf-8'))
        gj = gpd.GeoDataFrame.from_features(d['features'], crs='EPSG:4326')
        gj = gj[gj['name'].astype(str).str.len() > 0][['name', 'geometry']]
        mi = np.flatnonzero(prov.isna().to_numpy())
        pts = gpd.GeoDataFrame(pool.iloc[mi][['lon', 'lat']].copy(),
                               geometry=[Point(x, y) for x, y in
                                         zip(pool['lon'].to_numpy()[mi], pool['lat'].to_numpy()[mi])],
                               crs='EPSG:4326')
        sj = gpd.sjoin(pts, gj, how='left', predicate='intersects')
        sj = sj[~sj.index.duplicated(keep='first')]
        got = sj['name'].notna()
        idx = pool.index[mi][got.to_numpy()]
        prov.loc[idx] = sj.loc[got, 'name'].to_numpy()
        prov_src.loc[idx] = 'datav_sjoin'
        stats['prov_datav_sjoin'] = {'n_missing_before': n_miss0,
                                     'n_resolved': int(got.sum()),
                                     'n_still_missing': int(n_miss0 - got.sum())}
        emit('省归属 DataV 点面兜底：%d 缺失 → 补 %d，仍缺 %d' % (
            n_miss0, int(got.sum()), n_miss0 - int(got.sum())))
    prov_src[prov.isna()] = 'none'
    pool['prov'] = prov
    pool['prov_src'] = prov_src
    eco = prov.map(PROV2ECO)
    eco = eco.where(eco.notna(), pool['eco_v2'])
    pool['eco'] = eco.fillna('未匹配')
    # cz 可由 bio06 精确推导（实测 work_base 466,135 点 100% 一致）→ 仅用于 bio06 有值而 cz 缺失的行
    cz_miss = pool['cz'].isna() & pool['bio06'].notna()
    if cz_miss.any():
        b = pool.loc[cz_miss, 'bio06'].to_numpy(float)
        pool.loc[cz_miss, 'cz'] = np.where(b > 30, 'z1', np.where(b > 0, 'z2',
                                          np.where(b > -50, 'z3', np.where(b > -100, 'z4', 'z5'))))
        stats['cz_derived_from_bio06'] = int(cz_miss.sum())
    stats['prov_src_counts'] = {str(k): int(v) for k, v in pool['prov_src'].value_counts().items()}
    pool = pool.drop(columns=['prov_wb', 'prov_cta', 'prov_v2', 'eco_v2'])

    # ---- QC ----
    qc = {}
    lon = pool['lon'].to_numpy(float); lat = pool['lat'].to_numpy(float)
    nan_ll = ~np.isfinite(lon) | ~np.isfinite(lat)
    oob = (~nan_ll) & ((lon < 73) | (lon > 136) | (lat < 17) | (lat > 54))
    aef_nan = pool[AEF].isna().all(axis=1).to_numpy()
    qc['lonlat_nan'] = int(nan_ll.sum())
    qc['out_of_bbox'] = int(oob.sum())
    qc['out_of_bbox_detail'] = {
        'lon_range': [float(lon[oob].min()), float(lon[oob].max())] if oob.any() else None,
        'lat_range': [float(lat[oob].min()), float(lat[oob].max())] if oob.any() else None,
        'by_prov': {str(k): int(v) for k, v in pool.loc[oob, 'prov'].value_counts().items()},
        'by_cls6': {str(k): int(v) for k, v in pool.loc[oob, 'cls6'].value_counts().items()},
    }
    qc['aef_all_nan'] = int(aef_nan.sum())
    qc['aef_partial_nan_rows'] = int(pool[AEF].isna().any(axis=1).sum())
    drop = nan_ll | oob | aef_nan
    qc['dropped_total'] = int(drop.sum())
    pool = pool.loc[~drop].copy()
    qc['rows_after_qc'] = int(len(pool))
    dup = int(pool['key'].duplicated().sum())
    qc['dup_key_after_dedup'] = dup
    assert dup == 0, '去重后仍有重复坐标键 %d' % dup
    # 坐标键自洽（防 join 覆盖 lon6/lat6 —— 曾发生：work_base/v2 也带同名列）
    k_chk = np.char.add(np.char.add(np.char.mod('%.6f', pool['lon'].to_numpy()[~drop]),
                                    '_'), np.char.mod('%.6f', pool['lat'].to_numpy()[~drop]))
    k_src = np.char.add(np.char.add(np.char.mod('%.6f', pool['lon6'].to_numpy()),
                                    '_'), np.char.mod('%.6f', pool['lat6'].to_numpy()))
    qc['key_recompute_mismatch'] = int((k_chk != k_src).sum())
    assert qc['key_recompute_mismatch'] == 0, 'lon6/lat6 与 lon/lat 不自洽 %d 行' % qc['key_recompute_mismatch']

    A = pool[AEF].to_numpy(np.float64)
    colstat, bad = {}, []
    for i, c in enumerate(AEF):
        v = A[:, i]; ok = np.isfinite(v)
        mn, mx, sd = float(v[ok].min()), float(v[ok].max()), float(v[ok].std())
        colstat[c] = {'min': round(mn, 5), 'max': round(mx, 5), 'mean': round(float(v[ok].mean()), 5),
                      'std': round(sd, 5), 'n_nan': int((~ok).sum()), 'n_allzero': int((v[ok] == 0).sum())}
        if mn < -3 or mx > 3 or sd == 0:
            bad.append(c)
    gmin, gmax = float(np.nanmin(A)), float(np.nanmax(A))
    qc['aef_range'] = {'global_min': round(gmin, 5), 'global_max': round(gmax, 5),
                       'global_mean': round(float(np.nanmean(A)), 5),
                       'bad_columns': bad, 'per_column': colstat}
    qc['aef_verdict'] = 'PASS' if (not bad and -3 <= gmin and gmax <= 3) else 'FAIL'
    del A
    qc['all_pass'] = bool(qc['out_of_bbox'] == 0 and qc['lonlat_nan'] == 0 and
                          qc['aef_all_nan'] == 0 and dup == 0 and
                          qc['key_recompute_mismatch'] == 0 and qc['aef_verdict'] == 'PASS')
    stats['qc'] = qc

    # ---- 缺失率 ----
    stats['missing_rate'] = {c: round(float(pool[c].isna().mean()), 6) for c in pool.columns}
    miss = stats['missing_rate']
    stats['missing_by_group'] = {
        'AEF64_all_nan_rows': round(float(pool[AEF].isna().all(axis=1).mean()), 6),
        'AEF64_any_nan_rows': round(float(pool[AEF].isna().any(axis=1).mean()), 6),
        'pheno_amp2019': miss['amp2019'], 'pheno_amp2023': miss['amp2023'],
        'pheno_djf2019': miss['djf2019'], 'pheno_djf2023': miss['djf2023'],
        'pheno_v2019': miss['v2019'], 'pheno_v2023': miss['v2023'],
        'terrain_elev_m': miss['elev_m'], 'terrain_slope_deg': miss['slope_deg'],
        'terrain_aspect_deg': miss['aspect_deg'],
        'climate_bio01': miss['bio01'], 'climate_bio06': miss['bio06'],
        'climate_bio11': miss['bio11'], 'climate_bio12': miss['bio12'],
        'cz': miss['cz'], 'prov': miss['prov'], 'eco_unmatched_rows': int((pool['eco'] == '未匹配').sum()),
    }
    stats['feat_src_counts'] = {str(k): int(v) for k, v in pool['feat_src'].value_counts().items()}
    stats['terrain_src_counts'] = {str(k): int(v) for k, v in pool['terrain_src'].value_counts().items()}

    # ---- 六类计数 ----
    cnt = pool['cls6'].value_counts().sort_index()
    stats['cls6_counts'] = {str(int(k)): int(v) for k, v in cnt.items()}
    stats['cls6_names'] = {str(int(k)): CLS6_NAME.get(int(k), '?') for k in cnt.index}
    stats['cls6_by_source'] = {}
    for k, g in pool.groupby('cls6', sort=True):
        stats['cls6_by_source'][str(int(k))] = {
            'total': int(len(g)),
            'primary_gpkg': int((g['src_rank'] < 3).sum()),
            'primary_master': int((g['src_rank'] == 3).sum()),
            'layer': {str(a): int(b) for a, b in g['layer'].value_counts().items()},
            'tier': {str(a): int(b) for a, b in g['tier'].value_counts().items()},
            'by_eco': {str(a): int(b) for a, b in g['eco'].value_counts().items()},
            'by_prov': {str(a): int(b) for a, b in g['prov'].value_counts().items()},
        }
    # 混交点专属统计（后续 F2 的核心对象）
    if 45 in set(pool['cls6']) or 46 in set(pool['cls6']):
        mix = pool[pool['cls6'].isin([45, 46])]
        stats['mixed_45_46'] = {
            'n': int(len(mix)),
            'code_master': {str(k): int(v) for k, v in mix['code_master'].value_counts().items()},
            'by_eco': {str(k): int(v) for k, v in mix['eco'].value_counts().items()},
            'by_prov': {str(k): int(v) for k, v in mix['prov'].value_counts().items()},
            'lon_range': [round(float(mix['lon'].min()), 4), round(float(mix['lon'].max()), 4)],
            'lat_range': [round(float(mix['lat'].min()), 4), round(float(mix['lat'].max()), 4)],
            'pheno_missing': {c: round(float(mix[c].isna().mean()), 4)
                              for c in ['amp2019', 'amp2023', 'djf2019', 'djf2023']},
            'terrain_src': 'dem60 自算' if 'terrain:dem60' in ' '.join(mix['feat_src'].unique()) else 'join',
            'by_1deg_top10': {('%dE_%dN' % (int(a), int(b))): int(c) for (a, b), c in
                              mix.groupby([mix['lon'].round(0), mix['lat'].round(0)]).size()
                              .sort_values(ascending=False).head(10).items()},
        }

    # ---- 分区表 ----
    def pivot(df, col):
        t = pd.crosstab(df[col], df['cls6'])
        t.columns = ['n_%d' % c for c in t.columns]
        for c in ['n_0', 'n_41', 'n_42', 'n_43', 'n_44', 'n_45', 'n_46']:
            if c not in t.columns:
                t[c] = 0
        t = t[['n_41', 'n_42', 'n_43', 'n_44', 'n_45', 'n_46', 'n_0']]
        t['n_total'] = t.sum(axis=1)
        return t.sort_values('n_total', ascending=False).reset_index().rename(columns={col: 'name'})

    byeco = pivot(pool, 'eco'); byprov = pivot(pool, 'prov')
    byeco.to_csv(os.path.join(OUT_DIR, 'f1_by_eco.csv'), index=False, encoding='utf-8-sig')
    byprov.to_csv(os.path.join(OUT_DIR, 'f1_by_prov.csv'), index=False, encoding='utf-8-sig')
    stats['by_eco'] = byeco.to_dict('records')
    stats['by_prov'] = byprov.to_dict('records')
    stats['eco_order_present'] = [e for e in ECO_ORDER if e in set(pool['eco'])]
    stats['eco_missing_from_doc84'] = [e for e in ECO_ORDER if e not in set(pool['eco'])]
    stats['eco_unmatched'] = int((pool['eco'] == '未匹配').sum())

    # ---- missing_flags ----
    fl = np.full(len(pool), '', dtype=object)
    def addflag(mask, tag):
        nonlocal fl
        fl = np.where(mask, np.where(fl == '', tag, np.char.add(np.char.add(fl, ';'), tag)), fl)
    addflag(pool[AEF].isna().any(axis=1).to_numpy(), 'aef')
    addflag((pool['amp2023'].isna() & pool['amp2019'].isna()).to_numpy(), 'pheno')
    addflag(pool['elev_m'].isna().to_numpy(), 'terrain')
    addflag(pool['bio01'].isna().to_numpy(), 'climate')
    addflag(pool['cz'].isna().to_numpy(), 'cz')
    addflag((pool['eco'] == '未匹配').to_numpy(), 'eco')
    pool['missing_flags'] = fl
    stats['missing_flags_counts'] = {str(k): int(v) for k, v in pd.Series(fl).value_counts().items()}

    # ---- 落盘 ----
    pool['uid'] = np.arange(len(pool), dtype=np.int64)
    cols = (['uid', 'lon', 'lat', 'lon6', 'lat6', 'src_list', 'src_primary', 'src_n',
             'prov', 'prov_src', 'eco', 'cls6', 'cls4_gpkg', 'code_master',
             'conflict_4class', 'conflict_classes', 'layer', 'tier', 'src_orig', 'master_row_id']
            + AEF +
            ['amp2019', 'amp2023', 'djf2019', 'djf2023', 'v2019', 'v2023',
             'elev_m', 'slope_deg', 'aspect_deg', 'bio01', 'bio06', 'bio11', 'bio12', 'cz',
             'cgls_ft', 'terrain_src', 'feat_src', 'clim_nn_dist_km', 'missing_flags'])
    out = pool[cols].sort_values(['cls6', 'lon', 'lat'], kind='stable').reset_index(drop=True)
    out['uid'] = np.arange(len(out), dtype=np.int64)
    out['cls4_gpkg'] = out['cls4_gpkg'].where(out['cls4_gpkg'] > 0).astype('Int8')
    out['code_master'] = out['code_master'].where(out['code_master'] > 0).astype('Int16')
    out['master_row_id'] = out['master_row_id'].where(out['master_row_id'] >= 0).astype('Int64')
    for c in ['amp2019', 'amp2023', 'djf2019', 'djf2023', 'bio01', 'bio06', 'bio11',
              'bio12', 'cgls_ft', 'clim_nn_dist_km', 'elev_m', 'slope_deg', 'aspect_deg']:
        out[c] = out[c].astype(np.float32)
    out['lon'] = out['lon'].astype(np.float64); out['lat'] = out['lat'].astype(np.float64)
    pqpath = os.path.join(OUT_DIR, 'forest_pool_v1.parquet')
    out.to_parquet(pqpath, index=False, compression='zstd', row_group_size=100000)
    stats['output'] = {'path': pqpath, 'rows': int(len(out)), 'cols': int(out.shape[1]),
                       'size_mb': round(os.path.getsize(pqpath) / 1e6, 2), 'columns': list(out.columns)}
    bk = pq.read_table(pqpath, columns=['uid', 'cls6'])
    stats['output']['readback_rows'] = int(bk.num_rows)
    assert bk.num_rows == len(out)
    del bk, out, pool

    stats['elapsed_s'] = round(time.time() - T0, 1)
    with open(os.path.join(OUT_DIR, 'f1_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    emit('完成 %s：%d 行 %d 列 %.1f MB，用时 %.1f s' % (
        pqpath, stats['output']['rows'], stats['output']['cols'],
        stats['output']['size_mb'], stats['elapsed_s']))
    emit('六类：%s' % json.dumps(stats['cls6_counts'], ensure_ascii=False))
    emit('QC all_pass=%s, AEF verdict=%s' % (stats['qc']['all_pass'], stats['qc']['aef_verdict']))
    return stats


if __name__ == '__main__':
    main()
