# -*- coding: utf-8 -*-
"""
s3_external.py — r1 步骤2：外部点集统一入库 → r1_ext.parquet
* 八个来源，全部经：统一映射(s0_conf/lc_conf) → 国界裁剪(s1_geom) → 互斥抽稀：
    1 esri_nat_2020   取2020年；built(7)→GUB拆190/200；青藏高原耕地降权0.3
    2 ne_crops        东北稻/玉米/大豆 2017-2025（label 时不变，保留全年份）
    3 worldcereal     our_class(10/12)，conf 0.9
    4 eglc            已映射成品
    5 gpw_grass       class∈{1,2}→130 草地，conf 0.95
    6 gwl_fcs30       code 181-187 → CODE_MAP 平移(→180-186)，conf 0.9；180语义不明丢弃
    7 gisa_gisd       修复旧链bug：gisa_year 是索引码，真实年份=1971+码；
                      ≤2000 首城市化→190 城镇，>2000→200 乡村
    8 orchard_glc12   →11 园地，conf 0.6
    （GLC_FCS10 不在此：由 s4 从本地栅格全覆盖重采；amur 码表未知不入库）
* 互斥：距 r1_base <100m 丢弃；源内 250m 贪心抽稀；跨源 100m 去重（稀缺专题源优先保留）。
* 输出列: lon/lat/class_new/src/src_conf/year，tier='external'
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

R_KM = 0.1      # 距底座/跨源互斥半径 100m
THIN_M = 250    # 源内抽稀间距

# 跨源去重保留优先级（稀缺/专题源在前）
SRC_PRIORITY = ['orchard_glc12', 'gwl_fcs30_stable', 'gpw_grass_vhr',
                'glc12_2020_single', 'worldcereal', 'eglc', 'ne_crops',
                'esri_nat_2020', 'gisa_gisd']

def to_km(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])

def clip_china(df):
    lon = df['lon'].to_numpy(float); lat = df['lat'].to_numpy(float)
    b = C.BBOX
    m = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
    df = df[m].reset_index(drop=True)
    if not len(df):
        return df
    keep = G.china_contains(df['lon'].to_numpy(), df['lat'].to_numpy())
    return df[keep].reset_index(drop=True)

def std(lon, lat, cls, src, conf, year, class_raw=None):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    n = len(lon)
    df = pd.DataFrame({'lon': lon, 'lat': lat,
                       'class_new': np.broadcast_to(np.asarray(cls, np.int64), (n,)).copy(),
                       'src': src, 'src_conf': np.full(n, conf, float),
                       'year': np.full(n, year, np.int64)})
    if class_raw is not None:
        df['class_raw'] = np.asarray(class_raw)[df.index] if len(np.atleast_1d(class_raw)) == n else np.broadcast_to(np.asarray(class_raw), (n,)).copy()
    return clip_china(df)

def load_sources():
    out = {}

    # 1 ESRI 全国 2020
    es = pd.read_parquet(os.path.join(C.EXT, 'esri_national_yearly.parquet'))
    es = es[es['year'] == 2020].copy()
    built = es['class_new'] == 7
    if built.any():
        cls = es['class_new'].to_numpy(np.int64).copy()
        in_gub = G.gub_contains(es.loc[built, 'lon'].to_numpy(), es.loc[built, 'lat'].to_numpy())
        cls[built] = np.where(in_gub, 190, 200)
        es['class_new'] = cls
        es['src'] = 'esri_nat_2020'
    # ESRI 青藏高原耕地已知噪声降权（lat>30 & lon<104）
    tp = (es['class_new'] == 10) & (es['lat'] > 30) & (es['lon'] < 104)
    es.loc[tp, 'src_conf'] = 0.3
    es['year'] = 2020
    es['src'] = 'esri_nat_2020'
    out['esri_nat_2020'] = es[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'class_raw']]
    print(f"esri_nat_2020: {len(es):,}（built拆分 {int(built.sum()):,}，青藏耕地降权 {int(tp.sum()):,}）")

    # 3 东北作物
    n = pd.read_parquet(os.path.join(C.EXT, 'ne_crops_samples.parquet'))
    out['ne_crops'] = n[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year']]
    print(f"ne_crops: {len(n):,}")

    # 4 WorldCereal
    w = pd.read_parquet(os.path.join(C.EXT, 'worldcereal_china.parquet'))
    out['worldcereal'] = std(w.lon, w.lat, w.our_class.astype(int),
                             'worldcereal_2018_Asia', 0.9, 2018)
    print(f"worldcereal: {len(out['worldcereal']):,}")

    # 5 EGLC
    e = pd.read_parquet(os.path.join(C.EXT, 'eglc_china.parquet'))
    out['eglc'] = e[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year']]
    print(f"eglc: {len(e):,}")

    # 6 GPW 草地
    gp = pd.read_parquet(os.path.join(C.EXT, 'grass_gpw_china.parquet'))
    g = gp[gp['class'].isin([1, 2])]
    out['gpw_grass_vhr'] = std(g.lon, g.lat, 130, 'gpw_grass_vhr', 0.95, 2024)
    print(f"gpw_grass_vhr: {len(out['gpw_grass_vhr']):,}")

    # 7 GWL 湿地（181-187 → CODE_MAP 平移；180 语义不明丢弃）
    wl = pd.read_parquet(os.path.join(C.EXT, 'gee_thematic/wetland_gwl.parquet'))
    wl = wl[wl['code'].between(181, 187)].copy()
    wcls = C.CODE_MAP  # 181:180 ... 187:186
    out['gwl_fcs30_stable'] = std(wl.lon, wl.lat,
                                  wl['code'].map(wcls).astype(int),
                                  'gwl_fcs30_stable', 0.9, 2022,
                                  class_raw=wl['code'])
    print(f"gwl_fcs30_stable: {len(out['gwl_fcs30_stable']):,}")

    # 8 GISA/GISD 不透水（修复旧链城乡 bug：年份=1971+索引码）
    im = pd.read_parquet(os.path.join(C.EXT, 'gee_thematic/imperv_gisa_raw.parquet'))
    uy = 1971 + pd.to_numeric(im['gisa_year'], errors='coerce').fillna(0).astype(int)
    icls = np.where(uy <= 2000, 190, 200)
    dfi = std(im.lon, im.lat, icls, 'gisa_gisd', 0.75, 2021, class_raw=im['gisa_year'])
    out['gisa_gisd'] = dfi
    print(f"gisa_gisd: {len(dfi):,}（城镇 {int((dfi.class_new==190).sum()):,} / 乡村 {int((dfi.class_new==200).sum()):,}）")

    # 9 园地 glc12
    orc = pd.read_parquet(os.path.join(C.EXT, 'gee_thematic/orchard_glc12.parquet'))
    oc = orc[orc['code'] == 12]
    out['orchard_glc12'] = std(oc.lon, oc.lat, 11, 'glc12_2020_single', 0.6, 2020)
    print(f"orchard_glc12: {len(out['orchard_glc12']):,}")

    return out

def thin_within(df, thin_m=THIN_M, seed=C.SEED):
    """源内贪心抽稀（0.25km 桶加速，n4 模式）"""
    if len(df) <= 1:
        return df, 0
    n0 = len(df)
    km = to_km(df.lon.to_numpy(), df.lat.to_numpy())
    order = np.random.default_rng(seed).permutation(len(df))
    cell = 0.25
    kept, keep_idx = {}, []
    for i in order:
        key = (int(km[i, 0] // cell), int(km[i, 1] // cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((key[0] + dx, key[1] + dy), []):
                    if np.hypot(km[i, 0] - km[j, 0], km[i, 1] - km[j, 1]) < thin_m / 1000:
                        ok = False; break
                if not ok: break
            if ok: break
        if ok:
            kept.setdefault(key, []).append(i)
            keep_idx.append(i)
    return df.iloc[sorted(keep_idx)].reset_index(drop=True), n0 - len(keep_idx)

def main():
    t0 = time.time()
    base = pd.read_parquet(C.R1_BASE, columns=['lon', 'lat'])
    base_tree = cKDTree(to_km(base.lon.to_numpy(), base.lat.to_numpy()))
    print(f'底座 {len(base):,} 点，建树完成 ({time.time()-t0:.0f}s)', flush=True)

    srcs = load_sources()
    frames = []
    print('\n=== 各源：国界内 → 距底座<100m剔除 → 源内250m抽稀 ===')
    for name in SRC_PRIORITY:
        if name not in srcs:
            continue
        df = srcs[name]
        n0 = len(df)
        d, _ = base_tree.query(to_km(df.lon.to_numpy(), df.lat.to_numpy()), k=1)
        df = df[d > R_KM].reset_index(drop=True)
        n1 = len(df)
        df, ndrop = thin_within(df)
        print(f'  {name:<20} {n0:>9,} → 距底座剔{n0-n1:>8,} → 源内抽稀剔{ndrop:>8,} → {len(df):>9,}')
        frames.append(df)

    ext = pd.concat(frames, ignore_index=True)
    # 跨源 100m 互去重（优先级序已拼接，保留前者）
    tree = cKDTree(to_km(ext.lon.to_numpy(), ext.lat.to_numpy()))
    pairs = tree.query_pairs(R_KM)
    drop = {b for _, b in pairs}
    ext = ext.drop(index=list(drop)).reset_index(drop=True)
    print(f'\n跨源互去重剔 {len(drop):,} → 合计 {len(ext):,}')

    ext['tier'] = 'external'
    ext.to_parquet(C.R1_EXT, index=False)
    print('\nr1_ext 类别分布:')
    for k, v in ext['class_new'].value_counts().sort_index().items():
        print(f'  {k:>4}: {v:,}')
    print('按源:')
    print(ext['src'].str.split('_').str[:2].str.join('_').value_counts().to_string())

    summary = {'total': int(len(ext)),
               'by_class': {str(k): int(v) for k, v in ext['class_new'].value_counts().sort_index().items()},
               'by_src': {k: int(v) for k, v in ext['src'].value_counts().items()},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(C.R1_EXT.replace('.parquet', '_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('\n输出:', C.R1_EXT, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
