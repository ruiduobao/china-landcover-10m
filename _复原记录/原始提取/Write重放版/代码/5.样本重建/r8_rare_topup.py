# -*- coding: utf-8 -*-
"""
r8_rare_topup.py — P3.1 稀有类定向补样（候选点生成；纯本地栅格，不发 GEE 请求）
* 缺口（样本诊断/class_balance_design.json）：91 缺 992 / 140 缺 1737 /
  183 缺 1087 / 184 缺 861（目标 ~2000 点/类）。
* 来源（码表见 lc_conf.CODE_MAP）：
    FCS10 2023 raw 92  → 产品 91（郁闭针阔混交林）
    FCS10 2023 raw 140 → 产品 140（地衣苔藓）
    FCS10 2023 raw 184 → 产品 183（盐渍湿地）
    GMW v3 2020 value=1 → 产品 184（红树林，主源）
* 采样：3×3 纯净像元、每 0.05° 格每类 ≤2（固定种子）、国界 + 生态规则过滤。
* 输出：数据/本地处理/样本重建/r8_rare_candidates.parquet
   （仅候选点，嵌入提取在 P3.1b 单独发起，避免与 r7 主提取抢 GEE 配额）
"""
import os, re, sys, glob, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

# raw FCS10 → 产品码（仅本任务 3 类）
TARGET = {92: 91, 140: 140, 184: 183}
T_LUT = np.zeros(256, dtype=bool)
for _k in TARGET:
    T_LUT[_k] = True

PER_FINE_CELL = 2          # 每 0.05° 格每类上限
FINE = 0.05
GMW_DIR = os.path.join(C.RARE_DIR, 'GlobalMangroveWatch_v3', 'china_tiles_2020')

_polys = None


def _winit():
    global _polys
    _polys = G.china_polys()


def pure_mask(a):
    c = a[1:-1, 1:-1]
    p = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            p &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return p


def thin_fine(df, key_cls='class_new'):
    """每 0.05° 格每类 ≤ PER_FINE_CELL（固定种子乱序后 head）"""
    if len(df) == 0:
        return df
    fine = (np.floor(df.lon.to_numpy() / FINE).astype(np.int64) * 100000 +
            np.floor(df.lat.to_numpy() / FINE).astype(np.int64))
    df = df.assign(_f=fine)
    df = df.sample(frac=1.0, random_state=C.SEED)
    return df.groupby(['_f', key_cls], group_keys=False).head(PER_FINE_CELL).drop(columns='_f')


def fcs10_tiles():
    out = []
    for fp in glob.glob(os.path.join(C.FCS10_TILE_ROOT, 'GLC_FCS10maps_*', '*.tif')):
        m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not m:
            continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        b = C.BBOX
        if lon0 + 5 > b[0] and lon0 < b[2] and lat0 > b[1] and lat0 - 5 < b[3]:
            out.append(fp)
    return out


def tile_task(fp):
    m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    b = C.BBOX
    if lon0 + 5 <= b[0] or lon0 >= b[2] or lat0 - 5 >= b[3] or lat0 <= b[1]:
        return None
    out = []
    try:
        with rasterio.open(fp) as s:
            inv = ~s.transform
            for cx in range(max(int(np.floor(b[0])), lon0), min(int(np.ceil(b[2])), lon0 + 5)):
                for cy in range(max(int(np.floor(b[1])), lat0 - 5), min(int(np.ceil(b[3])), lat0)):
                    cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                    cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                    cc0 = max(0, cc0 - 1); rr0 = max(0, rr0 - 1)
                    cc1 = min(s.width, cc1 + 2); rr1 = min(s.height, rr1 + 2)
                    if cc1 <= cc0 or rr1 <= rr0:
                        continue
                    a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                    if a.ndim == 3:
                        a = a[0]
                    a = a.astype(np.uint8)
                    center = a[1:-1, 1:-1]
                    rows, cols = np.where(pure_mask(a) & T_LUT[center])
                    if len(rows) == 0:
                        continue
                    raw_c = center[rows, cols]
                    cls_c = np.array([TARGET[int(v)] for v in raw_c], dtype=np.int64)
                    X, Y = riotrans.xy(s.transform, rows + rr0 + 0.5, cols + cc0 + 0.5)
                    X = np.asarray(X); Y = np.asarray(Y)
                    out.append(pd.DataFrame({'lon': X, 'lat': Y, 'class_new': cls_c,
                                             'class_raw': raw_c}))
        if out:
            df = pd.concat(out, ignore_index=True)
            return thin_fine(df)
        return None
    except Exception as e:
        print('FCS10 ERR', os.path.basename(fp), str(e)[:90], flush=True)
        return None


def gmw_task(fp):
    try:
        with rasterio.open(fp) as s:
            a = s.read(1)
            if a.ndim == 3:
                a = a[0]
            center = a[1:-1, 1:-1]
            # GMW: 1=红树林；纯净 3×3
            rows, cols = np.where(pure_mask(a) & (center == 1))
            if len(rows) == 0:
                return None
            X, Y = riotrans.xy(s.transform, rows + 1 + 0.5, cols + 1 + 0.5)
            df = pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                               'class_new': 184, 'class_raw': 1})
            return thin_fine(df)
    except Exception as e:
        print('GMW ERR', os.path.basename(fp), str(e)[:90], flush=True)
        return None


def main():
    t0 = time.time()
    parts = []

    # ---- FCS10: 91 / 140 / 183 ----
    tiles = fcs10_tiles()
    print(f'FCS10 中国区瓦片 {len(tiles)}', flush=True)
    with Pool(3, initializer=_winit) as p:
        for i, d in enumerate(p.imap_unordered(tile_task, tiles, chunksize=1)):
            if d is not None and len(d):
                parts.append(d)
            if (i + 1) % 20 == 0:
                n = sum(len(x) for x in parts)
                print(f'  FCS10 {i+1}/{len(tiles)} 候选 {n:,} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df['src'] = 'fcs10_2023_rare'
        df['src_conf'] = 0.85
        df['year'] = C.FCS10_YEAR
    print(f'FCS10 汇总 {len(df):,}  ' +
          str(df.class_new.value_counts().to_dict() if len(df) else {}), flush=True)

    # ---- GMW v3: 184 ----
    gparts = []
    gmw_tiles = sorted(glob.glob(os.path.join(GMW_DIR, '*.tif')))
    print(f'GMW 瓦片 {len(gmw_tiles)}', flush=True)
    if gmw_tiles:
        with Pool(3, initializer=_winit) as p:
            for d in p.imap_unordered(gmw_task, gmw_tiles, chunksize=1):
                if d is not None and len(d):
                    gparts.append(d)
        g = pd.concat(gparts, ignore_index=True) if gparts else pd.DataFrame()
        if len(g):
            g['src'] = 'gmw_v3_2020'
            g['src_conf'] = 0.9
            g['year'] = 2020
        print(f'GMW 汇总 {len(g):,}', flush=True)
        df = pd.concat([df, g], ignore_index=True) if len(df) else g

    if len(df) == 0:
        print('无候选，退出'); return

    # ---- 国界 + 生态过滤 ----
    df = df.drop_duplicates(['lon', 'lat', 'class_new']).reset_index(drop=True)
    keep = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    print(f'国界内 {keep.sum():,}/{len(df):,}', flush=True)
    df = df[keep].reset_index(drop=True)
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    print(f'生态剔除 {int(bad.sum()):,}', flush=True)
    df = df[~bad].reset_index(drop=True)
    df['tier'] = 'external'
    df['agree_n'] = -1
    df = df[['lon', 'lat', 'class_new', 'class_raw', 'src', 'src_conf', 'year',
             'tier', 'agree_n']]

    out = os.path.join(C.WORK, 'r8_rare_candidates.parquet')
    df.to_parquet(out, index=False)
    print('输出:', out, len(df), dict(df.class_new.value_counts()), flush=True)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
