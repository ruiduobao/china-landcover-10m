# -*- coding: utf-8 -*-
"""
r8b_rare_boundary.py — P3.3 稀有类边界样本 + CW 红树林补充（纯本地栅格）
* 背景: r8_rare_topup（3×3 纯净核心样本）对 91/140/184 仍不足（核心样本天然少），
  评审阶段F 明确要求补「中心像元纯、3×3 混合」的边界样本，权重 0.5-0.8。
* 来源:
    FCS10 2023 中心=raw{92→91, 140→140} 但 3×3 非纯净 → 边界样本
    GMW v3 2020 中心=1 但 3×3 非纯净 → 184 边界样本
    ChinaWetlands CW_2020 值=2（红树林）核心+边界 → 184
  （183 核心已 8,099 超目标，不再补边界）
* 采样: 每 0.05° 格每类 ≤1（固定种子），国界 + 生态过滤，每类总量封顶。
* 输出: 数据/本地处理/样本重建/r8b_rare_boundary.parquet
   （sample_type=boundary/core，weight=0.6/1.0，供 P3.3 合并进年度训练集）
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
import r8_rare_topup as R

B_TARGET = {92: 91, 140: 140}          # 仅 91/140 需边界补样
B_LUT = np.zeros(256, dtype=bool)
for _k in B_TARGET:
    B_LUT[_k] = True
CAP = {91: 3000, 140: 3500, 184: 2500, 183: 3000}   # 合并去重前每类上限
CW = os.path.join(C.RARE_DIR, 'ChinaWetlands', 'annual', 'CW_2020.tif')
CW_MANGROVE = 2                        # 说明.md 实测图例: 1潮滩/2红树林/3盐沼

_polys = None


def _winit():
    global _polys
    _polys = G.china_polys()


def boundary_mask(a):
    """中心像元合法但 3×3 非纯净"""
    c = a[1:-1, 1:-1]
    p = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            p &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return ~p


def thin1(df):
    """每 0.05° 格每类 ≤1（固定种子）"""
    if len(df) == 0:
        return df
    fine = (np.floor(df.lon.to_numpy() / R.FINE).astype(np.int64) * 100000 +
            np.floor(df.lat.to_numpy() / R.FINE).astype(np.int64))
    df = df.assign(_f=fine).sample(frac=1.0, random_state=C.SEED)
    return df.groupby(['_f', 'class_new'], group_keys=False).head(1).drop(columns='_f')


def fcs10_boundary_task(fp):
    m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
    if not m:
        return None
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
                    rows, cols = np.where(boundary_mask(a) & B_LUT[center])
                    if len(rows) == 0:
                        continue
                    raw_c = center[rows, cols]
                    cls_c = np.array([B_TARGET[int(v)] for v in raw_c], dtype=np.int64)
                    X, Y = riotrans.xy(s.transform, rows + rr0 + 0.5, cols + cc0 + 0.5)
                    out.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                                             'class_new': cls_c, 'class_raw': raw_c}))
        if out:
            return thin1(pd.concat(out, ignore_index=True))
        return None
    except Exception as e:
        print('FCS10b ERR', os.path.basename(fp), str(e)[:90], flush=True)
        return None


def gmw_boundary_task(fp):
    try:
        with rasterio.open(fp) as s:
            a = s.read(1)
            if a.ndim == 3:
                a = a[0]
            center = a[1:-1, 1:-1]
            rows, cols = np.where(boundary_mask(a) & (center == 1))
            if len(rows) == 0:
                return None
            X, Y = riotrans.xy(s.transform, rows + 1 + 0.5, cols + 1 + 0.5)
            return thin1(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                                       'class_new': 184, 'class_raw': 1}))
    except Exception as e:
        print('GMWb ERR', os.path.basename(fp), str(e)[:90], flush=True)
        return None


def cw_task(args):
    """ChinaWetlands 1° 窗口；value=2 红树林 → 184（核心+边界）"""
    cx, cy = args
    try:
        with rasterio.open(CW) as s:
            inv = ~s.transform
            cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
            cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
            cc0 = max(0, cc0 - 1); rr0 = max(0, rr0 - 1)
            cc1 = min(s.width, cc1 + 2); rr1 = min(s.height, rr1 + 2)
            if cc1 <= cc0 or rr1 <= rr0:
                return None
            a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
            if a.ndim == 3:
                a = a[0]
            center = a[1:-1, 1:-1]
            frames = []
            for mask, typ in ((R.pure_mask(a) & (center == CW_MANGROVE), 'core'),
                              (boundary_mask(a) & (center == CW_MANGROVE), 'boundary')):
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue
                X, Y = riotrans.xy(s.transform, rows + rr0 + 1 + 0.5, cols + cc0 + 1 + 0.5)
                frames.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y),
                                            'class_new': 184, 'class_raw': CW_MANGROVE,
                                            'sample_type': typ}))
            if not frames:
                return None
            return thin1(pd.concat(frames, ignore_index=True))
    except Exception as e:
        print('CW ERR', cx, cy, str(e)[:90], flush=True)
        return None


def cap_per_class(df):
    out = []
    for c, g in df.groupby('class_new'):
        n = CAP.get(int(c), 3000)
        if len(g) > n:
            g = g.sample(n=n, random_state=C.SEED)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    parts = []

    tiles = R.fcs10_tiles()
    print(f'FCS10 边界扫描 {len(tiles)} 瓦片（目标 91/140）', flush=True)
    with Pool(3, initializer=_winit) as p:
        for i, d in enumerate(p.imap_unordered(fcs10_boundary_task, tiles, chunksize=1)):
            if d is not None and len(d):
                parts.append(d)
            if (i + 1) % 20 == 0:
                print(f'  FCS10b {i+1}/{len(tiles)} 候选 {sum(len(x) for x in parts):,} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df['src'] = 'fcs10_2023_rare_boundary'; df['src_conf'] = 0.75
        df['year'] = C.FCS10_YEAR; df['sample_type'] = 'boundary'
    else:
        df = pd.DataFrame()
    print(f'FCS10 边界汇总 {len(df):,} ' +
          str(df.class_new.value_counts().to_dict() if len(df) else {}), flush=True)

    gm = sorted(glob.glob(os.path.join(R.GMW_DIR, '*.tif')))
    print(f'GMW 边界扫描 {len(gm)}', flush=True)
    gparts = []
    with Pool(3, initializer=_winit) as p:
        for d in p.imap_unordered(gmw_boundary_task, gm, chunksize=1):
            if d is not None and len(d):
                gparts.append(d)
    if gparts:
        g = pd.concat(gparts, ignore_index=True)
        g['src'] = 'gmw_v3_2020_boundary'; g['src_conf'] = 0.85
        g['year'] = 2020; g['sample_type'] = 'boundary'
        df = pd.concat([df, g], ignore_index=True)
    print(f'GMW 边界汇总 {sum(len(x) for x in gparts):,}', flush=True)

    wl = [(cx, cy) for cy in range(18, 42) for cx in range(108, 125)]
    print(f'CW 窗口 {len(wl)}', flush=True)
    cparts = []
    with Pool(3, initializer=_winit) as p:
        for d in p.imap_unordered(cw_task, wl, chunksize=4):
            if d is not None and len(d):
                cparts.append(d)
    if cparts:
        c = pd.concat(cparts, ignore_index=True)
        c['src'] = 'cw_2020_mangrove'; c['src_conf'] = 0.9; c['year'] = 2020
        df = pd.concat([df, c], ignore_index=True)
    print(f'CW 汇总 {sum(len(x) for x in cparts):,}', flush=True)

    if len(df) == 0:
        print('无候选'); return
    df = df.drop_duplicates(['lon', 'lat', 'class_new']).reset_index(drop=True)
    keep = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[keep].reset_index(drop=True)
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    df = cap_per_class(df)
    df['tier'] = 'external'; df['agree_n'] = -1
    df['weight'] = np.where(df.sample_type == 'core', 1.0, 0.6)
    df = df[['lon', 'lat', 'class_new', 'class_raw', 'src', 'src_conf', 'year',
             'sample_type', 'weight', 'tier', 'agree_n']]
    out = os.path.join(C.WORK, 'r8b_rare_boundary.parquet')
    df.to_parquet(out, index=False)
    print('输出:', out, len(df), flush=True)
    print('按类:', dict(df.class_new.value_counts()), flush=True)
    print('按类型:', dict(df.sample_type.value_counts()), flush=True)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
