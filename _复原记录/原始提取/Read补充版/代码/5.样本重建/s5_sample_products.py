# -*- coding: utf-8 -*-
"""
s5_sample_products.py — 任意点集的 5 本地产品实测采样（lc14 同款窗口机制，参数化版）
* 用途: r1 外部点/topup 点/底座漏点 的投票列生产（WC21/ESRI20/TH17/CLCD30/CN30）
* 接口: sample_products(pts_df) -> {'wc_l0','esri_l0','th_l0','clcd_l0','cn30_l0': int8(-1..10),
          'border': uint8}   （border = 任一 10m 票源 3×3 level0 多样性 ≥3）
* Windows spawn 安全: 点数组经 initargs 传入 worker，单例全局。
"""
import os, sys, glob, io, zipfile, tempfile, logging, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from multiprocessing import Pool
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C

# ---------------- 点集全局（worker 经 initargs 重建） ----------------
_G = {}

def _build(lon, lat):
    cx = np.floor(lon).astype(np.int64); cy = np.floor(lat).astype(np.int64)
    ck = cx * 1000 + cy
    order = np.argsort(ck, kind='stable')
    sk = ck[order]
    b = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    cellidx = {int(sk[b[i]]): order[b[i]:b[i + 1]] for i in range(len(b) - 1)}
    return {'lon': lon, 'lat': lat, 'cx': cx, 'cy': cy, 'ck': ck, 'cellidx': cellidx}

def _winit(lon, lat):
    global _G
    _G = _build(lon, lat)

def _cells_group(sel):
    ck = _G['ck'][sel]
    order = np.argsort(ck, kind='stable')
    sk = ck[order]
    b = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    return {int(sk[b[i]]): sel[order[b[i]]:b[i + 1]] for i in range(len(b) - 1)}

def _win_4326(s, cx, cy, margin=1):
    inv = ~s.transform
    c0, r0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
    c1, r1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
    c0 = max(0, c0 - margin); r0 = max(0, r0 - margin)
    c1 = min(s.width, c1 + margin + 1); r1 = min(s.height, r1 + margin + 1)
    return c0, r0, c1, r1

def _sample_l0(win, x0, y0, rows, cols, lut, need_div):
    win = np.asarray(win)
    if win.dtype != np.uint8:            # TH 等产品为 float32（NaN/nodata → 0 → LUT=-1）
        win = win.astype(np.uint8, casting='unsafe')
    rows = np.asarray(rows).astype(np.int64); cols = np.asarray(cols).astype(np.int64)
    rr = rows - y0; cc = cols - x0
    h, w = win.shape
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    raw = np.full(len(rr), 255, dtype=np.uint8)
    raw[ok] = win[rr[ok], cc[ok]]
    if not need_div:
        return raw, None
    nb = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r2 = np.clip(rr + dr, 0, h - 1); c2 = np.clip(cc + dc, 0, w - 1)
            nb.append(win[r2, c2])
    L = lut[np.stack(nb, axis=1)]
    Ls = np.sort(L, axis=1)
    d = (Ls[:, 1:] != Ls[:, :-1]).sum(axis=1) + 1
    d = d - (Ls[:, 0] == -1)
    center_l = lut[raw]
    d[center_l < 0] = 0
    return raw, d.astype(np.uint8)

# ---------------- 各产品任务 ----------------
def wc_tile_path(lat3, lon3):
    ml = 'S30' if lat3 < 30 else 'N30'
    mo = 'E060' if lon3 < 120 else 'E120'
    la = ('S%02d' % -lat3) if lat3 < 0 else ('N%02d' % lat3)
    lo = ('W%03d' % -lon3) if lon3 < 0 else ('E%03d' % lon3)
    zipn = ('【公众号 Geo地理数据研究所】ESA_WorldCover_10m_2021_v200_'
            f'60deg_macrotile_{ml}{mo}.zip')
    inner = f'ESA_WorldCover_10m_2021_v200_{la}{lo}_Map.tif'
    return f'/vsizip/{C.WC_DIR}/{zipn}/{inner}'

def wc_task(latlon3):
    lat3, lon3 = latlon3
    out = []
    try:
        with rasterio.open(wc_tile_path(lat3, lon3)) as s:
            for dk, idx in _G['cellidx'].items():
                cx, cy = dk // 1000, dk % 1000
                if not (lon3 <= cx < lon3 + 3 and lat3 <= cy < lat3 + 3):
                    continue
                c0, r0, c1, r1 = _win_4326(s, cx, cy)
                if c1 <= c0 or r1 <= r0:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                px, py = ~s.transform * (_G['lon'][idx], _G['lat'][idx])
                raw, div = _sample_l0(a, c0, r0, py.astype(int), px.astype(int),
                                      C.PROD_LUT['WC'], True)
                out.append((idx, raw, div))
    except Exception as e:
        logging.warning(f'WC {lat3},{lon3}: {str(e)[:100]}')
    return out

def wc_tasks():
    return sorted({(int(cy) // 3 * 3, int(cx) // 3 * 3)
                   for cx, cy in zip(_G['cx'], _G['cy'])})

def _utm_tile_cells(fn, dirpath):
    """返回 (tif路径, 命中的点索引)"""
    p = os.path.join(dirpath, fn)
    try:
        with rasterio.open(p) as s:
            tr = Transformer.from_crs(s.crs, 4326, always_xy=True)
            b = s.bounds
            lons, lats = tr.transform([b.left, b.right], [b.bottom, b.top])
            sel = np.flatnonzero((_G['lon'] >= min(lons) - 0.05) & (_G['lon'] <= max(lons) + 0.05) &
                                 (_G['lat'] >= min(lats) - 0.05) & (_G['lat'] <= max(lats) + 0.05))
            return (p, sel) if len(sel) else None
    except Exception:
        return None

def esri_th_task(args):
    """通用 UTM/4326 瓦片任务: args=(path, sel, key, need_div, is4326)"""
    p, sel, key, need_div, is4326 = args
    out = []
    try:
        with rasterio.open(p) as s:
            if is4326:
                for dk, idx in _cells_group(sel).items():
                    cx, cy = dk // 1000, dk % 1000
                    c0, r0, c1, r1 = _win_4326(s, cx, cy)
                    if c1 <= c0 or r1 <= r0:
                        continue
                    a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                    px, py = ~s.transform * (_G['lon'][idx], _G['lat'][idx])
                    raw, div = _sample_l0(a, c0, r0, py.astype(int), px.astype(int),
                                          C.PROD_LUT[key], need_div)
                    out.append((idx, raw, div))
            else:
                tr = Transformer.from_crs(4326, s.crs, always_xy=True)
                for dk, idx in _cells_group(sel).items():
                    cx, cy = dk // 1000, dk % 1000
                    xs, ys = [], []
                    for X in (cx, cx + 1):
                        for Y in (cy, cy + 1):
                            x, y = tr.transform(X, Y); xs.append(x); ys.append(y)
                    c0f, r0f = ~s.transform * (min(xs), max(ys))
                    c1f, r1f = ~s.transform * (max(xs), min(ys))
                    c0 = max(0, int(c0f) - 1); r0 = max(0, int(r0f) - 1)
                    c1 = min(s.width, int(c1f) + 2); r1 = min(s.height, int(r1f) + 2)
                    if c1 <= c0 or r1 <= r0 or (c1 - c0) * (r1 - r0) > 25000 * 25000:
                        continue
                    a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                    x, y = tr.transform(_G['lon'][idx], _G['lat'][idx])
                    rows, cols = riotrans.rowcol(s.transform, x, y,
                                                 op=lambda v: np.floor(v).astype(int))
                    raw, div = _sample_l0(a, c0, r0, np.array(rows), np.array(cols),
                                          C.PROD_LUT[key], need_div)
                    out.append((idx, raw, div))
    except Exception as e:
        logging.warning(f'{key} {os.path.basename(p)}: {str(e)[:100]}')
    return out

def clcd30_task(dks):
    out = []
    p = os.path.join(C.CLCD30_DIR, 'CLCD_v01_2020_albert.tif')
    try:
        with rasterio.open(p) as s:
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            for dk in dks:
                cx, cy = dk // 1000, dk % 1000
                idx = _G['cellidx'][dk]
                xs, ys = [], []
                for X in (cx, cx + 1):
                    for Y in (cy, cy + 1):
                        x, y = tr.transform(X, Y); xs.append(x); ys.append(y)
                c0f, r0f = ~s.transform * (min(xs), max(ys))
                c1f, r1f = ~s.transform * (max(xs), min(ys))
                c0 = max(0, int(c0f) - 1); r0 = max(0, int(r0f) - 1)
                c1 = min(s.width, int(c1f) + 2); r1 = min(s.height, int(r1f) + 2)
                if c1 <= c0 or r1 <= r0:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                x, y = tr.transform(_G['lon'][idx], _G['lat'][idx])
                rows, cols = riotrans.rowcol(s.transform, x, y,
                                             op=lambda v: np.floor(v).astype(int))
                raw, _ = _sample_l0(a, c0, r0, np.array(rows), np.array(cols),
                                    C.PROD_LUT['CLCD'], False)
                out.append((idx, raw, None))
    except Exception as e:
        logging.warning(f'CLCD30 {dks[0]}: {str(e)[:100]}')
    return out

def cn30_task(inner_name):
    out = []
    tmp = None
    try:
        z = zipfile.ZipFile(C.CN30_ZIP)
        inner = zipfile.ZipFile(io.BytesIO(z.read(inner_name)))
        tifn = [n for n in inner.namelist() if n.lower().endswith('.tif')][0]
        data = inner.read(tifn)
        tmp = os.path.join(tempfile.gettempdir(), f'cn30_{os.getpid()}_{abs(hash(inner_name))}.tif')
        with open(tmp, 'wb') as f:
            f.write(data)
        with rasterio.open(tmp) as s:
            tr2 = Transformer.from_crs(s.crs, 4326, always_xy=True)
            b = s.bounds
            lons, lats = tr2.transform([b.left, b.right], [b.bottom, b.top])
            sel = np.flatnonzero((_G['lon'] >= min(lons) - 0.05) & (_G['lon'] <= max(lons) + 0.05) &
                                 (_G['lat'] >= min(lats) - 0.05) & (_G['lat'] <= max(lats) + 0.05))
            if not len(sel):
                return []
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            xs, ys = tr.transform(_G['lon'][sel], _G['lat'][sel])
            rows, cols = riotrans.rowcol(s.transform, xs, ys,
                                         op=lambda v: np.floor(v).astype(int))
            cols = np.array(cols); rows = np.array(rows)
            ok = (rows >= 0) & (rows < s.height) & (cols >= 0) & (cols < s.width)
            sel = sel[ok]; rows = rows[ok]; cols = cols[ok]
            if not len(sel):
                return []
            c0 = max(0, cols.min() - 1); c1 = min(s.width, cols.max() + 2)
            r0 = max(0, rows.min() - 1); r1 = min(s.height, rows.max() + 2)
            if (c1 - c0) * (r1 - r0) > 24000 * 24000:
                for dk, idx in _cells_group(sel).items():
                    cx, cy = dk // 1000, dk % 1000
                    xs2, ys2 = [], []
                    for X in (cx, cx + 1):
                        for Y in (cy, cy + 1):
                            x, y = tr.transform(X, Y); xs2.append(x); ys2.append(y)
                    c0f, r0f = ~s.transform * (min(xs2), max(ys2))
                    c1f, r1f = ~s.transform * (max(xs2), min(ys2))
                    cc0 = max(0, int(c0f) - 1); rr0 = max(0, int(r0f) - 1)
                    cc1 = min(s.width, int(c1f) + 2); rr1 = min(s.height, int(r1f) + 2)
                    if cc1 <= cc0 or rr1 <= rr0:
                        continue
                    a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                    x, y = tr.transform(_G['lon'][idx], _G['lat'][idx])
                    r2, c2 = riotrans.rowcol(s.transform, x, y,
                                             op=lambda v: np.floor(v).astype(int))
                    raw, _ = _sample_l0(a, cc0, rr0, np.array(r2), np.array(c2),
                                        C.PROD_LUT['CN30'], False)
                    out.append((idx, raw, None))
            else:
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                raw, _ = _sample_l0(a, c0, r0, rows, cols, C.PROD_LUT['CN30'], False)
                out.append((sel, raw, None))
        return out
    except Exception as e:
        logging.warning(f'CN30 {inner_name}: {str(e)[:100]}')
        return []
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def _run(key, tasks, fn, n_arr):
    t0 = time.time()
    acc = np.full(n_arr, 255, dtype=np.uint8)
    div = np.zeros(n_arr, dtype=np.uint8)
    with Pool(5, initializer=_winit, initargs=(_G['lon'], _G['lat'])) as p:
        for i, res in enumerate(p.imap_unordered(fn, tasks, chunksize=1)):
            if res:
                for idx, raw, dv in res:
                    acc[idx] = raw
                    if dv is not None:
                        div[idx] = np.maximum(div[idx], dv)
            if (i + 1) % 20 == 0:
                print(f'    {key} {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    return acc, div

def sample_products(pts, products=None):
    """pts: DataFrame(lon,lat) → dict 投票列；products=None 全部五个，否则子集如 ['WC','CN30']"""
    global _G
    lon = pts['lon'].to_numpy(float); lat = pts['lat'].to_numpy(float)
    _G = _build(lon, lat)
    n = len(pts)
    P = set(products) if products else {'WC', 'ESRI', 'TH', 'CLCD', 'CN30'}
    res = {}
    if 'WC' in P:
        print('  [WC21] …', flush=True)
        raw, div = _run('WC', wc_tasks(), wc_task, n)
        res['wc_l0'] = C.PROD_LUT['WC'][raw]
        res['_wc_div'] = div
    if 'ESRI' in P:
        print('  [ESRI20] …', flush=True)
        fs = [f for f in os.listdir(C.ESRI_DIR) if '20200101' in f and f.endswith('.tif')]
        tasks = [t for t in (_utm_tile_cells(f, C.ESRI_DIR) for f in fs) if t]
        tasks = [(p, sel, 'ESRI', True, False) for p, sel in tasks]
        raw, div = _run('ESRI', tasks, esri_th_task, n)
        res['esri_l0'] = C.PROD_LUT['ESRI'][raw]
        res['_esri_div'] = div
    if 'TH' in P:
        print('  [TH17] …', flush=True)
        # 分省瓦片 bbox 相互重叠，点须按省界唯一归属（后写覆盖会拿邻省 nodata 抹掉正确值）
        import s1_geom as G1
        prov = G1.province_of(lon, lat)
        tifs = sorted(glob.glob(os.path.join(C.TH_DIR, '*.tif')))
        tasks = []
        for p in tifs:
            nm = os.path.basename(p).split('__')[0]
            sel = np.flatnonzero(prov == nm)
            if len(sel):
                tasks.append((p, sel, 'TH', True, True))
        print(f'    TH 命中任务数: {len(tasks)}（点 {n}）', flush=True)
        raw, div = _run('TH', tasks, esri_th_task, n)
        res['th_l0'] = C.PROD_LUT['TH'][raw]
        res['_th_div'] = div
    if 'CLCD' in P:
        print('  [CLCD30] …', flush=True)
        dks = sorted(set(_G['ck'].tolist()))
        batches = [dks[i:i + 40] for i in range(0, len(dks), 40)]
        raw, _ = _run('CLCD', batches, clcd30_task, n)
        res['clcd_l0'] = C.PROD_LUT['CLCD'][raw]
    if 'CN30' in P:
        print('  [CN30] …', flush=True)
        z = zipfile.ZipFile(C.CN30_ZIP)
        inners = [nm for nm in z.namelist() if nm.endswith('.zip')]
        raw, _ = _run('CN30', inners, cn30_task, n)
        res['cn30_l0'] = C.PROD_LUT['CN30'][raw]
    if {'WC', 'ESRI', 'TH'} <= P:
        border = ((np.stack([res['_wc_div'], res['_esri_div'], res['_th_div']]) >= 3)
                  .any(axis=0)).astype(np.uint8)
        res['border'] = border
    for k in ['_wc_div', '_esri_div', '_th_div']:
        res.pop(k, None)
    return res

if __name__ == '__main__':
    # 冒烟：北京城区 5 点
    pts = pd.DataFrame({'lon': [116.40, 116.35, 116.80, 117.20, 115.90],
                        'lat': [39.90, 40.10, 39.60, 40.80, 40.30]})
    import time
    t0 = time.time()
    r = sample_products(pts)
    print({k: v for k, v in r.items()}, f'{time.time()-t0:.0f}s')

