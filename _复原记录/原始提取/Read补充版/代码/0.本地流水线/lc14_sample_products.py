# -*- coding: utf-8 -*-
"""
lc14_sample_products.py — 本地多产品采样（不依赖GEE），为样本提纯 v2 提供投票列
* 对 849 万样本点，从 5 个本地产品采样：中心值 + (10m产品)3×3邻域多样性
  A) ESA WorldCover 2021 (10m,4326; 3°瓦片在4个60°macrotile zip内, vsizip直读)
  B) ESRI LULC 2020  (10m, UTM 55瓦片; 图例实证: 7=built 5=crops 6=scrub 2=trees)
  C) 清华2017 10m     (10m, 4326, 31省tiff)
  D) CLCD 2020 30m    (Albers 全国tiff; 按批次开文件, 逐1°格窗口)
  E) CLCD 100m 2000-2024 稳定性(与2020同类年数) + CLCD2020(100m) 投票列
  F) 中国30米精细2020 (嵌套zip, GLC码)
* 用法: python lc14_sample_products.py <A|B|C|D|E|F>
* 资源: Pool(6) / 峰值内存 < 16GB
"""
import os, re, sys, json, math, time, logging, tempfile, zipfile, io, glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans
from multiprocessing import Pool
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_conf import CLASSES, LEVEL0

logging.basicConfig(filename='数据/本地处理/日志/lc14.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', encoding='utf-8')
BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v1.parquet')
N = len(BASE)
LON = BASE['lon'].to_numpy(); LAT = BASE['lat'].to_numpy()
ROWID = BASE.index.to_numpy().astype(np.int64)
CX = np.floor(LON).astype(np.int64); CY = np.floor(LAT).astype(np.int64)
CELL_KEY = CX * 1000 + CY
OUT_DIR = '数据/本地处理/样本底座/v2_parts'
os.makedirs(OUT_DIR, exist_ok=True)

E = 'E:/gisrsdata.org/还未整理/地理资源/遥感地物分类'
WC_DIR = E + '/ESA-WorldCover全球10m分辨率土地覆被'
ESRI_DIR = E + '/ESRI10m土地利用数据(未裁剪)'
TH_DIR = E + '/2017年10m土地利用数据_分省裁剪_清华大学数据集'
CN30_ZIP = E + '/中国30米精细地表覆盖2020-2010-2000/2020年30米地表覆盖.zip'
CLCD30_DIR = r'Z:/Mywork/论文/中国人口密度2000-2026/1.数据/2.1_CLCD土地覆盖数据/30米'
CLCD100_DIR = r'Z:/Mywork/论文/中国人口密度2000-2026/1.数据/2.1_CLCD土地覆盖数据/100米'

XW = {
 'WC':   {10:2, 20:3, 30:4, 40:1, 50:6, 60:7, 70:10, 80:9, 90:5, 95:5, 100:7},
 'ESRI': {0:-1, 1:9, 2:2, 3:4, 4:5, 5:1, 6:3, 7:6, 8:7, 9:10, 10:4, 11:-1},
 'TH':   {0:-1, 10:1, 20:2, 30:4, 40:3, 50:5, 60:9, 70:7, 80:6, 90:7, 100:10},
 'CLCD': {0:-1, 1:1, 2:2, 3:3, 4:4, 5:9, 6:10, 7:7, 8:6, 9:5},
 'CN30': {0:-1, 250:-1, 10:1, 11:1, 12:1, 20:1, 51:2, 52:2, 61:2, 62:2, 71:2, 72:2,
          81:2, 82:2, 91:2, 92:2, 120:3, 121:3, 122:3, 130:4, 140:7, 150:7, 152:7,
          153:7, 181:5, 182:5, 183:5, 184:5, 185:5, 186:5, 187:5, 190:6, 200:7,
          201:7, 202:7, 210:9, 220:10},
}
LUT = {k: np.full(256, -1, dtype=np.int8) for k in XW}
for k, m in XW.items():
    for a, b in m.items():
        if 0 <= a < 256:
            LUT[k][a] = b

# ---------- 公共 ----------
CELL_IDX = {}
def build_cell_idx():
    """cell_key -> 点索引数组（一次性预建）"""
    order = np.argsort(CELL_KEY, kind='stable')
    sk = CELL_KEY[order]
    b = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    return {int(sk[b[i]]): order[b[i]:b[i+1]] for i in range(len(b)-1)}

def cells_group(sel):
    """把点索引按 1°格分组: {dk: idx数组}"""
    dk = CELL_KEY[sel]
    order = np.argsort(dk, kind='stable')
    sk = dk[order]
    b = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    return {int(sk[b[i]]): sel[order[b[i]]:b[i+1]] for i in range(len(b)-1)}

def sample_l0(win, x0, y0, rows, cols, lut, need_div):
    """窗口内绝对像素坐标采样; 返回 raw(uint8,255=缺), div(3×3 l0多样性, 0=未知)"""
    rows = np.asarray(rows).astype(np.int64)
    cols = np.asarray(cols).astype(np.int64)
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
    L = lut[np.stack(nb, axis=1)]                 # n×9 int8
    Ls = np.sort(L, axis=1)
    d = (Ls[:, 1:] != Ls[:, :-1]).sum(axis=1) + 1
    d = d - (Ls[:, 0] == -1)                      # 扣掉未知组
    center_l = lut[raw]
    d[center_l < 0] = 0
    return raw, d.astype(np.uint8)

def win_4326(s, cx, cy, margin=1):
    """4326 栅格上 1°格 的像素窗口"""
    inv = ~s.transform
    c0, r0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
    c1, r1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
    c0 = max(0, c0 - margin); r0 = max(0, r0 - margin)
    c1 = min(s.width, c1 + margin + 1); r1 = min(s.height, r1 + margin + 1)
    return c0, r0, c1, r1

# ---------- A: WorldCover ----------
def wc_tile_path(lat3, lon3):
    ml = 'S30' if lat3 < 30 else 'N30'
    mo = 'E060' if lon3 < 120 else 'E120'
    la = ('S%02d' % -lat3) if lat3 < 0 else ('N%02d' % lat3)
    lo = ('W%03d' % -lon3) if lon3 < 0 else ('E%03d' % lon3)
    zipn = ('【公众号 Geo地理数据研究所】ESA_WorldCover_10m_2021_v200_'
            f'60deg_macrotile_{ml}{mo}.zip')
    inner = f'ESA_WorldCover_10m_2021_v200_{la}{lo}_Map.tif'
    return f'/vsizip/{WC_DIR}/{zipn}/{inner}'

def wc_task(latlon3):
    lat3, lon3 = latlon3
    p = wc_tile_path(lat3, lon3)
    out = []
    try:
        with rasterio.open(p) as s:
            for dk, idx in CELL_IDX.items():
                cx, cy = dk // 1000, dk % 1000
                if not (lon3 <= cx < lon3 + 3 and lat3 <= cy < lat3 + 3):
                    continue
                c0, r0, c1, r1 = win_4326(s, cx, cy)
                if c1 <= c0 or r1 <= r0:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                px, py = ~s.transform * (LON[idx], LAT[idx])
                raw, div = sample_l0(a, c0, r0, py.astype(int), px.astype(int),
                                     LUT['WC'], True)
                out.append(pd.DataFrame({'row_id': ROWID[idx], 'wc_raw': raw,
                                         'wc_div': div}))
        return out
    except Exception as ex:
        logging.warning(f'WC {lat3},{lon3}: {str(ex)[:120]}')
        return []

def wc_tasks():
    l3 = sorted({(int(cy) // 3 * 3, int(cx) // 3 * 3)
                 for cx, cy in zip(CX, CY)})
    return l3

# ---------- B: ESRI ----------
def esri_tasks():
    fs = [f for f in os.listdir(ESRI_DIR)
          if '20200101' in f and f.endswith('.tif')]
    keep = []
    for f in fs:
        try:
            with rasterio.open(os.path.join(ESRI_DIR, f)) as s:
                b = s.bounds
                tr = Transformer.from_crs(s.crs, 4326, always_xy=True)
                lons, lats = tr.transform([b.left, b.right], [b.bottom, b.top])
                if (max(lons) > 72 and min(lons) < 136 and
                        max(lats) > 17 and min(lats) < 55):
                    keep.append(f)
        except Exception as ex:
            logging.warning(f'ESRI meta {f}: {str(ex)[:80]}')
    return keep

def esri_task(fn):
    p = os.path.join(ESRI_DIR, fn)
    out = []
    try:
        with rasterio.open(p) as s:
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            tr2 = Transformer.from_crs(s.crs, 4326, always_xy=True)
            b = s.bounds
            lons, lats = tr2.transform([b.left, b.right], [b.bottom, b.top])
            sel = np.flatnonzero((LON >= min(lons) - 0.05) & (LON <= max(lons) + 0.05) &
                                 (LAT >= min(lats) - 0.05) & (LAT <= max(lats) + 0.05))
            if not len(sel):
                return []
            groups = cells_group(sel)
            for dk, idx in groups.items():
                cx, cy = dk // 1000, dk % 1000
                xs, ys = [], []
                for X in (cx, cx + 1):
                    for Y in (cy, cy + 1):
                        x, y = tr.transform(X, Y); xs.append(x); ys.append(y)
                c0f, r0f = ~s.transform * (min(xs), max(ys))
                c1f, r1f = ~s.transform * (max(xs), min(ys))
                c0 = max(0, int(c0f) - 1); r0 = max(0, int(r0f) - 1)
                c1 = min(s.width, int(c1f) + 2); r1 = min(s.height, int(r1f) + 2)
                if c1 <= c0 or r1 <= r0 or (c1 - c0) * (r1 - r0) > 20000**2:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                x, y = tr.transform(LON[idx], LAT[idx])
                rows, cols = riotrans.rowcol(s.transform, x, y,
                                             op=lambda v: np.floor(v).astype(int))
                raw, div = sample_l0(a, c0, r0, np.array(rows), np.array(cols),
                                     LUT['ESRI'], True)
                out.append(pd.DataFrame({'row_id': ROWID[idx], 'esri_raw': raw,
                                         'esri_div': div}))
        return out
    except Exception as ex:
        logging.warning(f'ESRI {fn}: {str(ex)[:120]}')
        return []

# ---------- C: 清华2017 ----------
def th_tasks():
    """省界多边形精确归属（bbox 外包矩形重叠会重复计数），返回 (tif路径, 省多边形)"""
    import geopandas as gpd
    bp = '数据/边界/china_100000_full.json'
    gdf = gpd.read_file(bp)
    tifs = sorted(glob.glob(TH_DIR + '/*.tif'))
    # 省名 -> 多边形（union 该省所有要素）
    from shapely.ops import unary_union
    prov = {}
    for _, r in gdf.iterrows():
        nm = r.get('name') or r.get('adcode')
        prov[nm] = r.geometry
    pairs = []
    for t in tifs:
        base = os.path.basename(t)
        nm = base.split('__')[0]
        if nm in prov:
            pairs.append((t, prov[nm]))
        else:
            logging.warning(f'TH 找不到省多边形: {nm}')
    return pairs

def th_task(args):
    """省界多边形精确归属：点只属于含它的省"""
    p, poly = args
    out = []
    try:
        from shapely import contains_xy
        minx, miny, maxx, maxy = poly.bounds
        cand = np.flatnonzero((LON >= minx) & (LON <= maxx) &
                              (LAT >= miny) & (LAT <= maxy))
        if not len(cand):
            return []
        inside = contains_xy(poly, LON[cand], LAT[cand])
        sel = cand[inside]
        if not len(sel):
            return []
        with rasterio.open(p) as s:
            groups = cells_group(sel)
            for dk, idx in groups.items():
                cx, cy = dk // 1000, dk % 1000
                c0, r0, c1, r1 = win_4326(s, cx, cy)
                if c1 <= c0 or r1 <= r0:
                    continue
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                px, py = ~s.transform * (LON[idx], LAT[idx])
                raw, div = sample_l0(a, c0, r0, py.astype(int), px.astype(int),
                                     LUT['TH'], True)
                out.append(pd.DataFrame({'row_id': ROWID[idx], 'th_raw': raw,
                                         'th_div': div}))
        return out
    except Exception as ex:
        logging.warning(f'TH {os.path.basename(p)}: {str(ex)[:120]}')
        return []

# ---------- D: CLCD2020 30m ----------
def clcd30_batches(n_per=40):
    dks = sorted(set(CELL_KEY.tolist()))
    return [dks[i:i + n_per] for i in range(0, len(dks), n_per)]

def clcd30_task(dks):
    p = os.path.join(CLCD30_DIR, 'CLCD_v01_2020_albert.tif')
    out = []
    try:
        with rasterio.open(p) as s:
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            for dk in dks:
                cx, cy = dk // 1000, dk % 1000
                idx = CELL_IDX[dk]
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
                x, y = tr.transform(LON[idx], LAT[idx])
                rows, cols = riotrans.rowcol(s.transform, x, y,
                                             op=lambda v: np.floor(v).astype(int))
                raw, _ = sample_l0(a, c0, r0, np.array(rows), np.array(cols),
                                   LUT['CLCD'], False)
                out.append(pd.DataFrame({'row_id': ROWID[idx], 'clcd_raw': raw}))
        return out
    except Exception as ex:
        logging.warning(f'CLCD30 {dks[0]}: {str(ex)[:120]}')
        return []

# ---------- E: CLCD 100m 稳定性 ----------
def run_clcd100():
    years = [y for y in range(2000, 2025)]
    p20 = os.path.join(CLCD100_DIR, 'CLCD_v01_2020_albert_100m.tif')
    with rasterio.open(p20) as s:
        tr = Transformer.from_crs(4326, s.crs, always_xy=True)
        X, Y = tr.transform(LON, LAT)
        rows, cols = riotrans.rowcol(s.transform, X, Y,
                                     op=lambda v: np.floor(v).astype(int))
        cols = np.array(cols); rows = np.array(rows)
        ok = (rows >= 0) & (rows < s.height) & (cols >= 0) & (cols < s.width)
        a20 = s.read(1)
    sel = np.flatnonzero(ok)
    raw20 = a20[rows[sel], cols[sel]]
    ref = LUT['CLCD'][raw20]
    stab = np.zeros(len(sel), dtype=np.int16)
    for y in years:
        if y == 2020:
            continue
        with rasterio.open(os.path.join(CLCD100_DIR,
                                        f'CLCD_v01_{y}_albert_100m.tif')) as s:
            a = s.read(1)
        stab += (LUT['CLCD'][a][rows[sel], cols[sel]] == ref).astype(np.int16)
        del a
        logging.info(f'CLCD100 {y} done')
    df = pd.DataFrame({'row_id': ROWID[sel], 'clcd100_raw': raw20,
                       'stab_years': stab, 'stab_den': len(years)})
    df.to_parquet(f'{OUT_DIR}/E_clcd100.parquet', index=False)
    print('E done:', len(df), flush=True)

# ---------- F: 中国30米精细 2020 ----------
def cn30_tasks():
    z = zipfile.ZipFile(CN30_ZIP)
    return [n for n in z.namelist() if n.endswith('.zip')]

def cn30_task(inner_name):
    out = []
    tmp = None
    try:
        z = zipfile.ZipFile(CN30_ZIP)
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
            sel = np.flatnonzero((LON >= min(lons) - 0.05) & (LON <= max(lons) + 0.05) &
                                 (LAT >= min(lats) - 0.05) & (LAT <= max(lats) + 0.05))
            if not len(sel):
                return []
            tr = Transformer.from_crs(4326, s.crs, always_xy=True)
            xs, ys = tr.transform(LON[sel], LAT[sel])
            rows, cols = riotrans.rowcol(s.transform, xs, ys,
                                         op=lambda v: np.floor(v).astype(int))
            cols = np.array(cols); rows = np.array(rows)
            ok = (rows >= 0) & (rows < s.height) & (cols >= 0) & (cols < s.width)
            sel = sel[ok]; rows = rows[ok]; cols = cols[ok]
            if not len(sel):
                return []
            c0 = max(0, cols.min() - 1); c1 = min(s.width, cols.max() + 2)
            r0 = max(0, rows.min() - 1); r1 = min(s.height, rows.max() + 2)
            if (c1 - c0) * (r1 - r0) > 24000**2:
                # 分 1° 格窗口读
                for dk, idx in cells_group(sel).items():
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
                    x, y = tr.transform(LON[idx], LAT[idx])
                    r2, c2 = riotrans.rowcol(s.transform, x, y,
                                             op=lambda v: np.floor(v).astype(int))
                    raw, _ = sample_l0(a, cc0, rr0, np.array(r2), np.array(c2),
                                       LUT['CN30'], False)
                    out.append(pd.DataFrame({'row_id': ROWID[idx], 'cn30_raw': raw}))
            else:
                a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
                raw, _ = sample_l0(a, c0, r0, rows, cols, LUT['CN30'], False)
                out.append(pd.DataFrame({'row_id': ROWID[sel], 'cn30_raw': raw}))
        return out
    except Exception as ex:
        logging.warning(f'CN30 {inner_name}: {str(ex)[:120]}')
        return []
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def run_product(key, tasks, fn):
    t0 = time.time()
    results = []
    with Pool(6, initializer=_winit) as p:
        for i, r in enumerate(p.imap_unordered(fn, tasks, chunksize=1)):
            if r:
                results.extend(r)
            if (i + 1) % 10 == 0:
                print(f'  {key} {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)', flush=True)
    if results:
        df = pd.concat(results, ignore_index=True)
        df.to_parquet(f'{OUT_DIR}/{key}.parquet', index=False)
        print(f'[{key}] 完成: {len(df)} 行, {time.time()-t0:.0f}s')
    else:
        print(f'[{key}] 无结果')

def _winit():
    """worker 初始化：重建点格索引（Windows spawn 下模块级变量已随 import 就绪）"""
    global CELL_IDX
    CELL_IDX = build_cell_idx()

if __name__ == '__main__':
    key = sys.argv[1]
    CELL_IDX = build_cell_idx()
    if key == 'A':
        run_product('A_wc21', wc_tasks(), wc_task)
    elif key == 'B':
        run_product('B_esri20', esri_tasks(), esri_task)
    elif key == 'C':
        run_product('C_th17', th_tasks(), th_task)
    elif key == 'D':
        run_product('D_clcd20', clcd30_batches(), clcd30_task)
    elif key == 'E':
        run_clcd100()
    elif key == 'F':
        run_product('F_cn30', cn30_tasks(), cn30_task)

