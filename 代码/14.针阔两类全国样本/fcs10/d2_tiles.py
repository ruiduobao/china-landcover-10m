# -*- coding: utf-8 -*-
"""d2_tiles.py — 从本地 5° FCS10-2023 源块整数像素直拷成 2°×2° 全国瓦片
* 输入 / E:/data/全球土地覆盖数据/数据/GLC_FCS10 与 E:/data/河南农作物提取_2025.04/GLC_FCS10m
         及 F:/lc_work/prod5p_2023/xcomp/fcs10_unzip（E130 救援块）
* 规则源 / 现有 76 瓦反推的 GEE 生产网格（res=8.983152841195215e-05，
           col0=floor(lon/res), row0=ceil(lat/res)，W=ceil(x1/res)-floor(x0/res)）
* 门槛 / 零重采样（源块与目标网格共用全局像素格，纯整数窗口直拷）
* 输出 / Z:/nb_national_2026/fcs10_dl/tiles/T{ix}{iy}_fcs10.tif（EPSG:4326, uint8, nodata=0）
         Z:/nb_national_2026/fcs10_dl/state/tile_build.json
* 用法 / python d2_tiles.py [--tiles T1505 T2002] [--all-plan] [--workers 3]
"""
import os, sys, json, math, glob, re, time, argparse, shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
from rasterio.windows import Window

RES = 8.983152841195215e-05
OUT = r'Z:/nb_national_2026/fcs10_dl/tiles'
STATE = r'Z:/nb_national_2026/fcs10_dl/state/tile_build.json'
F_WORK = r'F:/lc_work/nb_national_2026/fcs10_dl'

SRC_DIRS = [
    r'E:/data/河南农作物提取_2025.04/GLC_FCS10m',
    r'E:/data/全球土地覆盖数据/数据/GLC_FCS10',
    r'F:/lc_work/prod5p_2023/xcomp/fcs10_unzip',
]
PAT = re.compile(r'GLC_FCS10_2023_([EW])(\d{3})([NS])(\d{2})\.tif$')


def build_index():
    """(lon0,lat0) -> path，优先最大文件（避免半截副本）"""
    idx = {}
    for root in SRC_DIRS:
        for p in glob.glob(root + '/**/GLC_FCS10_2023_*.tif', recursive=True):
            m = PAT.search(os.path.basename(p))
            if not m:
                continue
            ew, d3, ns, d2 = m.groups()
            lon = int(d3) * (1 if ew == 'E' else -1)
            lat = int(d2) * (1 if ns == 'N' else -1)
            k = (lon, lat)
            sz = os.path.getsize(p)
            if k not in idx or sz > idx[k][0]:
                idx[k] = (sz, p)
    return {k: v[1] for k, v in idx.items()}


def grid(t):
    """tile id -> (col0,row0,W,H,transform_params)"""
    ix, iy = int(t[1:3]), int(t[3:5])
    x0, y0 = 73.0 + 2 * ix, 18.0 + 2 * iy
    x1, y1 = x0 + 2, y0 + 2
    c0, c1 = math.floor(x0 / RES), math.ceil(x1 / RES)
    r0, r1 = math.ceil(y1 / RES), math.floor(y0 / RES)
    return c0, r0, c1 - c0, r0 - r1


def do_tile(args):
    t, idx, force = args
    out = os.path.join(OUT, f'{t}_fcs10.tif')
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000 and not force:
        return (t, 'skip', os.path.getsize(out))
    c0, r0, W, H = grid(t)
    ix, iy = int(t[1:3]), int(t[3:5])
    x0, y0 = 73.0 + 2 * ix, 18.0 + 2 * iy
    x1, y1 = x0 + 2, y0 + 2
    # 通用重叠取块：块 lon∈[blo,blo+5]，lat∈[bla-5,bla]（**块名纬度是北边界**）
    # 容差 2e-4° 吸收源块右/下边界比名义值短 1e-4° 的切片
    TOL = 2e-4
    blocks = []
    for (blo, bla), p in idx.items():
        if blo - TOL <= x1 and blo + 5 + TOL >= x0 and (bla - 5) - TOL <= y1 and bla + TOL >= y0:
            blocks.append(((blo, bla), p))
    if not blocks:
        return (t, 'no_block', 0)

    dst = np.zeros((H, W), dtype=np.uint8)
    covered = 0
    for (blo, bla), bpath in sorted(blocks):
        # 源块左上角在全局格上的绝对列/行索引
        bcol = math.floor(blo / RES)
        brow = math.ceil(bla / RES)      # 北边界 → 绝对 y 索引（向上增大）
        # 目标瓦左上角绝对索引 c0/r0。x 向右增大 → k = j + (c0-bcol)
        # y 索引向上增大而栅格行向下增大 → 块行 k = (brow - r0) + j
        dc = c0 - bcol
        dr = brow - r0
        with rasterio.open(bpath) as src:
            sw, sh = src.width, src.height
            c_start = max(0, -dc)
            r_start = max(0, -dr)
            c_end = min(W, sw - dc)
            r_end = min(H, sh - dr)
            if c_end <= c_start or r_end <= r_start:
                continue
            w = c_end - c_start
            band = 2048
            for rr in range(r_start, r_end, band):
                hh = min(band, r_end - rr)
                a = src.read(1, window=Window(dc + c_start, dr + rr, w, hh))
                cur = dst[rr:rr + hh, c_start:c_start + w]
                # 共享全局像素格：重叠区两块的同一绝对像素必同值；
                # 仅当一块在边缘切片填 0（无数据）时用另一块的真值补上 → 非零优先
                np.copyto(cur, a, where=(a > 0))
            covered += 1
    prof = dict(driver='GTiff', width=W, height=H, count=1, dtype='uint8',
                crs='EPSG:4326', nodata=0, compress='deflate', predictor=2,
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    tmp = out + '.tmp'
    for attempt in range(5):
        try:
            with rasterio.open(tmp, 'w', **dict(prof, transform=(RES, 0.0, c0 * RES, 0.0, -RES, r0 * RES))) as d:
                d.write(dst, 1)
            for k in range(6):
                try:
                    os.replace(tmp, out)
                    break
                except PermissionError:
                    time.sleep(0.5 * (k + 1))
            else:
                raise IOError('replace failed')
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
    return (t, f'ok_blocks{covered}', os.path.getsize(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', nargs='*')
    ap.add_argument('--all-plan', action='store_true', help='全国 313 瓦')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    idx = build_index()
    print('源块索引:', len(idx), flush=True)
    if a.tiles:
        tiles = a.tiles
    elif a.all_plan:
        import pandas as pd
        df = pd.read_csv(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv')
        tiles = list(df.tile)
    else:
        tiles = [f'T{i:02d}{j:02d}' for i in range(31) for j in range(18)]
    # 只保留有陆地的
    st = {}
    t0 = time.time()
    jobs = [(t, idx, a.force) for t in tiles]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(do_tile, j): j[0] for j in jobs}
        import concurrent.futures as cf
        for n, f in enumerate(cf.as_completed(futs), 1):
            tt = futs[f]
            try:
                t, s, sz = f.result()
            except Exception as e:
                t, s, sz = tt, 'ERR:%s' % type(e).__name__, 0
            st[t] = dict(status=s, bytes=sz)
            if n % 10 == 0 or n == len(jobs):
                print('  %d/%d  %s  %.1f min' % (n, len(jobs), s, (time.time() - t0) / 60), flush=True)
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, 'w'), ensure_ascii=False, indent=1)
    from collections import Counter
    print(Counter(v['status'] for v in st.values()))


if __name__ == '__main__':
    main()
