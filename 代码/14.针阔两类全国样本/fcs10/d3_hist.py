# -*- coding: utf-8 -*-
"""d3_hist.py — 全国 FCS10 瓦片码值直方图（精确计数）+ 可用性表
* 输入 / Z:/nb_national_2026/fcs10_dl/tiles/*.tif
* 规则源 / 官方 GLC_FCS10 图例与 colormap.lyr（奇=Closed、偶=Open）
* 门槛 / 分块读入（<=2048 行/块），内存 <1 GB/任务
* 输出 / F:/lc_work/nb_national_2026/fcs10_dl/hist_tiles.json  （逐瓦 256 桶精确直方图）
         F:/lc_work/nb_national_2026/fcs10_dl/tile_availability.csv
* 用法 / python d3_hist.py [--workers 4] [--tiles T1505 ...]
"""
import os, sys, json, glob, time, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
from rasterio.windows import Window

TILES = r'Z:/nb_national_2026/fcs10_dl/tiles'
OUT_JSON = r'F:/lc_work/nb_national_2026/fcs10_dl/hist_tiles.json'
OUT_CSV = r'F:/lc_work/nb_national_2026/fcs10_dl/tile_availability.csv'
PLAN = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv'


def hist_one(t):
    p = os.path.join(TILES, f'{t}_fcs10.tif')
    if not os.path.exists(p):
        return t, None
    cum = np.zeros(256, dtype=np.int64)
    with rasterio.open(p) as d:
        H, W = d.height, d.width
        for r in range(0, H, 2048):
            h = min(2048, H - r)
            a = d.read(1, window=Window(0, r, W, h))
            cum += np.bincount(a.ravel(), minlength=256)
        tr = tuple(d.transform)
        crs = str(d.crs)
    return t, dict(total=int(cum.sum()), hist=cum.tolist(),
                   transform=[float(x) for x in tr], width=W, height=H, crs=crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tiles', nargs='*')
    a = ap.parse_args()
    if a.tiles:
        tiles = a.tiles
    else:
        tiles = sorted(os.path.basename(p)[:5] for p in glob.glob(TILES + '/*_fcs10.tif'))
    print('待统计瓦片:', len(tiles), flush=True)
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (t, h) in enumerate(ex.map(hist_one, tiles), 1):
            res[t] = h
            if n % 20 == 0 or n == len(tiles):
                print('  %d/%d  %.1f min' % (n, len(tiles), (time.time() - t0) / 60), flush=True)
    json.dump(res, open(OUT_JSON, 'w'), ensure_ascii=False)
    print('写出', OUT_JSON, os.path.getsize(OUT_JSON) // 1024, 'KB')


if __name__ == '__main__':
    main()
