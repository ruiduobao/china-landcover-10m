# -*- coding: utf-8 -*-
"""h8b_areas.py — 各分窗 FCS10 2023 稀疏植被/裸地面积（用于补样点密度对账）

* 输入：E 盘 GLC_FCS10 10m 瓦片；分窗定义取自 h1_harvest_fcs10.SUBBOX
* 输出：data/final/fcs10_areas.json  {region: {subwin: {sparse_km2, bare_km2}}}
* 用法：python h8b_areas.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1_harvest_fcs10 as H  # noqa

TILE_ROOT = H.TILE_ROOT
PX = 100.0 / 1e6


def main():
    t0 = time.time()
    tiles = []
    for t in sorted(glob.glob(os.path.join(TILE_ROOT, '*', 'GLC_FCS10_2023_*.tif'))):
        with rasterio.open(t) as ds:
            tiles.append((t, ds.bounds))
    out = {}
    for region, subs in H.SUBBOX.items():
        out[region] = {}
        for nm, bb, _, _ in subs:
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            sp = ba = 0
            for t, b in tiles:
                if b.right <= x0 or b.left >= x1 or b.top <= y0 or b.bottom >= y1:
                    continue
                with rasterio.open(t) as ds:
                    tr = ds.transform
                    xx = x0
                    while xx < x1 - 1e-9:
                        yy = y0
                        while yy < y1 - 1e-9:
                            xx2 = min(xx + 1, x1); yy2 = min(yy + 1, y1)
                            try:
                                w = from_bounds(xx, yy, xx2, yy2, tr)
                                w = w.intersection(Window(0, 0, ds.width, ds.height))
                                a = ds.read(1, window=w)
                            except Exception:
                                yy = yy2; continue
                            if a.size:
                                u, c = np.unique(a, return_counts=True)
                                sp += int(c[np.isin(u, (150, 152, 153))].sum())
                                ba += int(c[np.isin(u, (200, 201, 202))].sum())
                            yy = yy2
                        xx += 1
            out[region][nm] = {'sparse_km2': round(sp * PX, 1), 'bare_km2': round(ba * PX, 1)}
            print('%s / %-34s 稀疏 %9.1f km²  裸地 %9.1f km²  (%.0fs)' % (
                region, nm, sp * PX, ba * PX, time.time() - t0), flush=True)
    fp = r'F:/lc_work/bare_sparse_v3/data/final/fcs10_areas.json'
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', fp)


if __name__ == '__main__':
    main()
