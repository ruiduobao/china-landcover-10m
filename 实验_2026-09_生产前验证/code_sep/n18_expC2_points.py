# -*- coding: utf-8 -*-
"""n18_expC2_points.py — 实验 C 第二轮点表：三江 FCS10 湿地/裸地/森林层分层补样

* 背景（为什么）：第一轮 403 点判读后，关键对 P2（木本沼泽 vs 森林，12/5）与 P3（湖河滩地 vs 裸地，37/14）
  样本不足（<15/类），需补样；P1（草沼 vs 草地）已判"不可分"（AUC 0.742，n=205）不受影响。
* 输入：F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/*.tif（本地 FCS10-2023 90 m 瓦片，覆盖三江）
        + data/m3/expC_wetland_points.csv（第一轮点，用于 ≥3 km 去重）
* 规则源：FCS10 码**仅用于选点分层**（真值仍由 z18 影像判读）：
  181 木本沼泽 ×28；183 湖河滩地 ×20；200/201/202 裸地/沙地 ×24；51/52/61/62/71/72/81/82 森林 ×20；130 草地 ×16；
  点间距 ≥3 km（与第一轮点也不重叠）；seed 20261009。
* 门槛：分层不足时按可得取；输出含 stratum 与 fcs10_code。
* 输出：data/m3/expC2_points.csv + data/m3/expC2_plan.json
* 用法：python n18_expC2_points.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
import v31_common as VC

M3D = r'F:/lc_work/v31_exp/data/m3'
TILD = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10'
SEED = 20261009
MIN_KM = 3.0
QUOTA = {'181_木本沼泽': ([181], 28), '183_湖河滩地': ([183], 20),
         '200s_裸地沙地': ([200, 201, 202], 24),
         'forest_森林': ([51, 52, 61, 62, 71, 72, 81, 82], 20), '130_草地': ([130], 16)}


def greedy_spaced(xy, n, min_km, rng, fixed):
    idx = rng.permutation(len(xy))
    sel = []
    R = min_km / 111.0
    pts = list(fixed)
    for i in idx:
        lo, la = xy[i]
        if pts:
            arr = np.array(pts)
            d = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
            if d.min() < R:
                continue
        sel.append(i)
        pts.append((lo, la))
        if len(sel) >= n:
            break
    return sel


def main():
    rng = np.random.default_rng(SEED)
    ex = pd.read_csv(os.path.join(M3D, 'expC_wetland_points.csv'))
    x0, x1, y0, y1 = ex.lon.min(), ex.lon.max(), ex.lat.min(), ex.lat.max()
    fixed = list(zip(ex.lon.to_numpy(), ex.lat.to_numpy()))
    tiles = []
    for f in sorted(glob.glob(os.path.join(TILD, '*.tif'))):
        with rasterio.open(f) as ds:
            b = ds.bounds
            if b.left < x1 and b.right > x0 and b.bottom < y1 and b.top > y0:
                tiles.append(f)
    print('覆盖瓦片 %d 个: %s' % (len(tiles), [os.path.basename(t) for t in tiles]))
    # 收集各层候选坐标（栅格抽样 stride=4；每层每瓦片最多随机取 4000 个匹配像元再转坐标，控内存）
    cand = {k: [] for k in QUOTA}
    for f in tiles:
        with rasterio.open(f) as ds:
            arr = ds.read(1)
            h, w = arr.shape
            sub = arr[0:h:4, 0:w:4]
            for k, (codes, _) in QUOTA.items():
                m = np.isin(sub, codes)
                if not m.any():
                    continue
                rr, cc = np.nonzero(m)
                if len(rr) > 4000:
                    i = rng.choice(len(rr), 4000, replace=False)
                    rr, cc = rr[i], cc[i]
                xs, ys = rasterio.transform.xy(ds.transform, rr * 4, cc * 4)
                xs = np.atleast_1d(xs); ys = np.atleast_1d(ys)
                inb = (xs >= x0 - 0.5) & (xs <= x1 + 0.5) & (ys >= y0 - 0.5) & (ys <= y1 + 0.5)
                cand[k].extend(zip(xs[inb].tolist(), ys[inb].tolist()))
    rows, plan = [], {}
    picked = []
    for k, (codes, n) in QUOTA.items():
        xy = [(a, b) for a, b in cand[k]]
        sel = greedy_spaced(xy, n, MIN_KM, rng, fixed + picked)
        plan[k] = dict(want=n, avail=len(xy), got=len(sel))
        for i in sel:
            lon, lat = xy[i]
            picked.append((lon, lat))
            rows.append({'point_id': 'EXPC2-%03d' % len(rows), 'lon': lon, 'lat': lat,
                         'stratum': k, 'orig_id': 'fcs10_%s' % k.split('_')[0]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(M3D, 'expC2_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'seed': SEED, 'min_km': MIN_KM, 'plan': plan, 'tiles': [os.path.basename(t) for t in tiles],
              'n_total': int(len(df)), 'by_stratum': df.stratum.value_counts().to_dict(),
              'note': 'FCS10 码仅用于选点分层；真值由 z18 影像判读（rubric_C）'},
             os.path.join(M3D, 'expC2_plan.json'))
    print(json.dumps(plan, ensure_ascii=False))
    print('总点数 %d → %s' % (len(df), os.path.join(M3D, 'expC2_points.csv')))


if __name__ == '__main__':
    main()
