# -*- coding: utf-8 -*-
"""n21_expC4_points.py — 实验 C 第四批点表：黑龙江交付成品图的森林像元候选（补 P2"木本沼泽 vs 森林"）

* 背景（为什么）：C/C2/C3 三批后，"森林"判读点仍仅 9（三江 FCS10 森林层与 WorldCover 树层点被独立判读为湿地/草地），
  P2 无法裁决。本批改用**本项目 2023 黑龙江交付成品的森林类像元**找候选点（成品图只用于选点，真值仍由 z18 判读）。
* 输入：F:/lc_work/prod5p_2023/rasters_batch/T{2813..3015}_10m.tif（129–135E, 44–50N）
        + expC/expC2/expC3 已有坐标（≥3 km 去重）
* 规则源：类码 4/5/6/7/8（v31 森林五类）像元，200 m 步长扫描，expC bbox 内，贪心 ≥3 km，配额 40，seed 20261009。
* 门槛：真值仍由 z18 影像判读（rubric_C，森林细分五类名）。
* 输出：data/m3/expC4_points.csv + expC4_plan.json
* 用法：python n21_expC4_points.py
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
RB = r'F:/lc_work/prod5p_2023/rasters_batch'
N_WANT = 40
MIN_KM = 3.0
SEED = 20261009
FOREST = [4, 5, 6, 7, 8]


def main():
    rng = np.random.default_rng(SEED)
    ex = pd.read_csv(os.path.join(M3D, 'expC_wetland_points.csv'))
    fixed = list(zip(ex.lon.to_numpy(), ex.lat.to_numpy()))
    for f in ['expC2_points.csv', 'expC3_points.csv']:
        d = pd.read_csv(os.path.join(M3D, f))
        fixed += list(zip(d.lon.to_numpy(), d.lat.to_numpy()))
    x0, x1, y0, y1 = ex.lon.min(), ex.lon.max(), ex.lat.min(), ex.lat.max()
    cand = []
    for fp in sorted(glob.glob(os.path.join(RB, 'T*_10m.tif'))):
        with rasterio.open(fp) as ds:
            b = ds.bounds
            if not (b.left < x1 and b.right > x0 and b.bottom < y1 and b.top > y0):
                continue
            arr = ds.read(1)
            h, w = arr.shape
            stride = max(1, int(round(200.0 / 10.0)))          # 200 m 步长
            sub = arr[0:h:stride, 0:w:stride]
            m = np.isin(sub, FOREST)
            if not m.any():
                continue
            rr, cc = np.nonzero(m)
            if len(rr) > 3000:
                i = rng.choice(len(rr), 3000, replace=False)
                rr, cc = rr[i], cc[i]
            xs, ys = rasterio.transform.xy(ds.transform, rr * stride, cc * stride)
            for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
                if x0 - 0.3 <= x <= x1 + 0.3 and y0 - 0.3 <= y <= y1 + 0.3:
                    cand.append((float(x), float(y)))
    print('森林候选像元 %d' % len(cand))
    sel = []
    R = MIN_KM / 111.0
    idx = rng.permutation(len(cand))
    for i in idx:
        lo, la = cand[i]
        arr = np.array(fixed + sel)
        d = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
        if d.min() < R:
            continue
        sel.append((lo, la))
        if len(sel) >= N_WANT:
            break
    rows = [{'point_id': 'EXPC4-%03d' % i, 'lon': x, 'lat': y, 'stratum': 'prod_forest',
             'orig_id': 'prod2023_forest'} for i, (x, y) in enumerate(sel)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(M3D, 'expC4_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'n_cand': len(cand), 'n_sel': int(len(df)), 'step_m': 200, 'min_km': MIN_KM,
              'note': '本项目 2023 成品森林类像元仅用于选点；真值由 z18 影像判读'},
             os.path.join(M3D, 'expC4_plan.json'))
    print('选定 %d 点 → %s' % (len(df), os.path.join(M3D, 'expC4_points.csv')))


if __name__ == '__main__':
    main()
