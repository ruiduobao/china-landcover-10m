# -*- coding: utf-8 -*-
"""n20_expC3_points.py — 实验 C 第三批点表：三江 WorldCover 树覆盖层候选（补 P2"木本沼泽 vs 森林"样本）

* 背景（为什么）：C+C2 合并后 木本沼泽 21 点可用，但"森林"判读点仅 ~10（三江 FCS10 森林层点 17/20 被判为沼泽/草地），
  P2 仍 <15/类。第三批用 WorldCover v200(2021) 的树覆盖类(tree=10) 在 expC bbox 内抽样 —— 与 FCS10 独立来源的选点分层。
* 输入：GEE ESA/WorldCover/v200（class 10=tree）；expC/expC2 已有坐标（≥3 km 去重）
* 规则源：候选网格 0.05°；保留 class=10；贪心 ≥3 km；配额 40；seed 20261009。
* 门槛：真值仍由 z18 影像判读（rubric_C）；WorldCover 仅用于选点。
* 输出：data/m3/expC3_points.csv + expC3_plan.json
* 用法：python n20_expC3_points.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

M3D = r'F:/lc_work/v31_exp/data/m3'
N_WANT = 40
MIN_KM = 3.0
SEED = 20261009


def main():
    rng = np.random.default_rng(SEED)
    ex = pd.read_csv(os.path.join(M3D, 'expC_wetland_points.csv'))
    ex2 = pd.read_csv(os.path.join(M3D, 'expC2_points.csv'))
    fixed = list(zip(ex.lon.to_numpy(), ex.lat.to_numpy())) + list(zip(ex2.lon.to_numpy(), ex2.lat.to_numpy()))
    x0, x1, y0, y1 = ex.lon.min(), ex.lon.max(), ex.lat.min(), ex.lat.max()
    grid = [(x, y) for x in np.arange(x0, x1, 0.05) for y in np.arange(y0, y1, 0.05)]
    VC.ensure_ctx('zixen8v8')
    import ee
    pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'i': i}) for i, (x, y) in enumerate(grid)]
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    got = []
    for k in range(0, len(pts), 3000):
        r = wc.sampleRegions(collection=ee.FeatureCollection(pts[k:k + 3000]), scale=10, geometries=False,
                             tileScale=4).getInfo()['features']
        got += r
    cand = [(grid[f['properties']['i']], f['properties'].get('Map')) for f in got]
    tree = [(x, y) for (x, y), v in cand if v == 10]
    print('网格 %d；WorldCover 命中 %d；tree=10 %d' % (len(grid), len(got), len(tree)))
    sel = []
    fixed_xy = list(fixed)
    R = MIN_KM / 111.0
    idx = rng.permutation(len(tree))
    for i in idx:
        lo, la = tree[i]
        arr = np.array(fixed_xy + sel)
        d = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
        if d.min() < R:
            continue
        sel.append((lo, la))
        if len(sel) >= N_WANT:
            break
    rows = [{'point_id': 'EXPC3-%03d' % i, 'lon': x, 'lat': y, 'stratum': 'wc_tree', 'orig_id': 'wc200_10'}
            for i, (x, y) in enumerate(sel)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(M3D, 'expC3_points.csv'), index=False, encoding='utf-8-sig')
    VC.jsave({'n_grid': len(grid), 'n_hit': len(got), 'n_tree': len(tree), 'n_sel': int(len(df)),
              'note': 'WorldCover v200 tree 类仅用于选点分层；真值由 z18 影像判读'},
             os.path.join(M3D, 'expC3_plan.json'))
    print('选定 %d 点 → %s' % (len(df), os.path.join(M3D, 'expC3_points.csv')))


if __name__ == '__main__':
    main()
