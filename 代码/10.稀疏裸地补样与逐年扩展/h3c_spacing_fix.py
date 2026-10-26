# -*- coding: utf-8 -*-
"""h3c_spacing_fix.py — 用球面距离重做"与既有样本 250 m 去重 + 类内最小间距"（修正 h1 的等距圆柱度量缺陷）

* 背景：采集侧子代理核验发现 h1 的 km 换算用逐点 cos(lat) 构造绝对坐标，产生 0.1–0.13 km 剪切，
  导致约 1% 的点轻微违反 250/300/200 m 门限。本脚本用 pyproj.Geod(WGS84) 精确距离重做并剔除违规点。
* 输入：--final data/final/cand_qc.parquet（status=='keep'）；既有 2023 交付 gpkg（同类）
* 输出：--outdir/cand_qc_fixed.parquet（同 schema，仅保留通过者，status='keep'）+ spacing_fix.json
* 用法：python h3c_spacing_fix.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
SAMPLE_ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
NEAR_EXIST = 250.0
MIN_DIST = {'sparse': 300.0, 'bare': 200.0}
GEOD = Geod(ellps='WGS84')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', default=r'F:/lc_work/bare_sparse_v3/data/final/cand_qc.parquet')
    ap.add_argument('--outdir', default=r'F:/lc_work/bare_sparse_v3/data/final')
    a = ap.parse_args()
    d = pd.read_parquet(a.final)
    d = d[d.status == 'keep'].reset_index(drop=True)
    print('keep 输入 %d' % len(d), flush=True)

    # 既有同类点
    ex = {}
    for fam, nm in [('sparse', '稀疏植被'), ('bare', '裸地')]:
        xs, ys = [], []
        for f in glob.glob(os.path.join(SAMPLE_ROOT, nm, '2023', '*.gpkg')):
            import pyogrio
            g = pyogrio.read_dataframe(f, columns=[])
            xs.append(g.geometry.x.to_numpy()); ys.append(g.geometry.y.to_numpy())
        ex[fam] = (np.concatenate(xs), np.concatenate(ys))

    keep = np.ones(len(d), dtype=bool)
    stats = {}
    # 1) 与既有同类样本 250 m
    for fam in ('sparse', 'bare'):
        idx = np.flatnonzero((d.fam == fam).to_numpy())
        if len(idx) == 0:
            continue
        x, y = ex[fam]
        # 用球面距离：KDTree 在 3D 单位球上做近邻，再用 Geod 精确复核
        def to3(lon, lat):
            lonr = np.radians(lon); latr = np.radians(lat)
            return np.column_stack([np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)])
        # 粗筛半径：250 m ≈ 0.00225°；3D 弦长阈值用 300 m 保守
        thr3d = 2 * np.sin(300.0 / 6371000.0 / 2)
        tree = cKDTree(to3(x, y))
        dist3, nn = tree.query(to3(d.lon.to_numpy()[idx], d.lat.to_numpy()[idx]), k=1,
                               distance_upper_bound=thr3d)
        cand = np.flatnonzero(np.isfinite(dist3))
        n_drop = 0
        for j in cand:
            i = idx[j]
            _, _, dist = GEOD.inv(d.lon[i], d.lat[i], x[nn[j]], y[nn[j]])
            if dist < NEAR_EXIST:
                keep[i] = False; n_drop += 1
        stats[f'{fam}_near_existing_dropped'] = n_drop
        print('%s：与既有样本 <%.0f m 剔除 %d' % (fam, NEAR_EXIST, n_drop), flush=True)

    # 2) 类内最小间距（球面贪心，按 share9 降序）
    for fam in ('sparse', 'bare'):
        idx = np.flatnonzero((keep & (d.fam == fam).to_numpy()))
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-d.share9.to_numpy()[idx], kind='stable')]
        lon = d.lon.to_numpy(); lat = d.lat.to_numpy()
        tree = cKDTree(np.column_stack([lon, lat]))  # 度空间粗筛
        # 度→km 近似：1° lat ≈111.2 km；经度按最大纬度缩放（保守）
        latmax = np.abs(lat[order]).max()
        deg_km = 111.32 * max(np.cos(np.radians(latmax)), 0.2)
        radius_deg = MIN_DIST[fam] / deg_km
        kept_pts = []
        n_drop = 0
        for i in order:
            if len(kept_pts):
                arr = np.array(kept_pts)
                near = np.flatnonzero(np.abs(arr[:, 0] - lon[i]) < radius_deg)
                drop = False
                for j in near:
                    _, _, dist = GEOD.inv(lon[i], lat[i], arr[j, 0], arr[j, 1])
                    if dist < MIN_DIST[fam]:
                        drop = True; break
                if drop:
                    keep[i] = False; n_drop += 1
                    continue
            kept_pts.append((lon[i], lat[i]))
        stats[f'{fam}_min_dist_dropped'] = n_drop
        print('%s：类内 <%.0f m 剔除 %d' % (fam, MIN_DIST[fam], n_drop), flush=True)

    out = d[keep].reset_index(drop=True)
    out['status'] = 'keep'
    fp = os.path.join(a.outdir, 'cand_qc_fixed.parquet')
    out.to_parquet(fp, index=False)
    stats.update({'n_in': int(len(d)), 'n_out': int(len(out)),
                  'by_fam': {k: int(v) for k, v in out.fam.value_counts().items()}, 'file': fp})
    json.dump(stats, open(os.path.join(a.outdir, 'spacing_fix.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('→ %s  (%s)' % (fp, stats['by_fam']), flush=True)


if __name__ == '__main__':
    main()
