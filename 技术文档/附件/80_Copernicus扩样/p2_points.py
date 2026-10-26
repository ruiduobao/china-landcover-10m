# -*- coding: utf-8 -*-
"""p2_points.py — 从 CGLS 100 m 核心区提取 10 m 候选点（规范 §3.4）

* 输入：
    cgls/core/{TILE}_needle_core.tif、{TILE}_broad_core.tif（P1 产物，uint8，1=核心）
    瓦片清单 代码/8.全国生产/plan/tiles.csv
    v2 已有点 labels/needle_broad_national_v2.parquet（用于 C4 重叠判定与去重）
* 规则源：技术文档/79 §3.4（预注册）
    - 每个 100 m 核心像元最多取 1 个 10 m 点
    - 点在 100 m 像元内取中心 ±20 m 随机偏移（不取几何中心，避免系统偏移）
    - 0.01° 内同类最多 1 点（与 v2 一致）
    - 每类全国上限 60,000，按"瓦 × 类"分层均分
* 门槛：无 GEE 调用（纯本地），峰值内存 < 2 GB
* 输出：cgls/needle_broad_cgls_candidates.parquet + cgls/p2_stats.json
* 用法：python p2_points.py [--per-tile 400] [--cap 60000]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:/lc_work/nb_national_2026/cgls'
CORE = os.path.join(ROOT, 'core')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
TILES_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
V2 = r'F:/lc_work/nb_national_2026/labels/needle_broad_national_v2.parquet'
R_E = 6378137.0
DECLUTTER_DEG = 0.01


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def inv_merc(x, y):
    lon = np.degrees(x / R_E)
    lat = np.degrees(np.arctan(np.sinh(y / R_E)))
    return lon, lat


def tile_points(path, leaf, rng, per_tile):
    """每 100 m 核心像元取 1 个 10 m 点（中心 ±20 m 随机偏移）。"""
    with rasterio.open(path) as s:
        a = s.read(1)
        tr = s.transform
    rr, cc = np.nonzero(a > 0)
    if len(rr) == 0:
        return pd.DataFrame()
    if len(rr) > per_tile:
        pick = rng.choice(len(rr), size=per_tile, replace=False)
        rr, cc = rr[pick], cc[pick]
    # 像元中心 + U(-20,20) 米（Mercator 米）
    jx = rng.uniform(-20, 20, len(rr))
    jy = rng.uniform(-20, 20, len(rr))
    xm = tr.c + (cc + 0.5) * tr.a + jx
    ym = tr.f + (rr + 0.5) * tr.e + jy
    lon, lat = inv_merc(xm, ym)
    return pd.DataFrame({'lon': lon, 'lat': lat, 'leaf': leaf,
                         'row100': rr, 'col100': cc})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-tile', type=int, default=400, help='每瓦每类最多点数')
    ap.add_argument('--cap', type=int, default=60000, help='每类全国上限')
    ap.add_argument('--seed', type=int, default=20261018)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    plan = pd.read_csv(TILES_CSV, encoding='utf-8-sig')
    tiles = list(plan.tile)
    have = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(CORE, '*_core.tif'))}
    emit('瓦片 %d / 有核心 tif %d' % (len(tiles), len(have)))

    parts = []
    per = []
    for i, t in enumerate(tiles):
        if t not in have:
            continue
        got = {}
        for leaf in ('needle', 'broad'):
            p = os.path.join(CORE, '%s_%s_core.tif' % (t, leaf))
            if not os.path.isfile(p):
                continue
            d = tile_points(p, leaf, rng, a.per_tile)
            if len(d):
                d['tile'] = t
                parts.append(d)
                got[leaf] = int(len(d))
            else:
                got[leaf] = 0
        per.append(dict(tile=t, **got))
        if (i + 1) % 50 == 0:
            emit('  %d/%d（%.1f min）累计 %d 点' % (
                i + 1, len(tiles), (time.time() - t0) / 60,
                sum(len(x) for x in parts)))

    if not parts:
        emit('无候选点')
        return
    d = pd.concat(parts, ignore_index=True)
    emit('原始候选 %d（针 %d / 阔 %d）' % (
        len(d), int((d.leaf == 'needle').sum()), int((d.leaf == 'broad').sum())))

    # ---- 0.01° 去聚团（同类同格最多 1 点）----
    d['gk'] = (np.floor(d.lon / DECLUTTER_DEG).astype('int64') * 10_000_000
               + np.floor(d.lat / DECLUTTER_DEG).astype('int64'))
    before = len(d)
    d['_r'] = rng.random(len(d))
    d = d.sort_values(['leaf', 'gk', '_r']).drop_duplicates(subset=['leaf', 'gk'], keep='first')
    emit('去聚团 %d → %d' % (before, len(d)))

    # ---- 每类上限（按瓦分层均分）----
    out = []
    for leaf, g in d.groupby('leaf'):
        if len(g) > a.cap:
            cap_tile = max(1, a.cap // max(g.tile.nunique(), 1))
            g = g.groupby('tile', group_keys=False).head(cap_tile).head(a.cap)
        out.append(g)
    d = pd.concat(out, ignore_index=True).drop(columns=['_r', 'gk'])
    emit('上限截断后 %d（针 %d / 阔 %d）' % (
        len(d), int((d.leaf == 'needle').sum()), int((d.leaf == 'broad').sum())))

    # ---- C4：与 v2 的 0.01° 格重叠率 ----
    v2 = pd.read_parquet(V2, columns=['lon', 'lat'])
    v2k = set((np.floor(v2.lon / DECLUTTER_DEG).astype('int64') * 10_000_000
               + np.floor(v2.lat / DECLUTTER_DEG).astype('int64')).values)
    dk = (np.floor(d.lon / DECLUTTER_DEG).astype('int64') * 10_000_000
          + np.floor(d.lat / DECLUTTER_DEG).astype('int64'))
    in_v2 = dk.isin(v2k).values
    emit('C4 与 v2 的 0.01° 格重叠：%d / %d = %.4f' % (
        int(in_v2.sum()), len(d), in_v2.mean()))
    d['overlap_v2'] = in_v2.astype('int8')

    d['point_id'] = np.arange(len(d), dtype='int64') + 700_000_000
    d['src'] = 'cgls_core'
    d['coverage'] = 'cgls_lc100_c3_2017_2019'
    d = d[['point_id', 'lon', 'lat', 'tile', 'leaf', 'row100', 'col100',
           'overlap_v2', 'src', 'coverage']]
    outp = os.path.join(ROOT, 'needle_broad_cgls_candidates.parquet')
    d.to_parquet(outp, index=False)

    st = dict(
        n=int(len(d)), by_leaf=d.leaf.value_counts().to_dict(),
        tiles=int(d.tile.nunique()),
        raw_before_declutter=before,
        params=dict(per_tile=a.per_tile, cap=a.cap, seed=a.seed,
                    declutter_deg=DECLUTTER_DEG, jitter_m=20),
        C4_overlap_vs_v2=dict(n_overlap=int(in_v2.sum()), rate=round(float(in_v2.mean()), 4)),
        per_tile=per,
        elapsed_min=round((time.time() - t0) / 60, 2))
    json.dump(st, open(os.path.join(ROOT, 'p2_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('写 %s（%d 行，%.1f MB）' % (outp, len(d), os.path.getsize(outp) / 1e6))


if __name__ == '__main__':
    main()
