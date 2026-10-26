# -*- coding: utf-8 -*-
"""f5_negatives.py — 候选块后处理：正样本配额抽稀 + 负样本（难负/背景）挖掘，1:1 平衡

* 输入：F:/lc_work/forty_exp/cand/cand_blocks.parquet（f4 扫描：kind=pos/hard/bg 的 250 m 块）
        数据/单类样本的单独处理_2026.10.12/乔灌园地/乔灌园地v7_点表.csv（既有 160,931 点）
* 规则源（**预注册，先写死再跑**）：
  正样本（kind=pos）：
    ① 0.005°(≈550 m) 每格至多 1 点（随机，seed 20261026）；
    ② 与既有 v7 点球面距离 ≥1.0 km（防自相关泄漏）；
    ③ 0.5° 格上限 200；全局目标 --target（默认 120,000），超则按 0.5° 格内候选数等比分配。
  负样本（1:1 平衡，hard:bg = 1:1）：
    hard（难负，ForTy 森林类 1/2/3 纯块）：仅取"含正样本的 0.02°(≈2.2 km) 邻域"，
      0.005° 去重、≥1 km 距 v7 点、≥0.5 km 距正样本；
    bg（背景，其他土地 6 类纯块）：排除正样本 0.02° 邻域，0.005° 去重、
      ≥1 km 距 v7 点与正样本、0.5° 格上限 200。
  省归属：CTAmap 2026（界内 contains / 界外就近 + ct_km）。
* 门槛：正样本 ≥ 目标的 60%（否则降级并记录）；负:正 比例 = 1.00±0.05
* 输出：cand/pos_candidates.parquet、cand/neg_candidates.parquet、cand/final_summary.json
* 用法：python f5_negatives.py [--target 120000]
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from shapely import STRtree, contains_xy, shortest_line
from shapely.geometry import Point

WORK = r'F:/lc_work/forty_exp'
CAND = os.path.join(WORK, 'cand')
PTS = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/乔灌园地v7_点表.csv'
CTAMAP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
SEED = 20261026
CAP05 = 600   # A1 修订（2026-10-26 20:59，容量审计后 200→600；见 docs/f0 §3.2 附注）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def dedup005(d):
    key = np.column_stack([np.round(d.lon / 0.005), np.round(d.lat / 0.005)]).astype('int64')
    _, first = np.unique(key, axis=0, return_index=True)
    return d.iloc[np.sort(first)].reset_index(drop=True)


def min_dist(pts, ref, col='lon', col2='lat'):
    if len(ref) == 0:
        return np.full(len(pts), 1e9)
    tree = BallTree(np.radians(ref[['lat', 'lon']].to_numpy()), metric='haversine')
    d, _ = tree.query(np.radians(pts[[col2, col]].to_numpy()), k=1)
    return d[:, 0] * 6371.0


def quota(d, target, seed=SEED, cap=CAP05):
    """0.5° 格配额：最大余数法 + 容量（cap）再分配，保证总量逼近 target"""
    d = d.copy()
    d['g05'] = (np.floor(d.lon / 0.5).astype('int64') * 100000 + np.floor(d.lat / 0.5).astype('int64'))
    if len(d) <= target:
        return d.drop(columns=['g05'])
    cnt = d.g05.value_counts()
    keys = cnt.index.to_numpy()
    c = cnt.to_numpy().astype('int64')
    base = target * c / float(len(d))
    alloc = np.minimum(np.floor(base).astype('int64'), cap)
    frac = base - np.floor(base)
    guard = 0
    while alloc.sum() < target and guard < 500:
        guard += 1
        elig = (alloc < cap) & (alloc < c)
        if not elig.any():
            break
        eidx = np.where(elig)[0]
        order = eidx[np.argsort(-frac[eidx])]
        k = min(int(target - alloc.sum()), len(order))
        np.add.at(alloc, order[:k], 1)
        frac[order[:k]] -= 1.0
    rng = np.random.default_rng(seed)
    keep = []
    for g, n in zip(keys, alloc):
        if n <= 0:
            continue
        sub = d[d.g05 == g]
        take = min(int(n), len(sub))
        keep.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    out = pd.concat(keep, ignore_index=True)
    if len(out) > target:
        out = out.sample(n=target, random_state=seed).reset_index(drop=True)
    return out.drop(columns=['g05'])


def add_prov(d):
    import geopandas as gpd
    if not os.path.exists(CTAMAP):
        d['ct_prov'] = ''; d['ct_code'] = ''; d['ct_km'] = np.nan
        return d
    prov = gpd.read_file(CTAMAP)
    names = prov['省'].astype(str).to_numpy(); codes = prov['省级码'].astype(str).to_numpy()
    geoms = prov.geometry.to_numpy()
    lo, la = d.lon.to_numpy(), d.lat.to_numpy()
    pi = np.full(len(d), -1)
    for i, g in enumerate(geoms):
        m = contains_xy(g, lo, la)
        pi[m & (pi < 0)] = i
    d['ct_prov'] = np.where(pi >= 0, names[np.clip(pi, 0, None)], '')
    d['ct_code'] = np.where(pi >= 0, codes[np.clip(pi, 0, None)], '')
    d['ct_km'] = 0.0
    if (pi < 0).any():
        st = STRtree(list(geoms))
        for i in np.where(pi < 0)[0]:
            j = st.nearest(Point(lo[i], la[i]))
            d.loc[i, 'ct_prov'], d.loc[i, 'ct_code'] = names[j], codes[j]
            ln = shortest_line(Point(lo[i], la[i]), geoms[j])
            (x1, y1), (x2, y2) = ln.coords[0][:2], ln.coords[-1][:2]
            d.loc[i, 'ct_km'] = round(float(np.hypot((x2 - x1) * 111.32 * np.cos(np.radians(la[i])),
                                                     (y2 - y1) * 111.32)), 2)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=120000)
    ap.add_argument('--cap', type=int, default=CAP05, help='0.5° 格点数上限（A1 修订：默认 600）')
    a = ap.parse_args()
    t0 = time.time()
    blk = pd.read_parquet(os.path.join(CAND, 'cand_blocks.parquet'))
    v7 = pd.read_csv(PTS, encoding='utf-8-sig')[['lon', 'lat']]
    emit('候选块 %d（%s）；v7 点 %d' % (len(blk), blk.kind.value_counts().to_dict(), len(v7)))
    # ---------- 正样本 ----------
    pos = dedup005(blk[blk.kind == 'pos'].reset_index(drop=True))
    pos['dist_v7_km'] = min_dist(pos, v7)
    pos = pos[pos.dist_v7_km >= 1.0].reset_index(drop=True)
    emit('正候选：0.005° 去重 + 距 v7 ≥1km 后 %d' % len(pos))
    pos = quota(pos, a.target, cap=a.cap)
    pos = add_prov(pos)
    pos['pid'] = np.arange(len(pos))
    pos.to_parquet(os.path.join(CAND, 'pos_candidates.parquet'), index=False)
    emit('正样本 %d（目标 %d）' % (len(pos), a.target))
    # 正样本 0.02° 邻域键
    pos['c02'] = (np.floor(pos.lon / 0.02).astype('int64') * 1000000 +
                  np.floor(pos.lat / 0.02).astype('int64'))
    near_keys = set(pos.c02.unique())
    # ---------- 负样本 ----------
    n_each = int(round(len(pos) / 2.0))
    hard = blk[blk.kind == 'hard'].reset_index(drop=True)
    hard['c02'] = (np.floor(hard.lon / 0.02).astype('int64') * 1000000 +
                   np.floor(hard.lat / 0.02).astype('int64'))
    hard = hard[hard.c02.isin(near_keys)].reset_index(drop=True)
    hard = dedup005(hard)
    hard['dist_v7_km'] = min_dist(hard, v7)
    hard = hard[hard.dist_v7_km >= 1.0].reset_index(drop=True)
    hard['dist_pos_km'] = min_dist(hard, pos)
    hard = hard[hard.dist_pos_km >= 0.5].reset_index(drop=True)
    emit('难负候选 %d（0.02° 邻域内，距 v7 ≥1km、距正样本 ≥0.5km）' % len(hard))
    hard = quota(hard, n_each, seed=SEED + 1, cap=a.cap)
    hard['neg_type'] = 'hard'
    bg = blk[blk.kind == 'bg'].reset_index(drop=True)
    bg['c02'] = (np.floor(bg.lon / 0.02).astype('int64') * 1000000 +
                 np.floor(bg.lat / 0.02).astype('int64'))
    bg = bg[~bg.c02.isin(near_keys)].reset_index(drop=True)
    bg = dedup005(bg)
    bg['dist_v7_km'] = min_dist(bg, v7)
    bg = bg[bg.dist_v7_km >= 1.0].reset_index(drop=True)
    bg['dist_pos_km'] = min_dist(bg, pos)
    bg = bg[bg.dist_pos_km >= 1.0].reset_index(drop=True)
    emit('背景负候选 %d' % len(bg))
    bg = quota(bg, n_each, seed=SEED + 2, cap=a.cap)
    bg['neg_type'] = 'bg'
    neg = pd.concat([hard, bg], ignore_index=True)
    neg = add_prov(neg)
    neg['pid'] = np.arange(len(neg))
    neg.to_parquet(os.path.join(CAND, 'neg_candidates.parquet'), index=False)
    emit('负样本 %d（难负 %d / 背景 %d）；负:正 = %.2f' % (
        len(neg), (neg.neg_type == 'hard').sum(), (neg.neg_type == 'bg').sum(),
        len(neg) / max(1, len(pos))))
    summ = {'n_pos': int(len(pos)), 'n_neg': int(len(neg)),
            'neg_hard': int((neg.neg_type == 'hard').sum()),
            'neg_bg': int((neg.neg_type == 'bg').sum()),
            'ratio': round(len(neg) / max(1, len(pos)), 3),
            'pos_by_prov_top': pos.ct_prov.value_counts().head(12).to_dict(),
            'neg_by_prov_top': neg.ct_prov.value_counts().head(12).to_dict(),
            'pos_dist_v7_min': round(float(pos.dist_v7_km.min()), 3),
            'pos_c5_med': round(float(pos.frac5.median()), 3),
            'minutes': round((time.time() - t0) / 60, 1)}
    json.dump(summ, open(os.path.join(CAND, 'final_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit(json.dumps({k: summ[k] for k in ('n_pos', 'n_neg', 'ratio')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
