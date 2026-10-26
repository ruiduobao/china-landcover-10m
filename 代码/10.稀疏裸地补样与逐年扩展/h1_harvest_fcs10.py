# -*- coding: utf-8 -*-
"""h1_harvest_fcs10.py — 从本地 GLC_FCS10 2023 (10 m) 采东北/东部 稀疏植被·裸地 候选点

* 输入：E:/data/全球土地覆盖数据/数据/GLC_FCS10/GLC_FCS10maps_2023_*/GLC_FCS10_2023_*.tif
        （5°×5° uint8 瓦片；E 盘缺 130–135°E 带，报告注明）
* 规则源：F:/lc_work/bare_sparse_v3/spec/预注册判据.md §2
    - 目标码族：sparse={150,152,153} / bare={200,201,202}
    - 3×3 全纯 + 9×9 同类占比 ≥0.70；候选网格 5 px（≈50 m）
    - 0.025° 格配额：sparse ≤2 / bare ≤1；贪心最小间距 sparse 300 m / bare 200 m
    - 与既有 2023 交付同类样本 250 m 内去重
    - 分窗配额（§2.5）：避免单一沙地吃掉全部名额
* 输出：data/candidates/cand_<region>.parquet（两类合一）+ cand_<region>_stats.json
* 用法：
    python h1_harvest_fcs10.py --region north --outdir ../data/candidates
    python h1_harvest_fcs10.py --region north --limit-blocks 40      # 冒烟
* 自检：--self-check 随机 100 点/类回读栅格，复算 raw 码 + 3×3 纯，须 100% 一致
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
from rasterio.windows import Window, from_bounds
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
TILE_ROOT = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10'
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
SAMPLE_ROOT = os.path.join(PROJ, '数据', '单类样本的单独处理_2026.10.12')
FAM = {'sparse': (150, 152, 153), 'bare': (200, 201, 202)}
REGIONS = {'north': (113.0, 41.5, 126.0, 50.5),
           'east': (107.0, 18.0, 125.0, 38.5),
           'ne130': (130.0, 41.5, 135.0, 54.0)}
# 分窗：按顺序首个命中；配额 (sparse, bare)；'None' bbox 为兜底
SUBBOX = {
    'north': [
        ('科尔沁(118-124,42-45.5)', (118, 42, 124, 45.5), 2500, 1400),
        ('松嫩(122-126,44.5-48)', (122, 44.5, 126, 48), 900, 600),
        ('呼伦贝尔(116.5-121,47.5-50.5)', (116.5, 47.5, 121, 50.5), 800, 500),
        ('浑善达克(113.5-118,41.5-44.5)', (113.5, 41.5, 118, 44.5), 1100, 700),
        ('黑吉辽其余', None, 200, 300),
    ],
    'ne130': [
        ('黑龙江东部(130-135,42-54)', (130, 42, 135, 54), 30, 800),
        ('其余', None, 20, 200),
    ],
    'east': [
        ('华北(113-120,36-38.5)', (113, 36, 120, 38.5), 150, 500),
        ('黄淮山东(114-122,32-36)', (114, 32, 122, 36), 100, 250),
        ('华东华南(114-125,18-32)', (114, 18, 125, 32), 50, 250),
        ('黄土高原东(107-113.5,34-38.5)', (107, 34, 113.5, 38.5), 100, 200),
        ('其余', None, 50, 300),
    ],
}
CELL = 0.025
QUOTA = {'sparse': 2, 'bare': 1}
MIN_DIST = {'sparse': 300, 'bare': 200}
SHARE = 0.70
STRIDE = 5
BLOCK = 1.0
PAD = 5


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def tile_index():
    out = []
    for t in sorted(glob.glob(os.path.join(TILE_ROOT, '*', 'GLC_FCS10_2023_*.tif'))):
        with rasterio.open(t) as ds:
            out.append((t, ds.bounds))
    return out


def load_existing():
    xs, ys = [], []
    for nm in ('裸地', '稀疏植被'):
        for f in glob.glob(os.path.join(SAMPLE_ROOT, nm, '2023', '*.gpkg')):
            try:
                import pyogrio
                d = pyogrio.read_dataframe(f, columns=[])
                xs.append(d.geometry.x.to_numpy()); ys.append(d.geometry.y.to_numpy())
            except Exception as e:
                emit('  ! 读既有样本失败 %s: %s' % (f, e))
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


def assign_sub(region, lon, lat):
    idx = np.full(len(lon), -1, dtype=np.int16)
    for k, (nm, bb, _, _) in enumerate(SUBBOX[region]):
        if bb is None:
            continue
        m = (lon >= bb[0]) & (lon < bb[2]) & (lat >= bb[1]) & (lat < bb[3]) & (idx < 0)
        idx[m] = k
    idx[idx < 0] = len(SUBBOX[region]) - 1
    return idx


def thin_points(df, fam):
    if len(df) == 0:
        return df
    df = df.copy()
    df['_cell'] = (np.floor(df.lon / CELL).astype(np.int64) * 100000 +
                   np.floor(df.lat / CELL).astype(np.int64))
    df = df.sort_values(['_cell', 'share9'], ascending=[True, False])
    df = df.groupby('_cell', sort=False).head(QUOTA[fam]).reset_index(drop=True)
    dist_km = MIN_DIST[fam] / 1000.0
    km = np.column_stack([df.lon.to_numpy() * 111.32 * np.cos(np.radians(df.lat.to_numpy())),
                          df.lat.to_numpy() * 110.57])
    order = np.argsort(-df['share9'].to_numpy(), kind='stable')
    kept, keep = {}, np.zeros(len(df), dtype=bool)
    for i in order:
        kx = int(np.floor(km[i, 0] / dist_km)); ky = int(np.floor(km[i, 1] / dist_km))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((kx + dx, ky + dy), ()):
                    if (km[i, 0] - km[j, 0]) ** 2 + (km[i, 1] - km[j, 1]) ** 2 < dist_km ** 2:
                        ok = False; break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            kept.setdefault((kx, ky), []).append(i)
            keep[i] = True
    return df[keep].reset_index(drop=True)


def dedupe_existing(df, ex):
    if ex is None or len(df) == 0:
        return df, 0
    x, y = ex
    if len(x) == 0:
        return df, 0
    tree = cKDTree(np.column_stack([x * 111.32 * np.cos(np.radians(y)), y * 110.57]))
    km = np.column_stack([df.lon.to_numpy() * 111.32 * np.cos(np.radians(df.lat.to_numpy())),
                          df.lat.to_numpy() * 110.57])
    d, _ = tree.query(km, k=1, distance_upper_bound=0.25)
    hit = np.isfinite(d)
    return df[~hit].reset_index(drop=True), int(hit.sum())


def harvest(region, outdir, limit_blocks=None):
    bbox = REGIONS[region]
    tiles = tile_index()
    emit('region=%s bbox=%s tiles=%d' % (region, bbox, len(tiles)))
    ex = load_existing()
    emit('既有同类样本点：%s' % ('None' if ex is None else len(ex[0])))
    acc = {f: {'lon': [], 'lat': [], 'share9': [], 'raw': [], 'tile': []} for f in FAM}
    nb = 0
    t0 = time.time()
    last_log = [0]
    for t, b in tiles:
        if b.right <= bbox[0] or b.left >= bbox[2] or b.top <= bbox[1] or b.bottom >= bbox[3]:
            continue
        with rasterio.open(t) as ds:
            tr = ds.transform
            res = tr.a
            bx0 = max(bbox[0], np.ceil(b.left / BLOCK) * BLOCK)
            by0 = max(bbox[1], np.ceil(b.bottom / BLOCK) * BLOCK)
            bx1 = min(bbox[2], np.floor(b.right / BLOCK) * BLOCK)
            by1 = min(bbox[3], np.floor(b.top / BLOCK) * BLOCK)
            x = bx0
            while x < bx1 - 1e-9:
                y = by0
                while y < by1 - 1e-9:
                    nb += 1
                    if limit_blocks and nb > limit_blocks:
                        break
                    try:
                        w = from_bounds(x - PAD * res, y - PAD * res,
                                        x + BLOCK + PAD * res, y + BLOCK + PAD * res, tr)
                        w = w.intersection(Window(0, 0, ds.width, ds.height))
                        a = ds.read(1, window=w)
                    except Exception:
                        y += BLOCK; continue
                    if a.size == 0:
                        y += BLOCK; continue
                    for fam, codes in FAM.items():
                        m = np.isin(a, codes)
                        if int(m.sum()) == 0:
                            continue
                        pure = minimum_filter(m.astype(np.uint8), size=3, mode='nearest') == 1
                        sh = uniform_filter(m.astype(np.float32), size=9, mode='nearest')
                        cand = pure & (sh >= SHARE)
                        if not cand.any():
                            continue
                        rr, cc = np.nonzero(cand)
                        gr = rr + int(round(w.row_off)); gc = cc + int(round(w.col_off))
                        lon = tr.c + (gc + 0.5) * tr.a + (gr + 0.5) * tr.b
                        lat = tr.f + (gc + 0.5) * tr.d + (gr + 0.5) * tr.e
                        sel = ((lon >= x) & (lon < x + BLOCK) & (lat >= y) & (lat < y + BLOCK)
                               & ((gr % STRIDE) == 0) & ((gc % STRIDE) == 0))
                        if sel.any():
                            idx = np.flatnonzero(sel)
                            lo = lon[idx]; la = lat[idx]
                            sv = sh[rr[idx], cc[idx]]
                            rv = a[rr[idx], cc[idx]]
                            # 块内先按 0.025° 格配额稀释：否则单块可达 200 万候选，内存爆
                            cell = (np.floor(lo / CELL).astype(np.int64) * 100000
                                    + np.floor(la / CELL).astype(np.int64))
                            order = np.lexsort((-sv, cell))
                            cs = cell[order]
                            first = np.ones(len(order), dtype=bool)
                            first[1:] = cs[1:] != cs[:-1]
                            starts = np.flatnonzero(first)
                            run = np.cumsum(first) - 1
                            rank = np.arange(len(order)) - starts[run]
                            keep = order[rank < QUOTA[fam]]
                            acc[fam]['lon'].append(lo[keep]); acc[fam]['lat'].append(la[keep])
                            acc[fam]['share9'].append(sv[keep])
                            acc[fam]['raw'].append(rv[keep])
                            acc[fam]['tile'].extend([os.path.basename(t)] * len(keep))
                    y += BLOCK
                if limit_blocks and nb > limit_blocks:
                    break
                x += BLOCK
        if limit_blocks and nb > limit_blocks:
            break
        if nb - last_log[0] >= 20:
            last_log[0] = nb
            emit('  blocks=%d  sparse=%d  bare=%d  (%.0fs)' % (
                nb, sum(len(v) for v in acc['sparse']['raw']),
                sum(len(v) for v in acc['bare']['raw']), time.time() - t0))

    stats = {'region': region, 'blocks': int(nb), 'elapsed_scan_sec': round(time.time() - t0, 1)}
    parts = []
    fam_stats = {}
    for fam in FAM:
        a = acc[fam]
        if not a['raw']:
            fam_stats[fam] = {'raw_candidates': 0, 'final': 0}
            continue
        df = pd.DataFrame({'lon': np.concatenate(a['lon']), 'lat': np.concatenate(a['lat']),
                           'share9': np.concatenate(a['share9']),
                           'raw': np.concatenate(a['raw']).astype(np.int16),
                           'tile': a['tile']})
        df['fam'] = fam
        df['class'] = 150 if fam == 'sparse' else 201
        n_raw = len(df)
        df = df.drop_duplicates(['lon', 'lat']).reset_index(drop=True)
        n_dup = n_raw - len(df)
        df, n_ex = dedupe_existing(df, ex)
        df['subwin'] = assign_sub(region, df.lon.to_numpy(), df.lat.to_numpy())
        df = thin_points(df, fam)
        keep_parts = []
        sub_stats = {}
        for k, (nm, bb, t_sp, t_ba) in enumerate(SUBBOX[region]):
            target = t_sp if fam == 'sparse' else t_ba
            d = df[df.subwin == k]
            n0 = len(d)
            if len(d) > target:
                d = d.sample(n=target, random_state=7)
            sub_stats[nm] = {'after_thin': int(n0), 'kept': int(len(d)), 'target': int(target)}
            keep_parts.append(d)
        df = pd.concat(keep_parts).reset_index(drop=True)
        fam_stats[fam] = {'raw_candidates': int(n_raw), 'exact_dup': int(n_dup),
                          'near_existing_removed': int(n_ex),
                          'after_thin': int(sum(v['after_thin'] for v in sub_stats.values())),
                          'final': int(len(df)), 'subbox': sub_stats}
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['lon', 'lat', 'share9', 'raw', 'tile', 'fam', 'class', 'subwin'])
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, 'cand_%s.parquet' % region)
    out.to_parquet(fp, index=False)
    stats['fams'] = fam_stats
    stats['file'] = fp
    stats['total'] = int(len(out))
    emit('完成：%s  %d 点  → %s  （%.0fs）' % (region, len(out), fp, time.time() - t0))
    for fam in fam_stats:
        emit('   %s: %s' % (fam, fam_stats[fam]))
    return stats


def self_check(outdir, region):
    fp = os.path.join(outdir, 'cand_%s.parquet' % region)
    if not os.path.isfile(fp):
        return {}
    d = pd.read_parquet(fp)
    out = {}
    tiles = tile_index()
    for fam, codes in FAM.items():
        s = d[d.fam == fam]
        if len(s) == 0:
            out[fam] = {'n': 0}; continue
        s = s.sample(min(100, len(s)), random_state=7)
        ok_raw = 0; ok_pure = 0
        for _, r in s.iterrows():
            tt = [t for t, b in tiles if b.left <= r.lon < b.right and b.bottom <= r.lat < b.top]
            if not tt:
                continue
            with rasterio.open(tt[0]) as ds:
                tr = ds.transform
                col = int((r.lon - tr.c) / tr.a); row_ = int((r.lat - tr.f) / tr.e)
                a = ds.read(1, window=Window(col - 1, row_ - 1, 3, 3))
            if np.all(np.isin(a, codes)):
                ok_pure += 1
            if a[1, 1] in codes:
                ok_raw += 1
        out[fam] = {'n': int(len(s)), 'raw_ok': int(ok_raw), 'pure3_ok': int(ok_pure)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True, choices=list(REGIONS))
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'candidates'))
    ap.add_argument('--limit-blocks', type=int, default=None)
    ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()
    res = harvest(a.region, a.outdir, a.limit_blocks)
    if a.self_check:
        res['self_check'] = self_check(a.outdir, a.region)
        emit('自检: %s' % res['self_check'])
    fp = os.path.join(a.outdir, 'cand_%s_stats.json' % a.region)
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('stats → %s' % fp)


if __name__ == '__main__':
    main()
