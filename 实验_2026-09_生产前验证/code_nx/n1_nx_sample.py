# -*- coding: utf-8 -*-
"""n1_nx_sample.py — 宁夏小样本实验·中性判读集构建（M3-A 独立点 + 定向补点）

* 输入：F:/lc_work/v31_exp/data/m3/m3a_points.csv（全球独立布点，≥1km 间距）
        交付成品 6 瓦 F:/lc_work/prod5p_2023/rasters_batch/T{1509,1608,1609,1610,1709,1710}_10m.tif
        数据/边界/china_100000_full.json（宁夏面）
* 规则源：预注册实验设计（2026-10-09）——判 R0 全池 vs R4 全筛谁更接近"影像真值"：
        A 组 = M3-A 落在宁夏界内的点（区域×WorldCover 分层、≥1km 独立）→ 全区口径
        S1 组 = 成品图中"落叶/常绿灌丛"的内部像元（11×11 窗口同质）→ 直接检验灌丛层真假
        S2 组 = 成品图中灌丛与草地/裸地交界 5px 内像元 → 分歧最集中处
        三组并集统一 ≥1km 最小间距（贪心）；候选在 1/4 概览上抽取、落点在全分辨率上复核类值。
* 门槛：A 组全取；S1/S2 各 30 点（不足则全取）；抽样种子 20261009 可复现
* 输出：F:/lc_work/v31_exp/data/m3/nx_neutral_points.csv（point_id/lon/lat/组/map_class/source）
        + 统计打印（各组点数、成品类构成）
* 用法：python n1_nx_sample.py [--force]
幂等：输出已存在且 --force 未给则跳过。
"""
import csv
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import rasterio

M3D = r'F:/lc_work/v31_exp/data/m3'
M3A = os.path.join(M3D, 'm3a_points.csv')
OUT = os.path.join(M3D, 'nx_neutral_points.csv')
NX = ['T1509', 'T1608', 'T1609', 'T1610', 'T1709', 'T1710']
RAST = [r'F:/lc_work/prod5p_2023/rasters_batch/%s_10m.tif' % t for t in NX]
BOUND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
SEED = 20261009
SHRUB = (9, 10)
GRASSBARE = (11, 22)
N_S1, N_S2 = 30, 30
MIN_M = 1000.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nx_polygon():
    from shapely.geometry import shape
    from shapely import make_valid
    gj = json.load(open(BOUND, encoding='utf-8'))
    for ft in gj['features']:
        if ft.get('properties', {}).get('name', '').startswith('宁夏'):
            return make_valid(shape(ft['geometry']))
    raise SystemExit('宁夏面未找到')


def integral(mask):
    return np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def win_sum(ii, r0, r1, c0, c1):
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]


def candidates_from_tile(fp, poly, rng):
    """在 1/4 概览上找 S1（灌丛内部）与 S2（灌丛—草/裸交界）候选，返回 lon/lat 列表。"""
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
        h, w = H // 4, W // 4
        v = ds.read(1, out_shape=(h, w), resampling=rasterio.enums.Resampling.nearest)
        tr = ds.transform * ds.transform.scale(W / w, H / h)
        shrub = np.isin(v, SHRUB)
        gb = np.isin(v, GRASSBARE)
        k = 5                                     # 概览 5px ≈ 全分辨 20px ≈ 200m
        ii_s, ii_g = integral(shrub.astype(np.int32)), integral(gb.astype(np.int32))
        rr, cc = np.mgrid[0:h, 0:w]
        r0 = np.clip(rr - k, 0, h); r1 = np.clip(rr + k, 0, h)
        c0 = np.clip(cc - k, 0, w); c1 = np.clip(cc + k, 0, w)
        n_s = win_sum(ii_s, r0, r1, c0, c1)
        n_g = win_sum(ii_g, r0, r1, c0, c1)
        area = (r1 - r0) * (c1 - c0)
        s1 = shrub & (n_s == area)                # 全窗口同质灌丛
        s2 = shrub & (n_g > 0)                    # 与草/裸同窗
        out = {}
        for tag, mask, n in (('S1', s1, N_S1), ('S2', s2, N_S2)):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                out[tag] = []
                continue
            pick = rng.choice(len(ys), size=min(len(ys), n * 8), replace=False)
            pts = []
            for i in pick:
                r, c = int(ys[i]), int(xs[i])
                x, y = tr * (c + 0.5, r + 0.5)
                pts.append((x, y))
            out[tag] = pts
    return out


def full_res_class(lon, lat, cache):
    """全分辨率读类值（成品 10m）；返回 (class, tile) 或 (0, None)。"""
    for fp in RAST:
        t = os.path.basename(fp).split('_')[0]
        ds = cache.get(t)
        if ds is None:
            ds = rasterio.open(fp)
            cache[t] = ds
        b = ds.bounds
        if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
            r, c = ds.index(lon, lat)
            if 0 <= r < ds.height and 0 <= c < ds.width:
                return int(ds.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0, 0]), t
    return 0, None


def main():
    if os.path.exists(OUT) and '--force' not in sys.argv:
        emit('已存在 %s（--force 重写）' % OUT)
        return
    rng = np.random.RandomState(SEED)
    poly = nx_polygon()
    from shapely.geometry import Point

    # ---- A 组：M3-A 宁夏界内 ----
    A = []
    for p in csv.DictReader(open(M3A, encoding='utf-8-sig')):
        lon, lat = float(p['lon']), float(p['lat'])
        if poly.contains(Point(lon, lat)):
            A.append(dict(point_id=p['point_id'], lon=lon, lat=lat, grp='A',
                          map_class='', source='M3-A(region=%s,wc=%s)' % (p['region'], p['wc'])))
    emit('A 组（M3-A 界内）= %d 点' % len(A))

    # ---- S1/S2：从成品图抽 ----
    picked = []
    for fp in RAST:
        cand = candidates_from_tile(fp, poly, rng)
        for tag, pts in cand.items():
            for lon, lat in pts:
                if poly.contains(Point(lon, lat)):
                    picked.append((tag, lon, lat))
    emit('S1/S2 原始候选 = %d 个' % len(picked))

    # 贪心最小间距（与 A 组、组内均 ≥1km）
    kept = {'S1': [], 'S2': []}
    anchors = [(p['lon'], p['lat']) for p in A]
    rng.shuffle(picked)
    for tag, lon, lat in picked:
        if len(kept[tag]) >= (N_S1 if tag == 'S1' else N_S2):
            continue
        ok = all(haversine_m(lon, lat, x, y) >= MIN_M for x, y in anchors)
        if ok:
            for tt in kept:
                ok = ok and all(haversine_m(lon, lat, x, y) >= MIN_M for x, y in kept[tt])
        if ok:
            kept[tag].append((lon, lat))
            anchors.append((lon, lat))
    emit('S1 选定 %d，S2 选定 %d' % (len(kept['S1']), len(kept['S2'])))

    cache = {}
    rows = list(A)
    for tag in ('S1', 'S2'):
        for i, (lon, lat) in enumerate(kept[tag], 1):
            cls, tile = full_res_class(lon, lat, cache)
            rows.append(dict(point_id='NX-%s%03d' % (tag, i), lon=round(lon, 7), lat=round(lat, 7),
                             grp=tag, map_class=cls, source='成品图抽样(%s)' % (tile or '?')))
    for ds in cache.values():
        ds.close()
    with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['point_id', 'lon', 'lat', 'grp', 'map_class', 'source'])
        w.writeheader()
        w.writerows(rows)
    import collections
    emit('写出 %s：%d 点 %s' % (OUT, len(rows), dict(collections.Counter(r['grp'] for r in rows))))
    emit('S1/S2 成品类分布：%s' % dict(collections.Counter(r['map_class'] for r in rows if r['grp'] != 'A')))


if __name__ == '__main__':
    main()
