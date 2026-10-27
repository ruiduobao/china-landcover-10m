# -*- coding: utf-8 -*-
"""s1_anchors.py — 光伏电站候选锚点抽样（阶段 S1）

* 输入：D:/地物样本数据_2026-10-27/04_光伏/中国子集/SolarPV_China.gpkg（20,592 面）
* 规则源：F:/lc_work/new4_samples/solarpv/criteria_solarpv.json（§s1_sampling，预注册）
* 门槛：
    - 面积 < 10,000 m² 剔除；1–10 ha 内缩 -20 m，>10 ha 内缩 -30 m（空几何按 10/5 m 逐级放松）
    - 1–10 ha 每站 1 点；10–100 ha 按子带 2/3/4 点（≥300 m）；>100 ha 最多 15 点（≥500 m）
    - 全国球面 ≥250 m 贪心去重（大面积多边形优先占位）
    - 与大棚互斥：落 OSM 大棚多边形内 / China_GH_2022==1 的点剔除；PV∩大棚 >20% 标 pv_sub='agro_pv'
    - 面积 <2 ha 的点占比压到 ≤10%（固定随机种子 20261027）
* 输出：F:/lc_work/new4_samples/solarpv/data/pv_anchors.parquet
        F:/lc_work/new4_samples/solarpv/data/s1_dropped.csv（逐条剔除原因）
        F:/lc_work/new4_samples/solarpv/data/s1_stats.json
* 用法：python s1_anchors.py [--limit 500]
"""
import argparse
import json
import math
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

SRC = r'D:/地物样本数据_2026-10-27/04_光伏/中国子集/SolarPV_China.gpkg'
GH_OSM = r'D:/地物样本数据_2026-10-27/01_设施农业大棚/OSM_greenhouse/OSM_greenhouse_China.gpkg'
GH_TIF = (r'D:/地物样本数据_2026-10-27/01_设施农业大棚/China_GH_2010_2016_2022_Figshare28559747'
          r'/China_Agricultural_Greenhouse/China_Agricultural_Greenhouse_2022.tif')
WORK = r'F:/lc_work/new4_samples/solarpv'
DATA = os.path.join(WORK, 'data')
R_EARTH = 6371008.8
SEED = 20261027


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


# ---------------- 候选点生成 ----------------
def grid_candidates(geom, step):
    """多边形包围盒内 step 间距网格中落在多边形内的点。"""
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2))
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    inside = shapely.contains_xy(geom, gx, gy)
    return np.column_stack([gx[inside], gy[inside]])


def farthest_sample(cand, n, spacing, start_xy):
    """最远点贪心：从最近多边形容器的候选开始，每次取距已选集最远者，并 enforce 最小间距。"""
    if len(cand) == 0:
        return np.empty((0, 2))
    d0 = np.hypot(cand[:, 0] - start_xy[0], cand[:, 1] - start_xy[1])
    if n <= 1:
        return cand[[int(np.argmin(d0))]]
    d = d0.copy()
    picked = []
    cur = int(np.argmin(d))
    for _ in range(n):
        picked.append(cur)
        dist = np.hypot(cand[:, 0] - cand[cur, 0], cand[:, 1] - cand[cur, 1])
        d = np.minimum(d, dist)
        d[picked] = -1.0
        nxt = int(np.argmax(d))
        if d[nxt] < spacing:
            break
        cur = nxt
    return cand[picked]


def inset_geom(geom, buf_m):
    """负缓冲，空几何按 10/5 m 逐级放松；返回 (geom, relaxed_m)"""
    for b in [buf_m, 10, 5]:
        g = geom.buffer(-abs(b))
        if not g.is_empty and g.area > 200:
            return g, abs(b)
    return geom.representative_point().buffer(3), 3


def n_target(area_m2):
    ha = area_m2 / 10000.0
    if area_m2 < 100000:      # <10 ha
        return 1, 0
    if area_m2 < 1e6:         # 10-100 ha
        if ha < 20:
            return 2, 300
        if ha < 50:
            return 3, 300
        return 4, 300
    return 15, 500            # >100 ha


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='调试：只跑前 N 个多边形')
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()

    emit('读取源多边形 %s' % SRC)
    g = gpd.read_file(SRC)
    g['sid'] = np.arange(1, len(g) + 1)
    g['area_6933'] = g.to_crs('EPSG:6933').area.values
    dev = np.abs(g['area_6933'] / (g['area_km2'] * 1e6) - 1)
    emit('  面积口径核验：attr vs EPSG:6933 相对差中位 %.4f%%、99 分位 %.3f%%'
         % (np.median(dev) * 100, np.percentile(dev, 99) * 100))

    drop = []
    m = g['area_6933'] >= 10000
    drop.append(pd.DataFrame({'sid': g.loc[~m, 'sid'], 'reason': 'AREA_LT_1HA'}))
    g = g[m].reset_index(drop=True)
    emit('  ≥1 ha：%d 面（剔除 %d）' % (len(g), int((~m).sum())))
    if a.limit:
        g = g.iloc[:a.limit].reset_index(drop=True)

    # 修复无效几何
    bad = ~g.is_valid
    if bad.any():
        g.loc[bad, 'geometry'] = shapely.make_valid(g.loc[bad, 'geometry'].values)
        emit('  修复无效几何 %d 面' % int(bad.sum()))

    g['pv_type'] = np.where(g['area_6933'] >= 20000, 'centralized', 'rooftop')
    _b = g.geometry.bounds
    g['zone'] = np.clip(((np.floor((_b['minx'] + _b['maxx']) / 2) + 180) // 6 + 1).astype(int), 43, 53)

    rows = []
    for zone, sub in g.groupby('zone'):
        utm = 'EPSG:326%d' % int(zone)
        s = sub.to_crs(utm)
        emits = 0
        for r in s.itertuples():
            area = r.area_6933
            buf = 20 if area < 100000 else 30
            ig, relaxed = inset_geom(r.geometry, buf)
            n, spacing = n_target(area)
            cand = grid_candidates(ig, 150.0)
            if len(cand) == 0:
                pt = ig.representative_point()
                picked = np.array([[pt.x, pt.y]])
            else:
                c = cand if len(cand) <= 3000 else cand[:: max(1, len(cand) // 3000)]
                start = np.array([ig.representative_point().x, ig.representative_point().y])
                picked = farthest_sample(c, n, spacing, start)
            for j, (x, y) in enumerate(picked):
                rows.append((r.sid, r.pv_type, r.layer, r.constructi, r.construc_1,
                             r.landcover_, r.area_6933, relaxed, int(zone), x, y, j))
            emits += 1
            if emits % 2000 == 0:
                emit('    zone %d：%d/%d 面' % (zone, emits, len(s)))
        emit('  zone %d 完成（%d 面）' % (zone, len(s)))

    cols = ['sid', 'pv_type', 'layer', 'raw_yr', 'construc_1', 'landcover_',
            'area_6933', 'inset_relaxed_m', 'zone', 'x_utm', 'y_utm', 'pidx']
    pts = pd.DataFrame(rows, columns=cols)
    emit('内缩抽点：%d 点（%d 面 / %d 面有输出）' % (len(pts), pts['sid'].nunique(), len(g)))

    # UTM → WGS84
    out = []
    for zone, sub in pts.groupby('zone'):
        gg = gpd.GeoDataFrame(sub.copy(),
                              geometry=gpd.points_from_xy(sub['x_utm'], sub['y_utm']),
                              crs='EPSG:326%d' % int(zone)).to_crs('EPSG:4326')
        gg['lon'] = gg.geometry.x.values
        gg['lat'] = gg.geometry.y.values
        out.append(gg.drop(columns='geometry'))
    pts = pd.concat(out, ignore_index=True)

    # ---------------- 大棚互斥 ----------------
    emit('大棚互斥检查（OSM 多边形 + China_GH_2022 栅格）')
    gh = gpd.read_file(GH_OSM)
    gh = gh[gh.geometry.notna() & ~gh.geometry.is_empty]
    pv_gdf = gpd.GeoDataFrame(pts.copy(), geometry=gpd.points_from_xy(pts['lon'], pts['lat']),
                              crs='EPSG:4326')
    gh_union = shapely.union_all(gh.geometry.values)
    in_gh = shapely.contains_xy(gh_union, pts['lon'].values, pts['lat'].values)
    emit('  落在 OSM 大棚多边形内：%d 点' % int(in_gh.sum()))

    # China GH 2022 栅格
    in_gh_r = np.zeros(len(pts), dtype=bool)
    gh_val = np.full(len(pts), np.nan, dtype='float32')
    try:
        import rasterio
        from rasterio.sample import sample_gen
        with rasterio.open(GH_TIF) as ds:
            vals = np.array([v[0] for v in sample_gen(ds, zip(pts['lon'].values, pts['lat'].values))],
                            dtype='float32')
        gh_val = vals
        # 该产品是 0/1 二值且 nodata=0：正类只看 ==1，0 既表示非大棚也表示无数据（保守不剔除）
        in_gh_r = (vals == 1)
        emit('  China_GH_2022 采样：==1 %d 点（nodata=%s）' % (int(in_gh_r.sum()), nod))
    except Exception as e:
        emit('  !! China_GH_2022 栅格采样失败（不阻断）：%s' % str(e)[:150])

    pts['in_gh_osm'] = in_gh
    pts['gh2022'] = gh_val
    mutex = in_gh | in_gh_r
    drop.append(pd.DataFrame({'sid': pts.loc[mutex, 'sid'].values,
                              'lon': pts.loc[mutex, 'lon'].values,
                              'lat': pts.loc[mutex, 'lat'].values,
                              'reason': 'GREENHOUSE_OVERLAP'}))
    pts = pts[~mutex].reset_index(drop=True)
    emit('  大棚互斥后剩 %d 点（剔除 %d）' % (len(pts), int(mutex.sum())))

    # pv_sub='agro_pv'：PV 多边形与 OSM 大棚多边形相交面积比 >20%
    sid_geom = g.set_index('sid').geometry
    gh_4326 = gh_union
    agro = {}
    cand_sids = pts['sid'].unique()
    for sid in cand_sids:
        pg = sid_geom.loc[sid]
        try:
            inter = pg.intersection(gh_4326).area
            if inter > 0 and (inter / pg.area) > 0.20:
                agro[sid] = True
        except Exception:
            pass
    pts['pv_sub'] = pts['sid'].map(lambda s: 'agro_pv' if agro.get(s) else '')
    emit('  agro_pv 站数 %d（点数 %d）' % (len(agro), int((pts['pv_sub'] == 'agro_pv').sum())))

    # ---------------- 全国 250 m 贪心去重（大面积优先） ----------------
    emit('全国 ≥250 m 贪心去重')
    pts = pts.sort_values(['area_6933', 'sid', 'pidx'], ascending=[False, True, True]).reset_index(drop=True)
    cell = 250.0
    grid = {}
    keep = np.zeros(len(pts), dtype=bool)
    lons = pts['lon'].values
    lats = pts['lat'].values
    lat_m = 111320.0
    for i in range(len(pts)):
        la = lats[i]
        lo = lons[i]
        ky = int(math.floor(la * lat_m / cell))
        kx = int(math.floor(lo * lat_m * math.cos(math.radians(la)) / cell))
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in grid.get((kx + dx, ky + dy), ()):
                    # haversine
                    dlat = math.radians(la - lats[j])
                    dlon = math.radians(lo - lons[j])
                    aa = (math.sin(dlat / 2) ** 2
                          + math.cos(math.radians(la)) * math.cos(math.radians(lats[j])) * math.sin(dlon / 2) ** 2)
                    if 2 * R_EARTH * math.asin(min(1.0, math.sqrt(aa))) < 250.0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep[i] = True
            grid.setdefault((kx, ky), []).append(i)
    drop.append(pd.DataFrame({'sid': pts.loc[~keep, 'sid'].values,
                              'lon': pts.loc[~keep, 'lon'].values,
                              'lat': pts.loc[~keep, 'lat'].values,
                              'reason': 'DEDUP_250M'}))
    pts = pts[keep].reset_index(drop=True)
    emit('  去重后 %d 点（剔除 %d）' % (len(pts), int((~keep).sum())))

    # ---------------- 集中式占比 ≥90% ----------------
    n_roof = int((pts['pv_type'] == 'rooftop').sum())
    total = len(pts)
    share = n_roof / total if total else 0
    emit('集中式占比检查：rooftop %d/%d = %.2f%%' % (n_roof, total, share * 100))
    if share > 0.10:
        cap = int((total - n_roof) / 9.0)   # roof/(total-roof+roof) <= 0.10
        rng = np.random.default_rng(SEED)
        roof_idx = pts.index[pts['pv_type'] == 'rooftop'].to_numpy()
        if len(roof_idx) > cap:
            drop_idx = rng.choice(roof_idx, size=len(roof_idx) - cap, replace=False)
            drop.append(pd.DataFrame({'sid': pts.loc[drop_idx, 'sid'].values,
                                      'lon': pts.loc[drop_idx, 'lon'].values,
                                      'lat': pts.loc[drop_idx, 'lat'].values,
                                      'reason': 'ROOFTOP_SHARE_CAP'}))
            pts = pts.drop(index=drop_idx).reset_index(drop=True)
        emit('  屋顶点下采样到 %d（总 %d，集中式 %.2f%%）'
             % (len(pts) - int((pts['pv_type'] == 'centralized').sum()),
                len(pts), 100 * (pts['pv_type'] == 'centralized').mean()))

    # ---------------- 收尾 ----------------
    pts = pts.sort_values(['sid', 'pidx']).reset_index(drop=True)
    pts['pid'] = ['PV_%07d' % (i + 1) for i in range(len(pts))]
    pts['src'] = np.where(pts['layer'].astype(str).str.startswith('osm'), 'osm_solarpv', 'ms_solarpv_2024q2')
    keep_cols = ['pid', 'sid', 'lon', 'lat', 'area_6933', 'pv_type', 'pv_sub', 'src',
                 'raw_yr', 'construc_1', 'landcover_', 'inset_relaxed_m', 'zone']
    pts[keep_cols].to_parquet(os.path.join(DATA, 'pv_anchors.parquet'), index=False)

    dd = pd.concat([x for x in drop if len(x)], ignore_index=True) if drop else pd.DataFrame()
    dd.to_csv(os.path.join(DATA, 's1_dropped.csv'), index=False, encoding='utf-8-sig')

    stats = {
        'source_polygons': int(len(g)) if not a.limit else int(a.limit),
        'anchors': int(len(pts)),
        'sites_with_points': int(pts['sid'].nunique()),
        'centralized_share': float((pts['pv_type'] == 'centralized').mean()),
        'agro_pv_points': int((pts['pv_sub'] == 'agro_pv').sum()),
        'by_type': pts['pv_type'].value_counts().to_dict(),
        'by_src': pts['src'].value_counts().to_dict(),
        'by_area_band_ha': {
            '<2': int((pts['area_6933'] < 20000).sum()),
            '2-10': int(((pts['area_6933'] >= 20000) & (pts['area_6933'] < 1e5)).sum()),
            '10-100': int(((pts['area_6933'] >= 1e5) & (pts['area_6933'] < 1e6)).sum()),
            '>100': int((pts['area_6933'] >= 1e6).sum()),
        },
        'dropped_by_reason': dd['reason'].value_counts().to_dict() if len(dd) else {},
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(os.path.join(DATA, 's1_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    emit('完成：%s' % json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
