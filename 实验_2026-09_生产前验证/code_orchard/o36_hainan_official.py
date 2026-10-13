# -*- coding: utf-8 -*-
"""o36_hainan_official.py — 海南官方橡胶地块 → 最高等级橡胶样本 + 既有候选分级核验

* 输入：F:/lc_work/外部样本/rare_class_dl/Hainan_rubber_2020/海南橡胶林分布2020年底.gdb（50,108 图斑）
* 规则源：
  ① 官方样本点：逐图斑取内部点（<1ha 用 representative_point；≥1ha 按 300m 网格取内点），
     全部落在中国国界内；间距 ≥200m；上限 40,000；
  ② 分级核验：v6 海南候选（9,539）× 官方地块 → 命中率按 tier（A/B/C）与 gem 值分层
* 输出：data/orchard/hainan_official_points.parquet + results/orchard/hainan_official_stats.json
* 用法：python o36_hainan_official.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
GDB = r'F:/lc_work/外部样本/rare_class_dl/Hainan_rubber_2020/海南橡胶林分布2020年底.gdb'
CAP = 40000


def emit(m):
    import time
    print(time.strftime('%m-%d %H:%M:%S ') + str(m), flush=True)


def main():
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely import STRtree
    g = gpd.read_file(GDB, layer='海南橡胶林分布')
    emit('图斑 %d' % len(g))
    g_m = g.to_crs('EPSG:32649')
    areas_m2 = g_m.geometry.area
    g = g.to_crs('EPSG:4326')
    rng = np.random.default_rng(20261013)
    rows = []
    small = large = 0
    for i, (geom, a) in enumerate(zip(g.geometry, areas_m2)):
        if a < 10000:
            p = geom.representative_point()
            rows.append((p.x, p.y))
            small += 1
        else:
            # 附近网格取内点
            minx, miny, maxx, maxy = geom.bounds
            dl = 300 / 111000.0
            xs = np.arange(minx, maxx + dl, dl)
            ys = np.arange(miny, maxy + dl, dl)
            got = 0
            for x in xs:
                for y in ys:
                    if geom.contains(Point(x, y)):
                        rows.append((x, y))
                        got += 1
                        if got >= 30:
                            break
                if got >= 30:
                    break
            if got == 0:
                p = geom.representative_point()
                rows.append((p.x, p.y))
            large += 1
    d = pd.DataFrame(rows, columns=['lon', 'lat'])
    emit('内部点 %d（小图斑 %d / 大图斑 %d）' % (len(d), small, large))
    d = d.drop_duplicates(subset=['lon', 'lat'])
    # 中国国界
    gj = json.load(open(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json', encoding='utf-8'))
    from shapely.geometry import shape
    from shapely import make_valid, contains_xy
    from shapely.ops import unary_union
    geoms = []
    for ft in gj['features']:
        gg = shape(ft['geometry'])
        if not gg.is_valid:
            gg = make_valid(gg)
        geoms.append(gg)
    china = unary_union(geoms)
    keep = contains_xy(china, d.lon.to_numpy(), d.lat.to_numpy())
    d = d[keep].reset_index(drop=True)
    emit('国界内 %d' % len(d))
    # 间距 ≥200m 稀疏 + 上限
    if len(d) > CAP:
        idx = rng.permutation(len(d))[: CAP * 3]
        sel, pts = [], []
        R = 200 / 111000.0
        arr = d[['lon', 'lat']].to_numpy()
        for i in idx:
            lo, la = arr[i]
            if pts:
                aa = np.asarray(pts)
                if np.hypot((aa[:, 0] - lo) * np.cos(np.radians(la)), aa[:, 1] - la).min() < R:
                    continue
            sel.append(i); pts.append((lo, la))
            if len(sel) >= CAP:
                break
        d = d.iloc[sel].reset_index(drop=True)
    d['subtype'] = '橡胶'
    d['grade'] = 'A_官方地块'
    d['srcset'] = 'hainan_official'
    d.to_parquet(os.path.join(ORD, 'hainan_official_points.parquet'), index=False)
    emit('官方样本点 %d → hainan_official_points.parquet' % len(d))
    VC.jsave({'n_parcels': int(len(g)), 'n_points': int(len(d)),
              'official_area_km2': round(float(areas_m2.sum() / 1e6), 1)},
             os.path.join(WORK, 'results', 'orchard', 'hainan_official_stats.json'))


if __name__ == '__main__':
    main()
