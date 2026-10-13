# -*- coding: utf-8 -*-
"""o40_hainan_validate.py — 官方海南橡胶地块 × v6 候选分层核验（补落盘）

* 输入：D6 乔灌园地v6_橡胶样本.parquet（prov=tier/gem）+ hainan_rubber_parcels.parquet（50,108 图斑）
* 规则源：海南候选（prov=海南省）逐点判"落入任一官方图斑内"→ 按 tier 与 gem 分层命中率
* 输出：results/orchard/hainan_official_validate.json
* 用法：python o40_hainan_validate.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
D6 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v6_含苹果'


def main():
    import geopandas as gpd
    from shapely import make_valid
    rub = pd.read_parquet(os.path.join(D6, '乔灌园地v6_橡胶样本.parquet'),
                          columns=['lon', 'lat', 'prov', 'tier', 'gem'])
    hn = rub[rub.prov == '海南省'].reset_index(drop=True)
    print('海南候选 %d（全库 %d）' % (len(hn), len(rub)))
    GDB = r'F:/lc_work/外部样本/rare_class_dl/Hainan_rubber_2020/海南橡胶林分布2020年底.gdb'
    g = gpd.read_file(GDB, layer='海南橡胶林分布').to_crs('EPSG:4326')
    g['geometry'] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[g.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    print('官方图斑 %d' % len(g))
    pts = gpd.GeoDataFrame(hn, geometry=gpd.points_from_xy(hn.lon, hn.lat), crs='EPSG:4326')
    j = gpd.sjoin(pts, g[['geometry']], predicate='within', how='left')
    hn['hit'] = j.index_right.notna().to_numpy()
    print('总命中 %d/%d = %.1f%%' % (hn.hit.sum(), len(hn), 100 * hn.hit.mean()))
    out = {'n_cand_hn': int(len(hn)), 'n_parcels': int(len(g)),
           'hit_all': round(float(hn.hit.mean()), 4),
           'by_tier': {}, 'by_gem': {}}
    for t, gr in hn.groupby('tier'):
        out['by_tier'][t] = {'n': int(len(gr)), 'hit': round(float(gr.hit.mean()), 4)}
        print('  %-8s n=%5d 命中=%.1f%%' % (t, len(gr), 100 * gr.hit.mean()))
    gg = hn[hn.gem.notna()].copy()
    gg['gem_b'] = pd.cut(gg.gem, [0, 1, 2, 3], labels=['1', '2', '>=3'])
    for b, gr in gg.groupby('gem_b', observed=True):
        out['by_gem'][str(b)] = {'n': int(len(gr)), 'hit': round(float(gr.hit.mean()), 4)}
        print('  gem=%-4s n=%5d 命中=%.1f%%' % (b, len(gr), 100 * gr.hit.mean()))
    os.makedirs(os.path.join(WORK, 'results', 'orchard'), exist_ok=True)
    with open(os.path.join(WORK, 'results', 'orchard', 'hainan_official_validate.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('→ results/orchard/hainan_official_validate.json')


if __name__ == '__main__':
    main()
