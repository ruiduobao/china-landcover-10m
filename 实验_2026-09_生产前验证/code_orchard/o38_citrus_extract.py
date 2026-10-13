# -*- coding: utf-8 -*-
"""o38_citrus_extract.py — 中国柑橘园图 2020 候选点提取（抽稀读 + 国界 + 间距）

* 输入：F:/lc_work/外部样本/rare_class_dl/Citrus_2020/citrusmap202010m.tif（276947×238415, int32, 1=柑橘, nodata=-2^31）
* 规则源：抽稀读 1/20（≈215m 网格）→ value==1 → 中国国界 → ≥300m 间距 → 上限 30,000
* 输出：data/orchard/citrus_2020_cn_points.parquet + results/orchard/citrus_stats.json
* 用法：python o38_citrus_extract.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform as rt
from shapely.geometry import shape
from shapely import make_valid, contains_xy
from shapely.ops import unary_union

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
TIF = r'F:/lc_work/外部样本/rare_class_dl/Citrus_2020/citrusmap202010m.tif'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
DEC = 20
CAP = 30000


def main():
    t0 = time.time()
    with rasterio.open(TIF) as ds:
        a = ds.read(1, out_shape=(ds.height // DEC, ds.width // DEC))
        tr = ds.transform * rt.Affine.scale(DEC, DEC)
        rr, cc = np.nonzero(a == 1)
        xs, ys = rt.xy(tr, rr, cc, offset='center')
    lon = np.asarray(xs); lat = np.asarray(ys)
    print('citrus==1 抽稀命中 %d（%.0fs）' % (len(lon), time.time() - t0))
    inb = (lon >= 73) & (lon <= 136) & (lat >= 18) & (lat <= 54)
    lon, lat = lon[inb], lat[inb]
    gj = json.load(open(BND, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    keep = contains_xy(unary_union(geoms), lon, lat)
    lon, lat = lon[keep], lat[keep]
    print('中国境内 %d 点（面积估算 %.0f km² @1/20）' % (len(lon), len(lon) * 0.04))
    rng = np.random.default_rng(20261013)
    idx = rng.permutation(len(lon))[: min(len(lon), CAP * 4)]
    sel, pts = [], []
    R = 300 / 111000.0
    for i in idx:
        lo, la = lon[i], lat[i]
        if pts:
            arr = np.asarray(pts)
            if np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la).min() < R:
                continue
        sel.append(i); pts.append((lo, la))
        if len(sel) >= CAP:
            break
    out = pd.DataFrame({'lon': lon[sel], 'lat': lat[sel], 'src': 'citrusmap2020'})
    try:
        import k19_provattrib as PA
        out['prov'] = PA.attribute(out)
        prov = out.prov.value_counts().head(12).to_dict()
    except Exception as e:
        prov = {'err': str(e)[:60]}
    out.to_parquet(os.path.join(ORD, 'citrus_2020_cn_points.parquet'), index=False)
    import v31_common as VC
    VC.jsave({'n_cn': int(len(lon)), 'n_selected': int(len(out)), 'prov': prov},
             os.path.join(WORK, 'results', 'orchard', 'citrus_stats.json'))
    print('选定 %d；省 top: %s' % (len(out), prov))
    print('→ citrus_2020_cn_points.parquet')


if __name__ == '__main__':
    main()
