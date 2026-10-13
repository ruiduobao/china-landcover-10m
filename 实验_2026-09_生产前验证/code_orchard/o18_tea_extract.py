# -*- coding: utf-8 -*-
"""o18_tea_extract.py — TeaMap 2022 中国茶园候选点提取（10m tif，抽稀读，本地）

* 输入：F:/lc_work/外部样本/rare_class_dl/TeaMap_2022_China/CAS.TEA-MAP.CHN.tif（287290×215533, uint8, 1=茶园）
* 规则源：抽稀读（out_shape 1/20 ≈ 200m 网格）→ tea==1 像元取点 → 中国国界过滤 → 间距 ≥300m
* 门槛：全中国上限 30,000 点；报告每省点数与面积估算
* 输出：data/orchard/tea_2022_cn_points.parquet + results/orchard/tea_cn_stats.json
* 用法：python o18_tea_extract.py
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
from shapely.geometry import shape
from shapely import make_valid, contains_xy
from shapely.ops import unary_union

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
TEA = r'F:/lc_work/外部样本/rare_class_dl/TeaMap_2022_China/CAS.TEA-MAP.CHN.tif'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
DEC = 20          # 抽稀倍数（10m → 200m 网格）
CAP = 30000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    t0 = time.time()
    os.makedirs(ORD, exist_ok=True)
    emit('抽稀读 TeaMap（1/%d）…' % DEC)
    with rasterio.open(TEA) as ds:
        a = ds.read(1, out_shape=(ds.height // DEC, ds.width // DEC))
        tr = ds.transform * rasterio.Affine.scale(DEC, DEC)
        rr, cc = np.nonzero(a == 1)
        emit('tea==1 抽稀像元 %d（%.0fs）' % (len(rr), time.time() - t0))
        xs, ys = rasterio.transform.xy(tr, rr, cc, offset='center')
    lon = np.asarray(xs); lat = np.asarray(ys)
    inb = (lon >= 73) & (lon <= 136) & (lat >= 18) & (lat <= 54)
    lon, lat = lon[inb], lat[inb]
    emit('bbox 内 %d' % len(lon))
    gj = json.load(open(BND, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    china = unary_union(geoms)
    keep = contains_xy(china, lon, lat)
    lon, lat = lon[keep], lat[keep]
    emit('中国境内 %d 点（抽稀网格估算面积 %.0f km²）' % (len(lon), len(lon) * 0.04))
    # 间距 ≥300m 贪心
    rng = np.random.default_rng(20261013)
    idx = rng.permutation(len(lon))[:min(len(lon), CAP * 4)]
    sel, pts = [], []
    R = 300 / 111000.0
    for i in idx:
        lo, la = lon[i], lat[i]
        if pts:
            arr = np.asarray(pts)
            d = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
            if d.min() < R:
                continue
        sel.append(i); pts.append((lo, la))
        if len(sel) >= CAP:
            break
    out = pd.DataFrame({'lon': lon[sel], 'lat': lat[sel], 'src': 'teamap2022'})
    out.to_parquet(os.path.join(ORD, 'tea_2022_cn_points.parquet'), index=False)
    try:
        import k19_provattrib as PA
        out['prov'] = PA.attribute(out)
        prov = out.prov.value_counts().head(15).to_dict()
    except Exception as e:
        prov = {'err': str(e)[:60]}
    import v31_common as VC
    VC.jsave({'n_raw_cn': int(len(lon)), 'n_selected': int(len(out)), 'prov_top': prov,
              'area_est_km2': round(len(lon) * 0.04, 0)}, os.path.join(WORK, 'results', 'orchard', 'tea_cn_stats.json'))
    emit('选定 %d 点；省 top: %s' % (len(out), prov))
    print('→', os.path.join(ORD, 'tea_2022_cn_points.parquet'))


if __name__ == '__main__':
    main()
