# -*- coding: utf-8 -*-
"""o31_aomc_extract.py — AOMC 苹果园候选提取（13 省 30m GeoTIFF，Albers→WGS84）

* 输入：F:/lc_work/外部样本/rare_class_dl/AOMC_apple/AOMC_30_*_2020.tif（2020 年取用）
* 规则源：value==1（苹果园；**逐瓦片先统计全值域并报告**，若见其它值一并记录）；
          抽稀读 1/6（≈180m 网格）→ 取点 → 中国国界过滤 → ≥500m 间距 → 省级配额
* 门槛：全国上限 30,000；报告每省点数与面积估算
* 输出：data/orchard/aomc_apple_points.parquet + results/orchard/aomc_stats.json
* 用法：python o31_aomc_extract.py
"""
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely import make_valid, contains_xy
from shapely.ops import unary_union

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
SRC = r'F:/lc_work/外部样本/rare_class_dl/AOMC_apple'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
DEC = 6
CAP = 30000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    fs = sorted(glob.glob(os.path.join(SRC, 'AOMC_30_*_2020.tif')))
    emit('2020 瓦片 %d 个' % len(fs))
    parts = []
    vstats = {}
    for fp in fs:
        name = os.path.basename(fp)
        try:
            with rasterio.open(fp) as ds:
                a = ds.read(1, out_shape=(max(1, ds.height // DEC), max(1, ds.width // DEC)))
                v, c = np.unique(a, return_counts=True)
                vstats[name] = {int(k): int(n) for k, n in zip(v, c)}
                m = a == 1
                if not m.any():
                    emit('  %s: 无 value==1' % name)
                    continue
                rr, cc = np.nonzero(m)
                tr = ds.transform * rasterio.Affine.scale(ds.width / a.shape[1], ds.height / a.shape[0])
                xs, ys = rasterio.transform.xy(tr, rr, cc, offset='center')
                tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
                lon, lat = tf.transform(np.asarray(xs), np.asarray(ys))
                parts.append(pd.DataFrame({'lon': lon, 'lat': lat, 'tile': name}))
                emit('  %s: px==1 %d（值域 %s）' % (name, int(m.sum()),
                                                  {k: int(n) for k, n in list(zip(v, c))[:6]}))
        except Exception as e:
            emit('  %s 失败: %s' % (name, str(e)[:100]))
    d = pd.concat(parts, ignore_index=True)
    emit('原始命中 %d' % len(d))
    gj = json.load(open(BND, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    china = unary_union(geoms)
    inb = d[(d.lon >= 73) & (d.lon <= 136) & (d.lat >= 18) & (d.lat <= 54)].reset_index(drop=True)
    keep = contains_xy(china, inb.lon.to_numpy(), inb.lat.to_numpy())
    cn = inb[keep].reset_index(drop=True)
    emit('中国境内 %d' % len(cn))
    # ≥500m 间距
    rng = np.random.default_rng(20261013)
    idx = rng.permutation(len(cn))[: min(len(cn), CAP * 3)]
    sel, pts = [], []
    R = 500 / 111000.0
    arr = cn[['lon', 'lat']].to_numpy()
    for i in idx:
        lo, la = arr[i]
        if pts:
            aa = np.asarray(pts)
            if np.hypot((aa[:, 0] - lo) * np.cos(np.radians(la)), aa[:, 1] - la).min() < R:
                continue
        sel.append(i); pts.append((lo, la))
        if len(sel) >= CAP:
            break
    out = cn.iloc[sel][['lon', 'lat', 'tile']].reset_index(drop=True)
    try:
        import k19_provattrib as PA
        out['prov'] = PA.attribute(out)
        prov = out.prov.value_counts().to_dict()
    except Exception as e:
        prov = {'err': str(e)[:60]}
    out.to_parquet(os.path.join(ORD, 'aomc_apple_points.parquet'), index=False)
    import v31_common as VC
    VC.jsave({'n_raw_cn': int(len(cn)), 'n_selected': int(len(out)),
              'area_est_km2': round(len(cn) * (180 * 180) / 1e6, 0), 'prov': prov, 'value_stats': vstats},
             os.path.join(WORK, 'results', 'orchard', 'aomc_stats.json'))
    emit('选定 %d；省分布 %s' % (len(out), {k: v for k, v in list(prov.items())[:10]}))
    print('→ data/orchard/aomc_apple_points.parquet')


if __name__ == '__main__':
    main()
