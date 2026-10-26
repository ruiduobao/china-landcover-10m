# -*- coding: utf-8 -*-
"""h5_verify.py — 补样点独立核验（栅格回读 + 证据双路 + 统计）

* 1) 栅格回读：随机 300 点，回读 FCS10 10m 3×3 → raw 码族 + 3×3 纯（须 100%）
* 2) 证据双路：随机 200 点，用"独立最小构建"重取关键证据（NDVI夏/DW bare/water/built/WC/GSW），
     与 h2 存档值比 max|Δ|（须 0.0）
* 3) AEF 完备：A00..A63 全缺失行数（须 0）
* 输出：--outdir/verify.json + verify.md
* 用法：python h5_verify.py --final ../data/final/cand_qc.parquet --ev ../data/evidence \
        --aef ../data/aef --acct <acct> --outdir ../data/final
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
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
TILE_ROOT = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10'
FAM = {'sparse': (150, 152, 153), 'bare': (200, 201, 202)}
KEY_BANDS = ['s2_ndvi_xia', 'bare', 'water', 'built', 'wc', 'occurrence']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def raster_check(final, n=300):
    tiles = sorted(glob.glob(os.path.join(TILE_ROOT, '*', 'GLC_FCS10_2023_*.tif')))
    idx = []
    for t in tiles:
        with rasterio.open(t) as ds:
            idx.append((t, ds.bounds, ds.transform, ds.width, ds.height))
    s = final.sample(min(n, len(final)), random_state=11)
    ok_raw = ok_pure = ok_match = 0
    for _, r in s.iterrows():
        cand = [x for x in idx if x[1].left <= r.lon < x[1].right and x[1].bottom <= r.lat < x[1].top]
        if not cand:
            continue
        t, b, tr, W, H = cand[0]
        col = int((r.lon - tr.c) / tr.a); row = int((r.lat - tr.f) / tr.e)
        with rasterio.open(t) as ds:
            a = ds.read(1, window=Window(col - 1, row - 1, 3, 3))
        codes = FAM[r.fam] if 'fam' in r else (FAM['sparse'] if r['class'] == 150 else FAM['bare'])
        if np.all(np.isin(a, codes)):
            ok_pure += 1
        if a[1, 1] in codes:
            ok_raw += 1
        if int(a[1, 1]) == int(r['raw']):
            ok_match += 1
    return {'n': int(len(s)), 'raw_in_family': ok_raw, 'pure3': ok_pure,
            'raw_equals_stored': ok_match}


def evidence_double_arm(final, evdir, acct, n=200):
    import ee
    sys.path.insert(0, r'F:/lc_work/v31_exp/code')
    import v31_common as VC
    ee_mod, pid = VC.ctx(acct)
    # 独立最小构建（与 h2 不同写法：逐源单独影像、不同顺序、单独 sampleRegions 后合并两路）
    gsw = ee_mod.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(-1)
    dw = (ee_mod.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01', '2024-01-01')
          .select(['bare', 'water', 'built']).mean().unmask(-1))
    wc = ee_mod.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('wc').unmask(-1)
    s2 = ee_mod.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2023-06-01', '2023-09-01') \
        .filter(ee_mod.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
        .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                      .And(im.select('SCL').neq(9)))).median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).multiply(1000).rename('s2_ndvi_xia').toFloat().unmask(-9999)
    img = ee_mod.Image.cat([ndvi, dw, wc, gsw]).toFloat()

    ev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(evdir, 'ev_p*_b*.parquet')))],
                   ignore_index=True).drop_duplicates('pid')
    if 'pid' not in final.columns:
        final = final.copy()
        final['pid'] = [f'{r.fam}_{r.lon:.6f}_{r.lat:.6f}' for r in final.itertuples()]
    s = final.sample(min(n, len(final)), random_state=13).merge(
        ev, on='pid', how='left', suffixes=('', '_ev'))
    feats = [ee_mod.Feature(ee_mod.Geometry.Point([float(r.lon), float(r.lat)]), {'pid': r.pid})
             for r in s[['pid', 'lon', 'lat']].itertuples(index=False)]
    res = img.sampleRegions(collection=ee_mod.FeatureCollection(feats), scale=30,
                            tileScale=4, geometries=False).getInfo()
    got = pd.DataFrame([f['properties'] for f in res['features']])
    m = s.merge(got, on='pid', suffixes=('', '_new'), how='left')
    out = {}
    for b in KEY_BANDS:
        if b not in m.columns or b + '_new' not in m.columns:
            continue
        a = pd.to_numeric(m[b], errors='coerce')
        c = pd.to_numeric(m[b + '_new'], errors='coerce')
        ok = a.notna() & c.notna() & (a > -9000) & (c > -9000)
        d = (a[ok] - c[ok]).abs()
        out[b] = {'n': int(ok.sum()), 'max_abs_diff': float(d.max()) if len(d) else None,
                  'p99_abs_diff': float(d.quantile(.99)) if len(d) else None}
    return out


def aef_check(aefdir):
    fs = sorted(glob.glob(os.path.join(aefdir, 'aef_p*_b*.parquet')))
    if not fs:
        return {'blocks': 0}
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    feats = ['A%02d' % i for i in range(64)]
    miss = int(d[feats].isna().all(axis=1).sum())
    return {'blocks': len(fs), 'n': int(len(d)), 'aef_all_missing': miss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', required=True)
    ap.add_argument('--ev', required=True)
    ap.add_argument('--aef', default=None)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'final'))
    ap.add_argument('--skip-gee', action='store_true')
    a = ap.parse_args()
    d = pd.read_parquet(a.final)
    keep = d[d.status == 'keep'].reset_index(drop=True) if 'status' in d.columns else d
    emit('最终 keep 点：%d（sparse %d / bare %d）' % (
        len(keep), int((keep.fam == 'sparse').sum()), int((keep.fam == 'bare').sum())))
    res = {'raster': raster_check(keep), 'n_final': int(len(keep))}
    emit('栅格回读: %s' % res['raster'])
    if not a.skip_gee:
        res['evidence_double_arm'] = evidence_double_arm(keep, a.ev, a.acct)
        emit('证据双路: %s' % res['evidence_double_arm'])
    if a.aef:
        res['aef'] = aef_check(a.aef)
        emit('AEF: %s' % res['aef'])
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(res, open(os.path.join(a.outdir, 'verify.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ %s' % os.path.join(a.outdir, 'verify.json'))


if __name__ == '__main__':
    main()
