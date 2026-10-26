# -*- coding: utf-8 -*-
"""h14_verify_yearly.py — 逐年补样层核验（完备性/分布/跨年一致性/双路证据/外部产品一致性）

* 1) 逐年：点数、类码、AEF 缺失、区域分布、省分布
* 2) 跨年：全 8 年 keep 的比例、翻档次数分布
* 3) 双路证据：对 2018 与 2021 各随机 150 点，独立最小构建重取（DW bare/water + S2 夏 NDVI），max|Δ| 须 0
* 4) 外部产品一致性（非精度真值，仅旁证）：WorldCover(2021) 与 裸地/稀疏 的对应率
* 输出：data/yearly/verify_yearly.json + reports/yearly_verify.md
* 用法：python h14_verify_yearly.py --acct zzuoavsm
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
FEATS = ['A%02d' % i for i in range(64)]
YEARS = list(range(2017, 2025))


def jload(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return {}


def double_arm(year, n, acct):
    import ee
    sys.path.insert(0, r'F:/lc_work/v31_exp/code')
    import v31_common as VC
    k = pd.read_parquet(os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % year))
    s = k.sample(min(n, len(k)), random_state=17)
    ev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(
        BASE, 'data', 'evidence_yearly', str(year), '*.parquet')))], ignore_index=True).drop_duplicates('pid')
    s = s.merge(ev, on='pid', suffixes=('', '_store'), how='left')
    ee_mod, _ = VC.ctx(acct)
    dw = (ee_mod.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
          .select(['bare', 'water']).mean().unmask(-1))
    s2 = ee_mod.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('%d-06-01' % year, '%d-09-01' % year) \
        .filter(ee_mod.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
        .map(lambda im: im.updateMask(im.select('SCL').neq(3).And(im.select('SCL').neq(8))
                                      .And(im.select('SCL').neq(9)))).median()
    nd = s2.normalizedDifference(['B8', 'B4']).multiply(1000).rename('s2_ndvi_xia').toFloat().unmask(-9999)
    img = ee_mod.Image.cat([dw, nd]).toFloat()
    feats = [ee_mod.Feature(ee_mod.Geometry.Point([float(r.lon), float(r.lat)]), {'pid': r.pid})
             for r in s[['pid', 'lon', 'lat']].itertuples(index=False)]
    res = img.sampleRegions(collection=ee_mod.FeatureCollection(feats), scale=30,
                            tileScale=4, geometries=False).getInfo()
    got = pd.DataFrame([f['properties'] for f in res['features']])
    m = s.merge(got, on='pid', suffixes=('', '_new'), how='left')
    out = {}
    for b in ['bare', 'water', 's2_ndvi_xia']:
        if b not in m.columns or b + '_new' not in m.columns:
            continue
        a = pd.to_numeric(m[b], errors='coerce'); c = pd.to_numeric(m[b + '_new'], errors='coerce')
        ok = a.notna() & c.notna() & (a > -9000) & (c > -9000)
        d = (a[ok] - c[ok]).abs()
        out[b] = {'n': int(ok.sum()), 'max_abs_diff': float(d.max()) if len(d) else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--skip-gee', action='store_true')
    a = ap.parse_args()
    res = {'years': {}, 'cross_year': {}}
    first = None
    for y in YEARS:
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        if not os.path.isfile(fp):
            continue
        d = pd.read_parquet(fp)
        first = first if first is not None else d
        wc = d.get('wc')
        res['years'][y] = {
            'n': int(len(d)), 'sparse': int((d.cl == 13).sum()), 'bare': int((d.cl == 22).sum()),
            'aef_missing': int(d[FEATS].isna().all(axis=1).sum()),
            'cl_ok': bool(set(d.cl.unique()) <= {13, 22}),
            'bbox': [round(float(d.lon.min()), 2), round(float(d.lat.min()), 2),
                     round(float(d.lon.max()), 2), round(float(d.lat.max()), 2)]}
    # 跨年一致性
    panel = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'yearly_panel.parquet'))
    piv = panel.pivot_table(index='pid', columns='year', values='status', aggfunc='first')
    n_keep = (piv == 'keep').sum(axis=1)
    res['cross_year'] = {'n_points': int(len(piv)),
                         'keep_all_8': int((n_keep == len(piv.columns)).sum()),
                         'keep_ge6': int((n_keep >= 6).sum()),
                         'keep_mean': float(n_keep.mean()),
                         'flip_counts': {str(int(k)): int(v) for k, v in n_keep.value_counts().sort_index().items()}}
    # 区域分布（按年均值）
    reg = {}
    for nm, bb in {'NE': (113, 38, 135, 54), 'East': (107, 18, 125, 38.5)}.items():
        sp = ba = 0
        for y in YEARS:
            fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
            if not os.path.isfile(fp):
                continue
            d = pd.read_parquet(fp, columns=['lon', 'lat', 'cl'])
            m = (d.lon >= bb[0]) & (d.lon < bb[2]) & (d.lat >= bb[1]) & (d.lat < bb[3])
            sp += int(((d.cl == 13) & m).sum()); ba += int(((d.cl == 22) & m).sum())
        reg[nm] = {'sparse_point_years': sp, 'bare_point_years': ba}
    res['region_point_years'] = reg
    # 外部产品一致性（WorldCover 2021）
    if first is not None and 'wc' in first.columns:
        w = first
        res['wc_check'] = {
            'sparse_wc_20_30_60_pct': round(float(
                w[w.cl == 13].wc.isin([20, 30, 60]).mean() * 100), 1),
            'bare_wc_60_pct': round(float((w[w.cl == 22].wc == 60).mean() * 100), 1),
            'note': 'WorldCover v200(2021) 仅作旁证；其 60=裸地/稀疏合并类'}
    # 双路证据
    if not a.skip_gee:
        res['double_arm'] = {}
        for y in (2018, 2021):
            if os.path.isfile(os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)):
                res['double_arm'][y] = double_arm(y, 150, a.acct)
                print('%d 双路：%s' % (y, res['double_arm'][y]), flush=True)
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)
    json.dump(res, open(os.path.join(BASE, 'data', 'yearly', 'verify_yearly.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    lines = ['# 逐年补样层核验（2017–2024）\n', '```json',
             json.dumps(res, ensure_ascii=False, indent=1), '```']
    open(os.path.join(BASE, 'reports', 'yearly_verify.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print(json.dumps({k: res[k] for k in res if k != 'years'}, ensure_ascii=False)[:1200], flush=True)


if __name__ == '__main__':
    main()
