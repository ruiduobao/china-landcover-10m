# -*- coding: utf-8 -*-
"""f3_gap_features.py — 为缺口点集 + 甘宁参考集补齐 S2(12)+DEM(2) 特征（AEF/S1 已有）

* 输入：data/m3/gap_points.csv（255 点）、data/m3/gsnx_ref_judged.csv（304 点，已有 AEF+S1）
* 规则源：与 n11_exp_features.py 完全同口径（S2 四季 ndvi/ndwi/mndwi，SCL∉{3,8,9}+云量<60；
        DEM=COPERNICUS/DEM/GLO30 + slope）——直接 import n11 的 s2dem_image() 复用，避免口径漂移
* 门槛：逐点须 AEF64 完整；S2/DEM 缺失记 NaN 不阻断
* 输出：data/m3/gap_s2dem.csv、data/m3/gsnx_s2dem.csv（point_id + 14 列）
* 用法：python f3_gap_features.py [--set gap|gsnx|both] [--acct bx15mw]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC
import n11_exp_features as N11

M3 = os.path.join(WORK, 'data', 'm3')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def fetch(df, out_fp, acct):
    import ee
    VC.ensure_ctx(acct)
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'point_id': r.point_id})
           for r in df.itertuples()]
    want = N11.S2B + ['dem', 'slope']
    recs = []
    step = 4000
    for k in range(0, len(pts), step):
        recs += N11.sample(ee, N11.s2dem_image(), pts[k:k + step], want, 'S2+DEM %d' % k)
    e = pd.DataFrame(recs)
    e.to_csv(out_fp, index=False, encoding='utf-8-sig')
    emit('%d 点 → %s（S2 完整 %d）' % (len(e), out_fp,
                                    int(e[N11.S2B].notna().all(axis=1).sum()) if len(e) else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='both', choices=['gap', 'gsnx', 'both'])
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    if a.set in ('gap', 'both'):
        d = pd.read_csv(os.path.join(M3, 'gap_points.csv'), encoding='utf-8-sig')
        fetch(d, os.path.join(M3, 'gap_s2dem.csv'), a.acct)
    if a.set in ('gsnx', 'both'):
        d = pd.read_csv(os.path.join(M3, 'gsnx_ref_judged.csv'), encoding='utf-8-sig')
        d = d[d.Q1 != '无法判读'].copy()
        fetch(d, os.path.join(M3, 'gsnx_s2dem.csv'), a.acct)


if __name__ == '__main__':
    main()
