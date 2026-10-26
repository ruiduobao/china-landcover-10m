# -*- coding: utf-8 -*-
"""r3_s2_metrics.py — 盲判点 S2 物候旁证：2023 全年月合成 + 2019/2023 冬季

* 输入：blind/points_blind.csv（中性 point_id/lon/lat；无先验）
* 规则源：skill references/影像源与瓦片手册.md §4（生长季合成、SCL 云掩膜、指标口径）；
        主 agent 2026-10-15 指令：全年月合成 + winter2019/2023 多季节；**不得用 NDVI 阈值硬标针阔**。
* 门槛：分块 ≤--chunk 点（默认 200）；单请求硬超时 ≤--timeout（默认 90 s）；失败重试 ≤--retries；
        SCL 剔云/云影/雪(3/8/9/10/11) + CLOUDY_PIXEL_PERCENTAGE<60；scale=10（NDVI/NDWI 用 B8/B4/B3）。
* 输出：cache/s2_parts/s2_c####.parquet（分块落盘，可续跑）、blind/s2_metrics.csv（逐点指标）、
        qc/s2_fetch_summary.json（块数/耗时/失败）
* 用法：python r3_s2_metrics.py --acct p2vlkaia [--chunk 200] [--timeout 90] [--dry]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

sys.path.insert(0, os.path.join(WORK, 'v31_exp', 'code'))
PARTD = os.path.join(CACHE, 's2_parts')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build_image(ee, geom):
    """28 波段：2023 逐月 NDVI×1000(12) + 逐月 NDWI×1000(12) + 冬 2019/2023 NDVI/NDWI(4)。"""
    col = ee.ImageCollection(S2_COLLECTION).filterBounds(geom)

    def msk(im):
        scl = im.select('SCL')
        m = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return im.updateMask(m)

    def ndvi(d0, d1, tag):
        c = col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk)
        return c.median().normalizedDifference(['B8', 'B4']).rename(tag).multiply(1000).toFloat()

    def ndwi(d0, d1, tag):
        c = col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk)
        return c.median().normalizedDifference(['B3', 'B8']).rename(tag).multiply(1000).toFloat()

    bands = []
    for m in S2_MONTHS:
        d0 = '%d-%02d-01' % (S2_YEAR, m)
        d1 = '%d-%02d-01' % (S2_YEAR + (m == 12), (m % 12) + 1)
        bands.append(ndvi(d0, d1, 'ndvi_%02d' % m))
        bands.append(ndwi(d0, d1, 'ndwi_%02d' % m))
    for y in S2_WINTER_YEARS:
        bands.append(ndvi('%d-12-01' % (y - 1), '%d-03-01' % y, 'ndvi_djf%d' % y))
        bands.append(ndwi('%d-12-01' % (y - 1), '%d-03-01' % y, 'ndwi_djf%d' % y))
    return ee.Image.cat(bands).toFloat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='p2vlkaia')
    ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--timeout', type=float, default=S2_REQUEST_TIMEOUT_S)
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--dry', action='store_true', help='只打印计划，不发请求')
    a = ap.parse_args()
    ensure_dirs()
    os.makedirs(PARTD, exist_ok=True)
    pts = pd.read_csv(os.path.join(BLIND, 'points_blind.csv'), encoding='utf-8-sig')
    pts['point_id'] = pts['point_id'].astype(str)
    emit('盲判点 %d；账号 %s；chunk=%d；timeout=%.0fs' % (len(pts), a.acct, a.chunk, a.timeout))
    if a.dry:
        emit('--dry：不请求')
        return

    import ee
    import v31_common as VC
    VC.ensure_ctx(a.acct)                      # 串行：进程级 HOME/USERPROFILE（勿并发切号）
    try:
        ee.data.setDeadline(int(a.timeout * 1000))
    except Exception as e:
        emit('提示：setDeadline 不可用（%s），改用分块+重试控制' % type(e).__name__)

    summary = dict(account=a.acct, chunk=a.chunk, timeout_s=a.timeout, started=time.strftime('%Y-%m-%d %H:%M:%S'),
                   parts=[], failures=[])
    t0 = time.time()
    for k in range(0, len(pts), a.chunk):
        sub = pts.iloc[k:k + a.chunk]
        fp = os.path.join(PARTD, 's2_c%04d.parquet' % k)
        if os.path.exists(fp):
            summary['parts'].append(dict(part=k // a.chunk, file=fp, skip=True))
            emit('  块 %d 已存在，跳过' % (k // a.chunk))
            continue
        lo0, lo1 = float(sub['lon'].min()), float(sub['lon'].max())
        la0, la1 = float(sub['lat'].min()), float(sub['lat'].max())
        geom = ee.Geometry.BBox(lo0, la0, lo1, la1)     # 显式平面 BBox（doc71 几何纪律）
        img = build_image(ee, geom)
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
                                   for pid, x, y in zip(sub['point_id'], sub['lon'], sub['lat'])])
        t1 = time.time()
        rows, err = [], ''
        for att in range(a.retries):
            try:
                # 注意：Image.sampleRegions 的 Python API 无 dropNulls，默认丢弃含任一 null 的点
                # （实测 200 点只回 27）→ 改用 reduceRegions + first()，返回全部点、缺测为 null。
                r = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                      scale=10, tileScale=4).getInfo()['features']
                rows = [dict(f['properties']) for f in r]
                break
            except Exception as e:
                err = str(e)[:120]
                emit('  块 %d 重试 %d/%d：%s' % (k // a.chunk, att + 1, a.retries, err))
                time.sleep(min(30, 6 * (att + 1)))
        dt = time.time() - t1
        if rows:
            pd.DataFrame(rows).to_parquet(fp, index=False)
            summary['parts'].append(dict(part=k // a.chunk, file=fp, n=len(rows), seconds=round(dt, 1)))
            emit('  块 %d：%d 点 / %.1fs' % (k // a.chunk, len(rows), dt))
        else:
            summary['failures'].append(dict(part=k // a.chunk, error=err, seconds=round(dt, 1)))
            emit('  块 %d 失败：%s' % (k // a.chunk, err))

    # ---- 合并 + 指标 ----
    fs = sorted([os.path.join(PARTD, f) for f in os.listdir(PARTD) if f.endswith('.parquet')])
    if not fs:
        raise SystemExit('无分块产物')
    raw = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    raw['point_id'] = raw['point_id'].astype(str)
    ncol = [c for c in raw.columns if c.startswith('ndvi_') and c[5:7].isdigit()]
    wcol = [c for c in raw.columns if c.startswith('ndwi_') and c[5:7].isdigit()]
    nd = raw[ncol].astype(float)
    nw = raw[wcol].astype(float)
    out = pd.DataFrame(dict(point_id=raw['point_id']))
    out['n_months_valid'] = nd.notna().sum(axis=1)
    out['ndvi_mean'] = (nd.mean(axis=1) / 1000).round(4)
    out['ndvi_amp'] = ((nd.max(axis=1) - nd.min(axis=1)) / 1000).round(4)
    out['ndvi_min'] = (nd.min(axis=1) / 1000).round(4)
    out['ndvi_max'] = (nd.max(axis=1) / 1000).round(4)
    out['ndvi_peak_month'] = nd.idxmax(axis=1).str[5:7].astype('Int64')
    jja = ['ndvi_%02d' % m for m in (6, 7, 8)]
    djf23 = [c for c in raw.columns if c == 'ndvi_djf2023']
    out['ndvi_summer'] = (nd[[c for c in jja if c in nd.columns]].mean(axis=1) / 1000).round(4)
    out['ndvi_winter2023'] = (raw['ndvi_djf2023'].astype(float) / 1000).round(4)
    out['ndvi_winter2019'] = (raw['ndvi_djf2019'].astype(float) / 1000).round(4)
    out['winter_green_delta'] = (out['ndvi_winter2023'] - out['ndvi_winter2019']).round(4)
    out['ndwi_min'] = (nw.min(axis=1) / 1000).round(4)
    out['ndwi_max'] = (nw.max(axis=1) / 1000).round(4)
    fl = np.where(out['ndwi_min'] > 0.05, 'yes', np.where(out['ndwi_min'] > -0.05, 'weak', 'no'))
    out['flood_evidence'] = fl
    out['ndvi_peak_date'] = [('%d-%s-15' % (S2_YEAR, int(m)) if m == m else '') for m in out['ndvi_peak_month']]
    out['date_status'] = 'dated_s2_2023'
    out['s2_source'] = 'COPERNICUS/S2_SR_HARMONIZED (SCL 云掩膜, 月中位合成, scale 10 m)'
    out['note'] = '仅作旁证：不得用阈值直接判定林型（见协议 72 判读纪律）'
    # 原始逐月序列一并保留（可复核，无先验）
    for c in ncol + wcol + [c for c in raw.columns if c.startswith(('ndvi_djf', 'ndwi_djf'))]:
        if c not in out.columns:
            out[c] = raw[c].values
    out.to_csv(os.path.join(BLIND, 's2_metrics.csv'), index=False, encoding='utf-8-sig')
    summary.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1),
                   n_points=int(len(out)), n_valid_months_median=float(out['n_months_valid'].median()),
                   n_points_ge6_months=int((out['n_months_valid'] >= 6).sum()),
                   output=os.path.join(BLIND, 's2_metrics.csv'))
    jdump(os.path.join(QC, 's2_fetch_summary.json'), summary)
    emit('完成：%d 点，月有效中位 %.0f，≥6 月 %d 点，%.0fs → blind/s2_metrics.csv' % (
        len(out), out['n_months_valid'].median(), (out['n_months_valid'] >= 6).sum(), time.time() - t0))


if __name__ == '__main__':
    main()
