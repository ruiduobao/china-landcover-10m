# -*- coding: utf-8 -*-
"""h2_evidence.py — 为 稀疏植被/裸地 候选点抓 2023 独立证据（GEE；多账号分片；块级幂等）

* 输入：--list 逗号分隔的候选 parquet（列：lon/lat/fam/class/share9）
* 证据（一次 sampleRegions，scale=30，逐源 unmask 后 cat 防整点丢特征）：
    GSW1_4 点级 occurrence/max_extent/seasonality/recurrence（-1=无数据）
    DW 2023 概率 9 波段 mean（water/trees/grass/crops/built/bare/snow_and_ice/flooded_vegetation/shrub；-1）
    ESA WorldCover v200 Map（-1）
    SRTM dem/slope（-9999）
    S2 2023 夏（6–9 月）NDVI/NDWI、冬（12–2 月）NDVI（×1000；SCL 剔 3/8/9，云量<60；-9999）
* 分片：--part i --nparts N（按行号取模）；块 1500 点；已存在块跳过（幂等）
* 输出：--outdir/ev_p<part>_b<block>.parquet
* 用法：
    python h2_evidence.py --list ../data/candidates/cand_north.parquet --acct <acct> --part 0 --nparts 6
* 纪律：代理 socks5h（v31_common 自探 7890/7897/7891）；切账号同时设 HOME+USERPROFILE（VC.ctx）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC  # noqa

SENT = -9999.0
DW8 = ['water', 'trees', 'grass', 'crops', 'built', 'bare',
       'snow_and_ice', 'flooded_vegetation', 'shrub']
DW_RAW = ['water', 'trees', 'grass', 'crops', 'built', 'bare',
          'snow_and_ice', 'flooded_vegetation', 'shrub_and_scrub']
OUT_COLS = (['pid', 'lon', 'lat', 'fam', 'class']
            + ['occurrence', 'max_extent', 'seasonality', 'recurrence']
            + DW8 + ['wc', 'dem', 'slope',
                     's2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong'])


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build(ee):
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    g = gsw.select(['occurrence', 'max_extent', 'seasonality', 'recurrence']).unmask(-1)
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01', '2024-01-01').select(DW_RAW)
          .mean().unmask(-1).rename(DW8))
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('wc').unmask(-1)
    dem = ee.Image('USGS/SRTMGL1_003')
    topo = ee.Image.cat([dem.rename('dem'),
                         ee.Terrain.slope(dem).rename('slope')]).unmask(SENT)

    s2c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))

    def s2(d0, d1, bands, names):
        c = (s2c.filterDate(d0, d1)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk).median())
        out = []
        for (b1, b2), nm in zip(bands, names):
            out.append(c.normalizedDifference([b1, b2]).multiply(1000)
                       .rename(nm).toFloat().unmask(SENT))
        return ee.Image.cat(out)

    s2x = s2('2023-06-01', '2023-09-01', [('B8', 'B4'), ('B3', 'B8')],
             ['s2_ndvi_xia', 's2_ndwi_xia'])
    s2d = s2('2022-12-01', '2023-02-28', [('B8', 'B4')], ['s2_ndvi_dong'])
    return ee.Image.cat([g, dw, wc, topo, s2x, s2d]).toFloat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'evidence'))
    ap.add_argument('--chunk', type=int, default=1500)
    a = ap.parse_args()

    import ee
    os.makedirs(a.outdir, exist_ok=True)
    d = pd.concat([pd.read_parquet(p) for p in a.list.split(',')], ignore_index=True)
    d = d[d.index % a.nparts == a.part].reset_index(drop=True)
    d['pid'] = [f'{r.fam}_{r.lon:.6f}_{r.lat:.6f}' for r in d.itertuples()]
    emit('part %d/%d：%d 点（acct=%s）' % (a.part, a.nparts, len(d), a.acct))
    blocks = [d.iloc[i:i + a.chunk] for i in range(0, len(d), a.chunk)]
    emit('块数 %d' % len(blocks))

    ee_mod = None
    t0 = time.time()
    done = 0
    for bi, b in enumerate(blocks):
        fp = os.path.join(a.outdir, 'ev_p%02d_b%04d.parquet' % (a.part, bi))
        if os.path.isfile(fp):
            done += 1
            continue
        for attempt in range(4):
            try:
                if ee_mod is None:
                    ee_mod, pid = VC.ctx(a.acct)
                    img = build(ee_mod)
                bb = b.rename(columns={'class': 'cls'})[['pid', 'lon', 'lat', 'fam', 'cls']]
                feats = [ee_mod.Feature(ee_mod.Geometry.Point([float(r.lon), float(r.lat)]),
                                        {'pid': r.pid, 'lon': float(r.lon), 'lat': float(r.lat),
                                         'fam': r.fam, 'class': int(r.cls)})
                         for r in bb.itertuples(index=False)]
                fc = ee_mod.FeatureCollection(feats)
                res = img.sampleRegions(collection=fc, scale=30, tileScale=4, geometries=False)
                n = int(res.size().getInfo())
                if n == 0:
                    raise RuntimeError('sampleRegions 返回 0')
                info = res.getInfo()
                rows = [f['properties'] for f in info['features']]
                out = pd.DataFrame(rows)
                if 'cls' in out.columns and 'class' not in out.columns:
                    out = out.rename(columns={'cls': 'class'})
                for c in OUT_COLS:
                    if c not in out.columns:
                        out[c] = np.nan
                out = out[OUT_COLS]
                out.to_parquet(fp, index=False)
                done += 1
                if done % 2 == 0:
                    emit('  块 %d/%d 完成（累计 %d 点，%.0fs）' % (
                        bi + 1, len(blocks), len(out), time.time() - t0))
                break
            except Exception as e:
                emit('  块 %d 尝试 %d 失败：%s' % (bi + 1, attempt + 1, str(e)[:160]))
                time.sleep(10 + 10 * attempt)
                ee_mod = None
        else:
            emit('  !! 块 %d 连续失败，退出（可重跑本分片续跑）' % (bi + 1))
            sys.exit(2)
    emit('part %d 完成：%d 块（%.0fs）' % (a.part, done, time.time() - t0))


if __name__ == '__main__':
    main()
