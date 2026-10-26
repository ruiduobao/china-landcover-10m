# -*- coding: utf-8 -*-
"""h11_yearly_evidence.py — 为补样点抓逐年（2017–2024）动态证据（S2 + Dynamic World）

* 输入：--list data/final/final_2023_v2.parquet（7,098 点，pid/lon/lat/fam/class）
* 每年抓：DW 年概率 9 波段 mean；S2 夏 NDVI/NDWI（6–9 月）、冬 NDVI（上年 12–2 月）、
        宽窗 NDVI（4–11 月，2017/2018 南方夏季档案真空时兜底）
* 静态波段（WC/GSW/SRTM）不再重复抓，合并时从 2023 证据按 pid 取
* 输出：--outdir/ev_<year>_p<part>_b<block>.parquet
* 用法：python h11_yearly_evidence.py --year 2019 --acct zzuoavsm --part 0 --nparts 2
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
COLS = ['pid'] + DW8 + ['s2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong', 's2_ndvi_wide']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build(ee, year):
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
          .select(DW_RAW).mean().unmask(-1).rename(DW8))
    s2c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))

    def nd(d0, d1, bands, names):
        try:
            c = (s2c.filterDate(d0, d1)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk).median())
            out = [c.normalizedDifference([b1, b2]).multiply(1000).rename(nm).toFloat().unmask(SENT)
                   for (b1, b2), nm in zip(bands, names)]
            return ee.Image.cat(out)
        except Exception:
            return ee.Image.constant([SENT] * len(names)).rename(names).toFloat()

    xia = nd('%d-06-01' % year, '%d-09-01' % year, [('B8', 'B4'), ('B3', 'B8')],
             ['s2_ndvi_xia', 's2_ndwi_xia'])
    wide = nd('%d-04-01' % year, '%d-11-01' % year, [('B8', 'B4')], ['s2_ndvi_wide'])
    dong = nd('%d-12-01' % (year - 1), '%d-03-01' % year, [('B8', 'B4')], ['s2_ndvi_dong'])
    return ee.Image.cat([dw, xia, dong, wide]).toFloat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--list', default=r'F:/lc_work/bare_sparse_v3/data/final/final_2023_v2.parquet')
    ap.add_argument('--acct', required=True)
    ap.add_argument('--pid', default=None, help='项目ID（未登记账号用；配合 --acct）')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--chunk', type=int, default=1500)
    a = ap.parse_args()
    import ee
    outdir = a.outdir or os.path.join(r'F:/lc_work/bare_sparse_v3/data/evidence_yearly', str(a.year))
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_parquet(a.list)
    d = d[d.index % a.nparts == a.part].reset_index(drop=True)
    emit('year=%d part %d/%d：%d 点（acct=%s）' % (a.year, a.part, a.nparts, len(d), a.acct))
    blocks = [d.iloc[i:i + a.chunk] for i in range(0, len(d), a.chunk)]
    ee_mod = None
    t0 = time.time()
    for bi, b in enumerate(blocks):
        fp = os.path.join(outdir, 'ev_%d_p%02d_b%04d.parquet' % (a.year, a.part, bi))
        if os.path.isfile(fp):
            continue
        for attempt in range(4):
            try:
                if ee_mod is None:
                    if a.pid:
                        from prod_conf import apply_account_env
                        apply_account_env(a.acct)
                        ee_mod = ee
                        ee.Initialize(project=a.pid)
                    else:
                        ee_mod, _ = VC.ctx(a.acct)
                    img = build(ee_mod, a.year)
                feats = [ee_mod.Feature(ee_mod.Geometry.Point([float(r.lon), float(r.lat)]), {'pid': r.pid})
                         for r in b[['pid', 'lon', 'lat']].itertuples(index=False)]
                res = img.sampleRegions(collection=ee_mod.FeatureCollection(feats), scale=30,
                                        tileScale=4, geometries=False).getInfo()
                rows = [f['properties'] for f in res['features']]
                out = pd.DataFrame(rows)
                for c in COLS:
                    if c not in out.columns:
                        out[c] = np.nan
                out[COLS].to_parquet(fp, index=False)
                if bi % 2 == 0:
                    emit('  块 %d/%d 完成（%.0fs）' % (bi + 1, len(blocks), time.time() - t0))
                break
            except Exception as e:
                emit('  块 %d 尝试 %d 失败：%s' % (bi + 1, attempt + 1, str(e)[:140]))
                time.sleep(10 + 10 * attempt)
                ee_mod = None
        else:
            emit('  !! 块 %d 连续失败，退出（可续跑）' % (bi + 1))
            sys.exit(2)
    emit('year=%d part=%d 完成（%.0fs）' % (a.year, a.part, time.time() - t0))


if __name__ == '__main__':
    main()
