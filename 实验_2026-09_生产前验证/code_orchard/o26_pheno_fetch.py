# -*- coding: utf-8 -*-
"""o26_pheno_fetch.py — 物候指纹抓取：S2 月度 NDVI 6 期 + S1 冬夏 VV/VH（橡胶候选与参照点）

* 输入：--src rubber → data/orchard/rubber_sea_cn_points.parquet（30,000）
        --src gemcn  → data/orchard/gem/training_china.csv 提取的 lon/lat（3,265，带 GEM 标签）
* 规则源（与既往一致）：
  S2 = COPERNICUS/S2_SR_HARMONIZED，SCL∉{3,8,9} 掩膜 + CLOUDY<60，2020 年 1/3/5/7/9/11 月各取中位 NDVI（6 波段）；
  S1 = COPERNICUS/S1_GRD IW VV+VH，冬(DJF: 2020-01-01~03-01)/夏(JJA: 06-01~09-01) 中位（4 波段）
* 门槛：每块 5000 点，块落盘可续跑
* 输出：data/orchard/pheno/{rubber|gemcn}_p{ci}.parquet
* 用法：python o26_pheno_fetch.py --src rubber [--part 0 --nparts 3 --acct x]
"""
import argparse
import glob
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
OUTD = os.path.join(ORD, 'pheno')
MONTHS = [('m01', '2020-01-01', '2020-02-01'), ('m03', '2020-03-01', '2020-04-01'),
          ('m05', '2020-05-01', '2020-06-01'), ('m07', '2020-07-01', '2020-08-01'),
          ('m09', '2020-09-01', '2020-10-01'), ('m11', '2020-11-01', '2020-12-01')]
BANDS = ['nd_%s' % m for m, _, _ in MONTHS] + ['s1_vv_w', 's1_vh_w', 's1_vv_s', 's1_vh_s']
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def build_img(ee):
    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate('2020-01-01', '2021-01-01')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
    bands = []
    for name, a, b in MONTHS:
        c = col.filterDate(a, b).median()
        bands.append(c.normalizedDifference(['B8', 'B4']).rename('nd_' + name))
    s1c = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filterDate('2020-01-01', '2021-01-01').select(['VV', 'VH']))
    s1 = ee.Image.cat([
        s1c.filterDate('2020-01-01', '2020-03-01').median().rename(['s1_vv_w', 's1_vh_w']),
        s1c.filterDate('2020-06-01', '2020-09-01').median().rename(['s1_vv_s', 's1_vh_s'])])
    return ee.Image.cat(bands + [s1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, choices=['rubber', 'gemcn'])
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    if a.src == 'rubber':
        d = pd.read_parquet(os.path.join(ORD, 'rubber_sea_cn_points.parquet')).reset_index(drop=True)
        d['uid'] = np.arange(len(d))
    else:
        d = pd.read_csv(r'F:/lc_work/外部样本/GEM_Forest/training_china.csv',
                        encoding='utf-8-sig')[['lon', 'lat', 'forest']].reset_index(drop=True)
        d['uid'] = np.arange(len(d))
    chunks = [d.iloc[i:i + CHUNK] for i in range(0, len(d), CHUNK)]
    mine = [(i, c) for i, c in enumerate(chunks) if i % a.nparts == a.part]
    emit('%s：%d 点 / %d 块；本进程 %d 块' % (a.src, len(d), len(chunks), len(mine)))
    if not mine:
        return
    import ee
    VC.ensure_ctx(a.acct)
    img = build_img(ee)
    for ci, c in mine:
        fp = os.path.join(OUTD, '%s_p%02d.parquet' % (a.src, ci))
        if os.path.exists(fp):
            emit('  块 %02d 已存在' % ci)
            continue
        for attempt in range(4):
            try:
                feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'uid': int(r.uid)})
                         for r in c.itertuples()]
                got = img.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                                        tileScale=4).getInfo()['features']
                df = pd.DataFrame([{'uid': g['properties']['uid'],
                                    **{k: g['properties'].get(k) for k in BANDS},
                                    'forest': g['properties'].get('forest')} for g in got])
                m = c.merge(df, on='uid', how='left')
                m['pheno_ok'] = m['nd_m07'].notna()
                m.to_parquet(fp, index=False)
                emit('  块 %02d: %d/%d（nd_m07 命中）' % (ci, int(m.pheno_ok.sum()), len(c)))
                break
            except Exception as e:
                emit('  块 %02d 重试%d: %s' % (ci, attempt, str(e)[:140]))
                time.sleep(10)


if __name__ == '__main__':
    main()
