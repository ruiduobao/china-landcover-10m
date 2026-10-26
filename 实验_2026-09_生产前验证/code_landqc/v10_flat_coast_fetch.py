# -*- coding: utf-8 -*-
"""v10_flat_coast_fetch.py — 湖河滩地(182)海岸甄别证据抓取（Murray 潮间带 + 海陆距离）

* 输入：lists/flat_delivered.parquet（来自交付 SHP 的 26,507 点；point_id/lon/lat）
* 证据：
    it_n      = Murray v1_1 全球潮间带 11 期（1984–2016）中判为潮滩的期数（0–11；unmask 0）
    it_ever   = it_n > 0（该像元在任一历史期被制图为潮滩）
    d_it_m    = 到最近潮滩像元的距离（m）＝ it_ever.unmask(0).fastDistanceTransform(300).sqrt()×30
                （语义已实测：输出=到最近"非零（潮滩）"像元的距离；上限 300px≈9 km；30 m 原生网格）
    d_sea_m   = 到最近海域像元的距离（m）＝ ETOPO1(bedrock<0).unmask(0).fastDistanceTransform(60).sqrt()×1855.32
                ⚠️ ETOPO1 为 1′（≈1.855 km）网格 → 该距离 ±2–4 km 粗粒度，仅作辅助筛查
    etopo_m   = ETOPO1 bedrock 高程（m）
* 分片：--part/--nparts；块 2000（子请求 1000）；已存在块跳过（幂等）
* 用法：python v10_flat_coast_fetch.py --part 0 --nparts 3 --acct vidalmvtpronet18 --outdir ../data/flat_fix/feats
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
BASE = os.path.join(WORK, 'data', 'flat_fix')
COLS = ['it_n', 'it_ever', 'd_it_m', 'd_sea_m', 'etopo_m']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), str(m)), flush=True)


def build(ee):
    col = ee.ImageCollection('UQ/murray/Intertidal/v1_1/global_intertidal')

    def m1(im):
        return im.select('classification').eq(1).rename('it')

    n_img = col.map(m1).sum().unmask(0).rename('it_n')
    ever = n_img.gt(0).rename('it_ever')
    d_it = ever.unmask(0).fastDistanceTransform(300).sqrt().multiply(30).rename('d_it_m')
    et = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock')
    d_sea = et.lt(0).unmask(0).fastDistanceTransform(60).sqrt().multiply(1855.3248).rename('d_sea_m')
    img = ee.Image.cat([n_img, ever, d_it, d_sea, et.rename('etopo_m')]).toFloat()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=3)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--step', type=int, default=2000)
    ap.add_argument('--sub', type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(WORK, 'code'))
    import ee
    import v31_common as VC
    VC.ensure_ctx(a.acct)
    d = pd.read_parquet(os.path.join(BASE, 'lists', 'flat_delivered.parquet'), columns=['point_id', 'lon', 'lat'])
    d['point_id'] = d.point_id.astype(str)
    mine = d.iloc[a.part::a.nparts].reset_index(drop=True)
    os.makedirs(a.outdir, exist_ok=True)
    img = build(ee)
    emit('本片 %d 点（%d 波段）' % (len(mine), len(COLS)))
    nch = (len(mine) + a.step - 1) // a.step
    done = 0
    for k in range(0, len(mine), a.step):
        fp = os.path.join(a.outdir, 'flat_p%02d_c%04d.parquet' % (a.part, k // a.step))
        if os.path.exists(fp):
            done += 1; continue
        blk = mine.iloc[k:k + a.step]
        rows, failed = [], False
        for s0 in range(0, len(blk), a.sub):
            sub = blk.iloc[s0:s0 + a.sub]
            pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': p})
                   for p, x, y in zip(sub.point_id, sub.lon.astype(float), sub.lat.astype(float))]
            got, err = None, ''
            for att in range(6):
                try:
                    r = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=30,
                                          geometries=False, tileScale=4).getInfo()['features']
                    got = {f['properties']['point_id']: f['properties'] for f in r}; break
                except Exception as e:
                    err = str(e)[:110]
                    emit('  块%d 子%d 重试%d %s' % (k // a.step, s0 // a.sub, att + 1, err))
                    time.sleep(8 * (att + 1))
            if got is None:
                with open(os.path.join(a.outdir, '_fail_p%02d.json' % a.part), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(part=a.part, chunk=k // a.step, sub=s0 // a.sub, err=err),
                                       ensure_ascii=False) + '\n')
                failed = True; break
            rows += [dict(point_id=p, **{c: got.get(p, {}).get(c) for c in COLS}) for p in sub.point_id]
        if failed:
            continue
        o = pd.DataFrame(rows)
        for c in COLS:
            o[c] = pd.to_numeric(o[c], errors='coerce')
        o.to_parquet(fp, index=False); done += 1
        emit('  %d/%d 块' % (done, nch))
    emit('完成 p%02d：%d/%d 块 / %d 点' % (a.part, done, nch, len(mine)))


if __name__ == '__main__':
    main()
