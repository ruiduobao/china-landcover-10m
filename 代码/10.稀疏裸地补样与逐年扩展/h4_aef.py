# -*- coding: utf-8 -*-
"""h4_aef.py — 为最终补样点抓 AEF64（2023 年度嵌入；多账号分片；块级幂等）

* 输入：--list 最终点 parquet（列：pid/lon/lat/fam/class/...）
* 图像：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023 → mosaic → select(A00..A63)
* 采样：sampleRegions(scale=10, tileScale=4, geometries=False)；块 1000
* 输出：--outdir/aef_p<part>_b<block>.parquet（pid + A00..A63）
* 用法：python h4_aef.py --list ../data/final/final_2023.parquet --acct <acct> --part 0 --nparts 2
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC  # noqa

FEATS = ['A%02d' % i for i in range(64)]
YEAR = 2023


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--pid', default=None, help='项目ID（未登记账号用；配合 --acct）')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'aef'))
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--year', type=int, default=YEAR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import ee
    d = pd.read_parquet(a.list)
    d = d[d.index % a.nparts == a.part].reset_index(drop=True)
    if 'pid' not in d.columns:
        d['pid'] = [f'{r.fam}_{r.lon:.6f}_{r.lat:.6f}' for r in d.itertuples()]
    emit('part %d/%d：%d 点（acct=%s, year=%d）' % (a.part, a.nparts, len(d), a.acct, a.year))
    blocks = [d.iloc[i:i + a.chunk] for i in range(0, len(d), a.chunk)]
    ee_mod = None
    t0 = time.time()
    for bi, b in enumerate(blocks):
        fp = os.path.join(a.outdir, 'aef_p%02d_b%04d.parquet' % (a.part, bi))
        if os.path.isfile(fp):
            continue
        for attempt in range(4):
            try:
                if ee_mod is None:
                    if a.pid:
                        from prod_conf import apply_account_env
                        apply_account_env(a.acct)
                        import ee as _ee
                        _ee.Initialize(project=a.pid)
                        ee_mod = _ee
                    else:
                        ee_mod, pid = VC.ctx(a.acct)
                    img = (ee_mod.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                           .filterDate('%d-01-01' % a.year, '%d-01-01' % (a.year + 1))
                           .mosaic().select(FEATS))
                feats = [ee_mod.Feature(ee_mod.Geometry.Point([float(r.lon), float(r.lat)]),
                                        {'pid': r.pid})
                         for r in b[['pid', 'lon', 'lat']].itertuples(index=False)]
                fc = ee_mod.FeatureCollection(feats)
                res = img.sampleRegions(collection=fc, scale=10, tileScale=4, geometries=False)
                info = res.getInfo()
                rows = [f['properties'] for f in info['features']]
                out = pd.DataFrame(rows)
                for c in ['pid'] + FEATS:
                    if c not in out.columns:
                        out[c] = np.nan
                out = out[['pid'] + FEATS]
                miss = int(out[FEATS].isna().all(axis=1).sum())
                if miss:
                    emit('  块 %d：%d/%d 点 AEF 全缺失（保留待核）' % (bi + 1, miss, len(out)))
                out.to_parquet(fp, index=False)
                if bi % 2 == 0:
                    emit('  块 %d/%d 完成（%.0fs）' % (bi + 1, len(blocks), time.time() - t0))
                break
            except Exception as e:
                emit('  块 %d 尝试 %d 失败：%s' % (bi + 1, attempt + 1, str(e)[:150]))
                time.sleep(10 + 10 * attempt)
                ee_mod = None
        else:
            emit('  !! 块 %d 连续失败，退出（可续跑）' % (bi + 1))
            sys.exit(2)
    emit('part %d 完成（%.0fs）' % (a.part, time.time() - t0))


if __name__ == '__main__':
    main()
