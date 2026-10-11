# -*- coding: utf-8 -*-
"""k17_s2_fetch.py — 给放量选中的 11.5 万个点抓 S2 物候（分片 + 逐块增量落盘）

* 输入：data/shrub_nat/nat_points.csv（k15 产出：θ=0.7 高精档 + θ=0.6 走量档）
* 规则源：S2 = COPERNICUS/S2_SR_HARMONIZED，djf(2022-12~2023-03)/jja(2023-06~09) 中位 NDVI×1000；
        SCL 剔云/云影/雪，CLOUDY_PIXEL_PERCENTAGE<60；scale=20
* 为什么单独抓：判型（常绿/落叶）需要物候振幅；MODIS 替代已验证失败（一致率 0.667，见 k14）
* 工程：--part/--nparts 分片；每 4,000 点一块，**块级落盘**（r2 轮因只在进程末落盘被清理打断）
* 输出：data/shrub_nat/s2_parts/s2_p{pp}_c{kk}.parquet
* 用法：python k17_s2_fetch.py --part 0 --nparts 12 [--acct bx15mw]
"""
import argparse, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PARTD = os.path.join(OUTD, 's2_parts')
STEP = 4000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), m), flush=True)


def main():
    global ee
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, required=True)
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    os.makedirs(PARTD, exist_ok=True)
    import ee as _ee
    ee = _ee
    import v31_common as VC
    VC.ensure_ctx(a.acct)

    def s2_band(col, d0, d1, tag):
        def msk(im):
            scl = im.select('SCL')
            return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))
        c = (col.filterDate(d0, d1).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(msk))
        return c.median().normalizedDifference(['B8', 'B4']).rename(tag).multiply(1000).toFloat()

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    img = ee.Image.cat([s2_band(col, '2022-12-01', '2023-03-01', 'ndvi_djf'),
                        s2_band(col, '2023-06-01', '2023-09-01', 'ndvi_jja')]).toFloat()
    df = pd.read_csv(os.path.join(OUTD, 'nat_points.csv'), encoding='utf-8-sig')
    df['point_id'] = df['point_id'].astype(str)
    mine = df.iloc[a.part::a.nparts].reset_index(drop=True)
    emit('分片 %d/%d：%d 点' % (a.part, a.nparts, len(mine)))
    t0 = time.time()
    done_chunks = 0
    for k in range(0, len(mine), STEP):
        fp = os.path.join(PARTD, 's2_p%02d_c%04d.parquet' % (a.part, k // STEP))
        if os.path.exists(fp):
            done_chunks += 1
            continue
        sub = mine.iloc[k:k + STEP]
        pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
               for pid, x, y in zip(sub['point_id'], sub['lon'].to_numpy(float), sub['lat'].to_numpy(float))]
        rows = []
        for att in range(6):
            try:
                r = img.sampleRegions(collection=ee.FeatureCollection(pts), scale=20,
                                      geometries=False, tileScale=4).getInfo()['features']
                rows = [dict(point_id=f['properties']['point_id'],
                             ndvi_djf=f['properties'].get('ndvi_djf'),
                             ndvi_jja=f['properties'].get('ndvi_jja')) for f in r]
                break
            except Exception as e:
                emit('  块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                time.sleep(8 * (att + 1))
        if rows:
            pd.DataFrame(rows).to_parquet(fp, index=False)
        done_chunks += 1
        if done_chunks % 2 == 0:
            emit('  %d 块（%.0f 秒）' % (done_chunks, time.time() - t0))
    emit('完成 %d 块' % done_chunks)


if __name__ == '__main__':
    main()
