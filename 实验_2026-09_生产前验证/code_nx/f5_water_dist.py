# -*- coding: utf-8 -*-
"""f5_water_dist.py — 独立水体数据（JRC GSW）+ 距离场，用于"水体邻域"生态规则

* 输入：GEE `JRC/GSW1_4/GlobalSurfaceWater`（band `occurrence`，0–100，1984–2021 年水出现频率）
* 规则源（预注册，用于验证用户 1/2 问）：
        · 水判据两档：occurrence≥50 = 常年水；occurrence≥1 = 曾出现水
        · 距离场在 **1 km 网格**上算（欧氏距离，scipy.ndimage.distance_transform_edt），
          换算成 km（按纬度修正经向格距）；用于池点的"邻水距离"
* 门槛：下载分纬带（每带 ≤48 MB / ≤5000 要素限制不适用，栅格下载）→ 本地拼接 → 距离变换
* 输出：data/eco_gate/_water/jrc_occ_chn_500m.tif
        data/eco_gate/_water/dist_water_km.csv 由 f6 生成（本脚本只落栅格）
* 用法：python f5_water_dist.py [-​-acct bx15mw]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import rasterio
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'eco_gate', '_water')
BOX = [73.0, 18.0, 135.0, 54.0]
BANDS = [(y, y + 2) for y in range(18, 54, 2)]   # 2° 纬带（4° 带 50.4MB 仍略超 48MB 上限）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    fp = os.path.join(OUTD, 'jrc_occ_chn_500m.tif')
    if os.path.exists(fp):
        emit('已存在 %s，跳过' % fp)
        return
    import ee
    VC.ensure_ctx(a.acct)
    s, _ = VC.sess(a.acct)
    # 500 m 最大值池化：1 km 会把窄河/湿地平均掉（实测真湿地 occ=0），max 保留水体信号
    # 顺序关键：先在 30 m 上降采样（max 池化保留窄河），**再** unmask（否则 unmask 在原生分辨率
    # 上会超 2^31 像元上限——2026-10-10 实测报错 INVALID_ARGUMENT）
    occ = (ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
           .reduceResolution(reducer=ee.Reducer.max(), maxPixels=4096)
           .reproject(crs='EPSG:4326', scale=500)
           .unmask(0))
    parts = []
    for i, (y0, y1) in enumerate(BANDS):
        rect = ee.Geometry.Rectangle([BOX[0], y0, BOX[2], y1])
        p = os.path.join(OUTD, 'jrc_b%d.tif' % i)
        ok = False
        for attempt in range(4):
            try:
                url = occ.getDownloadURL({'scale': 500, 'crs': 'EPSG:4326', 'region': rect,
                                          'format': 'GEO_TIFF'})
                b = s.get(url, timeout=900).content
                if b[:2] in (b'II', b'MM') and len(b) > 1e5:      # TIFF magic + 非空
                    open(p, 'wb').write(b)
                    emit('  纬带 %d-%d°N → %.2f MB' % (y0, y1, len(b) / 2 ** 20))
                    ok = True
                    break
                emit('  纬带 %d-%d°N 第%d次异常（%.60s）' % (y0, y1, attempt + 1, b[:60]))
            except Exception as e:
                emit('  纬带 %d-%d°N 第%d次失败：%s' % (y0, y1, attempt + 1, str(e)[:90]))
            time.sleep(8 + 6 * attempt)
        if ok:
            parts.append(p)
        else:
            emit('  ⚠ 纬带 %d-%d°N 放弃' % (y0, y1))
    arrs, prof = [], None
    for p in parts:
        with rasterio.open(p) as ds:
            arrs.append(ds.read())
            prof = ds.profile.copy()
    emit('  实际拼接 %d / %d 带' % (len(parts), len(BANDS)))
    arr = np.concatenate(arrs, axis=1)
    prof.update(height=arr.shape[1], width=arr.shape[2],
                transform=rasterio.transform.from_origin(BOX[0], BOX[3], 0.005, 0.005),
                compress='deflate', tiled=True)
    with rasterio.open(fp, 'w', **prof) as dst:
        dst.write(arr)
    for p in parts:
        os.remove(p)
    emit('JRC 水体拼接完成 %s shape=%s 值域[%d,%d]' % (fp, arr.shape, arr.min(), arr.max()))
    # 统计水体占比
    v = arr[0]
    emit('  occurrence>=50 占比 %.3f%%；>=1 占比 %.3f%%' % (
        100 * float((v >= 50).mean()), 100 * float((v >= 1).mean())))


if __name__ == '__main__':
    main()
