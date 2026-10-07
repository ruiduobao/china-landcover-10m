# -*- coding: utf-8 -*-
"""
p7_postprocess.py — 8 期镶嵌裁剪 + 时序一致性（本地）
* 四象限 tif → 拼接 → 北京界内裁剪 → 逐年成品
* 时序一致性: 短期波动抑制（3 年投票）+ 城市化不可逆规则 → 最终版 + QC 层
* 输出: raster/CNLC10_BJ_{year}_final.tif, raster/QC_BJ.tif
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import shape

OUT = '数据/本地处理/北京试点/raster'
YEARS = list(range(2017, 2025))

def mosaic_year(year):
    fps = sorted(glob.glob(os.path.join(OUT, f'CNLC10_BJ_{year}_p*.tif')))
    if not fps:
        return None
    if len(fps) == 1:
        srcs = rasterio.open(fps[0])
    else:
        srcs = merge([rasterio.open(f) for f in fps])
        meta = rasterio.open(fps[0]).meta.copy()
        meta.update(height=srcs[0].shape[1], width=srcs[0].shape[2],
                    transform=srcs[1])
        mosaicked = os.path.join(OUT, f'CNLC10_BJ_{year}_mosaic.tif')
        with rasterio.open(mosaicked, 'w', **meta) as d:
            d.write(srcs[0])
        srcs = rasterio.open(mosaicked)
    # 北京界裁剪
    geom = [shape(PC.beijing_geojson())]
    clipped, tf = mask(srcs, geom, crop=True, nodata=0)
    meta = srcs.meta.copy()
    meta.update(height=clipped.shape[1], width=clipped.shape[2],
                transform=tf, nodata=0, compress='lzw')
    final = os.path.join(OUT, f'CNLC10_BJ_{year}_final.tif')
    with rasterio.open(final, 'w', **meta) as d:
        d.write(clipped)
    srcs.close()
    return final

def consistency():
    """8 期堆栈: 短期波动(1年孤立类)→用前后年众数替代; 城市化不可逆"""
    stacks = {}
    meta = None
    for y in YEARS:
        fp = os.path.join(OUT, f'CNLC10_BJ_{y}_final.tif')
        with rasterio.open(fp) as s:
            stacks[y] = s.read(1)
            meta = meta or s.meta.copy()
    Y = np.stack([stacks[y] for y in YEARS])   # T×H×W
    T, H, W = Y.shape
    qc = np.zeros((H, W), dtype=np.uint8)      # bit0=被平滑 bit1=不可逆修正
    # 1) 短期波动: Y[t] != 前后年 且 前年==后年 → 改为前后年值
    for t in range(1, T-1):
        iso = (Y[t] != Y[t-1]) & (Y[t] != Y[t+1]) & (Y[t-1] == Y[t+1]) & (Y[t] > 0)
        Y[t][iso] = Y[t-1][iso]
        qc[iso] |= 1
    # 2) 城市化不可逆: 一旦 190/200，之后不能回到耕地/植被（保持 190/200）
    urb = np.zeros((H, W), dtype=bool)
    for t in range(T):
        now_urb = np.isin(Y[t], [190, 200])
        new_urb = now_urb & ~urb
        back = urb & ~now_urb & np.isin(Y[t], [10, 12, 61, 71, 81, 91, 120, 121, 130])
        Y[t][back] = 190
        qc[back] |= 2
        urb |= now_urb
    # 输出
    for i, y in enumerate(YEARS):
        meta.update(compress='lzw')
        with rasterio.open(os.path.join(OUT, f'CNLC10_BJ_{y}_final.tif'), 'w', **meta) as d:
            d.write(Y[i][None])
    meta.update(count=1, dtype='uint8')
    with rasterio.open(os.path.join(OUT, 'QC_BJ.tif'), 'w', **meta) as d:
        d.write(qc[None])
    print('一致性后处理完成。平滑像元:', int((qc & 1).sum()), ' 不可逆修正:', int((qc & 2).sum()))

if __name__ == '__main__':
    finals = []
    for y in YEARS:
        f = mosaic_year(y)
        print(y, '->', f)
        if f:
            finals.append(f)
    if len(finals) == len(YEARS):
        consistency()
    print('完成')

