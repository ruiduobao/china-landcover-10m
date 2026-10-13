# -*- coding: utf-8 -*-
"""o1b_fcs10_sample.py — 乔灌园地：GLC-FCS10 2023 园地码(12) 全国候选点采样（本地零 GEE）

* 输入：E:/data/全球土地覆盖数据/数据/GLC_FCS10/GLC_FCS10maps_2023_E0{70..125}-*/*.tif（E 盘 5° 瓦片）
        + F:/lc_work/prod5p_2023/xcomp/fcs10_unzip/*.tif（E130/E135 带，F 盘）
        数据/边界/china_100000_full.json（国界过滤）
* 规则源：GLC 码 12 = Tree/shrub cropland（园地）；抽取 1/12 抽稀读（≈108 m 网格）后取码 12 像元中心
* 门槛：逐瓦最多 3000 点入池；全国过滤后随机抽 60,000 点；报告每瓦命中
* 输出：data/orchard/cand_fcs10.parquet（lon,lat,frame,tile）
* 用法：python o1b_fcs10_sample.py
幂等：输出存在即跳过（--force 重写）。
"""
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import shape
from shapely import contains_xy

WORK = r'F:/lc_work/v31_exp'
BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
E_BASE = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10'
F_UNZIP = r'F:/lc_work/prod5p_2023/xcomp/fcs10_unzip'
OUT = os.path.join(WORK, 'data', 'orchard', 'cand_fcs10.parquet')
DEC = 12            # 抽稀倍数（≈108 m 网格）
CAP_TILE = 3000
N_FINAL = 60000
LAT_ROWS = ['N15', 'N20', 'N25', 'N30', 'N35', 'N40', 'N45', 'N50', 'N55']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def tile_list():
    ts = []
    for pkg in ['E070-E075', 'E080-E085', 'E090-E095', 'E100-E105', 'E110-E115', 'E120-E125']:
        d = os.path.join(E_BASE, 'GLC_FCS10maps_2023_' + pkg)
        if not os.path.isdir(d):
            emit('缺包 %s' % d)
            continue
        for f in glob.glob(os.path.join(d, '*.tif')):
            b = os.path.basename(f)
            if any(r in b for r in LAT_ROWS) and 'S' not in b.split('_')[-1]:
                ts.append(f)
    ts += sorted(glob.glob(os.path.join(F_UNZIP, '*.tif')))
    return ts


def main():
    import random
    random.seed(20261012)
    rng = np.random.default_rng(20261012)
    if os.path.exists(OUT) and '--force' not in sys.argv:
        emit('已存在 %s（--force 重写）' % OUT)
        return
    gj = json.load(open(BND, encoding='utf-8'))
    from shapely import make_valid
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        if not g.is_valid:
            g = make_valid(g)
        geoms.append(g)
    from shapely.ops import unary_union
    try:
        china = unary_union(geoms)
    except Exception as e:
        emit('union 失败（%s）→ 用 buffer(0) 逐件兜底' % str(e)[:60])
        china = unary_union([g.buffer(0) for g in geoms])
    ts = tile_list()
    emit('瓦片 %d 个' % len(ts))
    lonl, latl, tl = [], [], []
    for i, f in enumerate(ts):
        try:
            with rasterio.open(f) as ds:
                a = ds.read(1, out_shape=(ds.height // DEC, ds.width // DEC))
                rr, cc = np.nonzero(a == 12)
                if len(rr) == 0:
                    continue
                if len(rr) > CAP_TILE:
                    sel = rng.choice(len(rr), CAP_TILE, replace=False)
                    rr, cc = rr[sel], cc[sel]
                # 抽稀网格 → 原图坐标
                r0, c0 = rr * DEC + DEC // 2, cc * DEC + DEC // 2
                xs, ys = rasterio.transform.xy(ds.transform, r0, c0, offset='center')
                lonl += list(np.atleast_1d(xs))
                latl += list(np.atleast_1d(ys))
                tl += [os.path.basename(f)] * len(r0)
        except Exception as e:
            emit('  瓦片失败 %s: %s' % (os.path.basename(f), str(e)[:80]))
        if (i + 1) % 20 == 0:
            emit('  进度 %d/%d，累计命中 %d' % (i + 1, len(ts), len(lonl)))
    lon = np.asarray(lonl); lat = np.asarray(latl)
    inb = (lon >= 73) & (lon <= 136) & (lat >= 18) & (lat <= 54)
    lon, lat = lon[inb], lat[inb]
    tl = [t for t, k in zip(tl, inb) if k]
    emit('抽稀命中 %d，国界 bbox 内 %d' % (len(lonl), len(lon)))
    keep = contains_xy(china, lon, lat)
    lon, lat, tl = lon[keep], lat[keep], [t for t, k in zip(tl, keep) if k]
    emit('国内 %d' % len(lon))
    idx = rng.permutation(len(lon))[:N_FINAL]
    df = pd.DataFrame({'lon': lon[idx], 'lat': lat[idx],
                       'frame': 'fcs10_12', 'tile': [tl[i] for i in idx]})
    df.to_parquet(OUT, index=False)
    emit('候选 %d → %s（瓦片数 %d）' % (len(df), OUT, df.tile.nunique()))


if __name__ == '__main__':
    main()
