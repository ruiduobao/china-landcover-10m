# -*- coding: utf-8 -*-
"""s1b_dem60.py — 为抽样点取 60 m 地形（Copernicus GLO-30，Planetary Computer 签名 COG 分块读取）
* 背景：项目内 1 km SRTM 镶嵌（srtm1000_chn.tif）在 121°E 以东有大片 nodata（东北 DNF 区 100% 缺失），
        无法用于坡向 OR 验收（G2）。故改用 Copernicus GLO-30 原生 30 m，按 1°×1° 分块降采样到 60 m 计算
        坡度/坡向（与 doc75 的 30/60 m 口径一致；60 m 是坡向分析常用尺度，降采样用 GDAL 概览，非插值伪造）。
* 输入：labels/_stage1_points.parquet
* 规则源：技术文档/76 G2；坡向 ESRI 分支（90−a / 450−a，见 代码/13.针阔可分性攻关/topo_corr_lib.py）
* 门槛：坡向只在 slope>5° 参与 OR；计算失败的点留 NaN 并在报告单列
* 输出：labels/_stage1_points_topo.parquet（追加 dem60_elev/slope/aspect/aspect_band/dem60_src）
        labels/_dem60_blocks.json（逐 1° 块读取状态，可复跑）
* 用法：python s1b_dem60.py [--workers 6] [--res 1800] [--force]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

BASE = ('https://elevationeuwest.blob.core.windows.net/copernicus-dem/COP30_hh/'
        'Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def _url(lon0, lat0):
    ns = 'N' if lat0 >= 0 else 'S'
    ew = 'E' if lon0 >= 0 else 'W'
    return BASE.format(ns=ns, lat=abs(lat0), ew=ew, lon=abs(lon0))


def read_block(lon0, lat0, res):
    """读 1°×1° 块并计算 60 m 坡度/坡向。返回 (elev, slope, asp, tfm) 或 None。"""
    import planetary_computer
    url = planetary_computer.sign(_url(lon0, lat0))
    with rasterio.open('/vsicurl/' + url) as s:
        a = s.read(1, out_shape=(res, res))
        tfm = s.transform * s.transform.scale(s.width / res, s.height / res)
    a = a.astype('float32')
    a[a < -1000] = np.nan
    slope, asp = C._terrain(a, tfm)
    return a, slope, asp, tfm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--res', type=int, default=1800, help='每 1° 块的栅格边长（1800=60 m）')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    src = os.path.join(C.LAB, '_stage1_points.parquet')
    dst = os.path.join(C.LAB, '_stage1_points_topo.parquet')
    logp = os.path.join(C.LAB, '_dem60_blocks.json')
    d = pd.read_parquet(src)
    d['blk'] = (np.floor(d.lon).astype(int) * 1000 + np.floor(d.lat).astype(int))
    blocks = sorted(d.blk.unique())
    emit('点 %d / 1°块 %d' % (len(d), len(blocks)))

    state = json.load(open(logp, encoding='utf-8')) if os.path.isfile(logp) else {}
    elev = np.full(len(d), np.nan, dtype='float32')
    slope = np.full(len(d), np.nan, dtype='float32')
    asp = np.full(len(d), np.nan, dtype='float32')

    def work(b):
        lon0 = int(b // 1000); lat0 = int(b % 1000)
        t0 = time.time()
        try:
            e, s, a_, tfm = read_block(lon0, lat0, a.res)
            return b, (e, s, a_, tfm), round(time.time() - t0, 1), ''
        except Exception as ex:
            return b, None, round(time.time() - t0, 1), '%s: %s' % (type(ex).__name__, str(ex)[:120])

    todo = [b for b in blocks if state.get(str(b), {}).get('ok') is not True] if not a.force else blocks
    emit('待取块 %d（已完成 %d）' % (len(todo), len(blocks) - len(todo)))
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, b): b for b in todo}
        for f in as_completed(futs):
            b, res, dt, err = f.result()
            done += 1
            state[str(b)] = dict(ok=res is not None, secs=dt, err=err)
            if res is not None:
                e, s, a_, tfm = res
                m = (d.blk == b).values
                sub = d[m]
                ev, ok1 = C.sample_raster(e, tfm, sub.lon.values, sub.lat.values, a.res, a.res)
                sv, ok2 = C.sample_raster(s, tfm, sub.lon.values, sub.lat.values, a.res, a.res)
                av, ok3 = C.sample_raster(a_, tfm, sub.lon.values, sub.lat.values, a.res, a.res)
                elev[m] = ev; slope[m] = sv; asp[m] = av
            if done % 10 == 0 or done == len(todo):
                emit('  块 %d/%d（%.0f%%）最近 %s %.1fs %s' % (done, len(todo), 100 * done / max(len(todo), 1), b, dt, err))
            if done % 25 == 0:
                json.dump(state, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    json.dump(state, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    d['dem60_elev'] = np.round(elev, 1)
    d['dem60_slope'] = np.round(slope, 2)
    d['dem60_aspect'] = np.round(asp, 1)
    d['aspect_band60'] = C.aspect_band(asp, slope, 5.0)
    bad = int((~np.isfinite(elev)).sum())
    emit('完成；DEM 缺失 %d 点（%.2f%%）' % (bad, 100 * bad / len(d)))
    d.drop(columns=['blk']).to_parquet(dst, index=False)
    emit('写 %s' % dst)


if __name__ == '__main__':
    main()
