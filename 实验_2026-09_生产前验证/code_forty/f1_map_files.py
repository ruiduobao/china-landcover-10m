# -*- coding: utf-8 -*-
"""f1_map_files.py — ForTy v1 全量文件清单 → 地理范围映射表（只读 E 盘，不拷贝全量）

* 输入：E:/地理所/论文/博士论文/数据/ForTy_v1/32301177/raster_<MGRS>/*.tif（613 瓦片目录）
* 规则源：
  ① 每文件读**头信息**（rasterio 只读）：CRS / 尺寸 / 原生 bounds / 文件字节；
  ② bounds 用 rasterio.warp.transform_bounds 转 WGS84（EPSG:4326）→ lon/lat 四至与中心；
  ③ 命名解析：raster_<MGRS>_<YY>_<row>_<col>.tif（YY=20 表示 2020 年）；
  ④ 中国相关判定：WGS84 bbox ∩ 中国边界（china_100000_full.json，含 0.05° 缓冲）。
* 门槛：无（普查脚本）；失败文件记入 err 列，最后汇总缺失数。
* 输出：F:/lc_work/forty_exp/map/forty_file_index.csv（逐文件）
        F:/lc_work/forty_exp/map/forty_tile_summary.csv（逐瓦片：文件数/覆盖 bbox/是否中国）
        F:/lc_work/forty_exp/map/china_tiles.txt（中国相关瓦片码，供 f2 拷贝）
* 用法：python f1_map_files.py [--workers 8]
     幂等：已存在 index CSV 时跳过（--force 重算）。
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from shapely import contains_xy
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely import make_valid

SRC = r'E:/地理所/论文/博士论文/数据/ForTy_v1/32301177'
WORK = r'F:/lc_work/forty_exp'
OUT = os.path.join(WORK, 'map')
CHINA = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
PAT = re.compile(r'^raster_([0-9]{1,2}[A-Z])_(\d{2})_(\d+)_(\d+)\.tif$')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_china():
    gj = json.load(open(CHINA, encoding='utf-8'))
    geoms = []
    for ft in gj['features']:
        g = shape(ft['geometry'])
        geoms.append(g if g.is_valid else make_valid(g))
    return unary_union(geoms)


def one(fp):
    rel = os.path.relpath(fp, SRC).replace('\\', '/')
    tile = os.path.basename(os.path.dirname(fp))
    m = PAT.match(os.path.basename(fp))
    rec = {'file': os.path.basename(fp), 'tile': tile, 'relpath': rel,
           'year': int(m.group(2)) if m else -1,
           'row': int(m.group(3)) if m else -1, 'col': int(m.group(4)) if m else -1,
           'size_bytes': os.path.getsize(fp)}
    try:
        with rasterio.open(fp) as s:
            rec.update({'crs': str(s.crs), 'width': s.width, 'height': s.height,
                        'xmin': s.bounds.left, 'ymin': s.bounds.bottom,
                        'xmax': s.bounds.right, 'ymax': s.bounds.top})
            if str(s.crs) != 'EPSG:4326':
                l, b, r, t = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
            else:
                l, b, r, t = s.bounds
            rec.update({'lon_min': round(l, 6), 'lat_min': round(b, 6),
                        'lon_max': round(r, 6), 'lat_max': round(t, 6),
                        'lon_c': round((l + r) / 2, 6), 'lat_c': round((b + t) / 2, 6)})
    except Exception as e:
        rec['err'] = str(e)[:120]
    return rec


def one_tile(tile):
    """进程级任务：解析一个瓦片目录下的全部 tif（避免 GDAL 多线程 open 挂死）"""
    td = os.path.join(SRC, tile)
    files = [os.path.join(td, f) for f in sorted(os.listdir(td)) if f.lower().endswith('.tif')]
    return [one(f) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    IDX = os.path.join(OUT, 'forty_file_index.csv')
    if os.path.exists(IDX) and not a.force:
        emit('已存在映射表，跳过（--force 重算）')
        return
    tiles = sorted(d for d in os.listdir(SRC)
                   if d.startswith('raster_') and os.path.isdir(os.path.join(SRC, d)))
    emit('瓦片目录 %d 个，开始解析（%d 进程）' % (len(tiles), a.workers))
    t0, recs, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows in ex.map(one_tile, tiles, chunksize=2):
            recs += rows
            done += 1
            if done % 25 == 0:
                emit('  瓦片 %d/%d，文件 %d（%.1f 分钟）' % (
                    done, len(tiles), len(recs), (time.time() - t0) / 60))
    d = pd.DataFrame(recs)
    errs = int(d['err'].notna().sum()) if 'err' in d else 0
    emit('解析完成：%d 行，失败 %d（%.1f 分钟）' % (len(d), errs, (time.time() - t0) / 60))
    d['in_china'] = False
    ok = d.dropna(subset=['lon_min']).copy()
    if len(ok):
        china = load_china()
        cb = china.bounds
        cand = ok[(ok.lon_max >= cb[0] - 0.05) & (ok.lon_min <= cb[2] + 0.05) &
                  (ok.lat_max >= cb[1] - 0.05) & (ok.lat_min <= cb[3] + 0.05)]
        flag = np.zeros(len(ok), dtype=bool)
        pos = ok.index.get_indexer(cand.index)
        for i, row in zip(pos, cand.itertuples()):
            flag[i] = china.intersects(box(row.lon_min, row.lat_min, row.lon_max, row.lat_max))
        ok['in_china'] = flag
        d.loc[ok.index, 'in_china'] = ok['in_china'].to_numpy()
    d.to_csv(IDX, index=False, encoding='utf-8-sig')
    g = d.groupby('tile').agg(n_files=('file', 'size'),
                              size_gb=('size_bytes', lambda x: round(x.sum() / 1e9, 3)),
                              lon_min=('lon_min', 'min'), lat_min=('lat_min', 'min'),
                              lon_max=('lon_max', 'max'), lat_max=('lat_max', 'max'),
                              in_china=('in_china', 'max')).reset_index()
    g.to_csv(os.path.join(OUT, 'forty_tile_summary.csv'), index=False, encoding='utf-8-sig')
    ct = sorted(g[g.in_china].tile.tolist())
    open(os.path.join(OUT, 'china_tiles.txt'), 'w', encoding='utf-8').write('\n'.join(ct))
    emit('中国相关：瓦片组 %d 个，文件 %d 个（%.2f GB）' % (
        len(ct), int(d[d.in_china].shape[0]), d[d.in_china].size_bytes.sum() / 1e9))
    emit('→ %s' % IDX)


if __name__ == '__main__':
    main()
