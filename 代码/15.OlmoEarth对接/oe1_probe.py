# -*- coding: utf-8 -*-
"""oe1_probe — 连通性 / 覆盖 / 延迟体检（跑任何批量任务前先跑这个）
* 输入：无（内置中国 12 点探针 + 可传 --csv 自定义点）
* 规则源：oe0_conf（原生 S3 端点、覆盖矩阵）
* 门槛：任一探针点 2022–2025 全部 nodata 视为 FAIL
* 输出：stdout 覆盖矩阵表 + 可选 JSON（--json）
* 用法：
    python oe1_probe.py
    python oe1_probe.py --csv pts.csv --year 2024 --json probe.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe0_conf import (YEARS, N_DIM_64, NODATA, open_root, lonlat_to_zone)

# 中国 12 点：覆盖 9 个 UTM 带、湿润/干旱/高寒/热带
PROBES = [
    ('漠河', 122.5, 52.9), ('三江平原', 133.0, 47.5), ('呼伦贝尔', 119.7, 49.2),
    ('北京', 116.4, 39.9), ('延安', 109.5, 36.6), ('和田', 79.9, 37.1),
    ('拉萨', 91.1, 29.7), ('成都', 104.1, 30.7), ('武汉', 114.3, 30.6),
    ('广州', 113.3, 23.1), ('海南儋州', 109.5, 19.5), ('昆明', 102.7, 25.0),
]


def read_point(root, lon, lat, year, nband=N_DIM_64):
    """读单点单年嵌入。返回 (vec_float32_or_None, 用时秒)。"""
    z = lonlat_to_zone(lon)
    g = root[f'utm{z:02d}']
    xa, ya = g['x'][:], g['y'][:]
    x, y = Transformer.from_crs('EPSG:4326', f'EPSG:326{z:02d}',
                                always_xy=True).transform(lon, lat)
    ix = int(np.argmin(np.abs(xa - x)))
    iy = int(np.argmin(np.abs(ya - y)))
    t0 = time.time()
    blk = g['embeddings'][year - YEARS[0], 0:nband, iy:iy + 1, ix:ix + 1]
    dt = time.time() - t0
    v = blk[:, 0, 0]
    if int(v[0]) == NODATA:
        return None, dt, (z, ix, iy)
    return v, dt, (z, ix, iy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', help='自定义点 CSV（name,lon,lat）')
    ap.add_argument('--year', type=int, action='append',
                    help='只测指定年份，可重复；缺省测全部 2017-2025')
    ap.add_argument('--json', help='输出 JSON 路径')
    ap.add_argument('--endpoint', choices=['s3', 'cf'], default='s3',
                    help='s3=原生 AWS（推荐）; cf=Cloudflare 官方端点')
    a = ap.parse_args()

    from oe0_conf import S3_URL_HTTP, S3_URL_CF
    url = S3_URL_HTTP if a.endpoint == 's3' else S3_URL_CF
    print(f'端点: {url}')
    t0 = time.time()
    root = open_root(url)
    print(f'打开 Zarr 根组用时 {time.time()-t0:.1f}s；组数 {len(list(root.group_keys()))}')

    pts = PROBES
    if a.csv:
        import csv as _csv
        pts = []
        with open(a.csv, encoding='utf-8-sig') as f:
            for row in _csv.DictReader(f):
                pts.append((row['name'], float(row['lon']), float(row['lat'])))
    years = a.year or YEARS

    print()
    print(f"{'点位':<11s}{'zone':<7s}" + ''.join(f'{y:>7d}' for y in years))
    matrix = {}
    lat_all = []
    for nm, lon, lat in pts:
        row = {}
        cells = []
        for y in years:
            try:
                v, dt, (z, ix, iy) = read_point(root, lon, lat, y)
                row[y] = (v is not None)
                lat_all.append(dt)
                cells.append('      ✓' if v is not None else '      ·')
            except Exception as e:
                row[y] = None
                cells.append('      !')
                print(f'  [warn] {nm} {y}: {type(e).__name__} {str(e)[:60]}',
                      file=sys.stderr)
        matrix[nm] = row
        z = lonlat_to_zone(lon)
        print(f'{nm:<11s}utm{z:02d}  ' + ''.join(cells))

    print()
    ok = True
    for y in years:
        n = sum(1 for r in matrix.values() if r.get(y))
        tag = ''
        if y <= 2020 and n == 0:
            tag = '  ← 中国无覆盖（预期）'
        elif y == 2021 and 0 < n < len(matrix):
            tag = '  ← 部分覆盖'
        elif y >= 2022 and n < len(matrix):
            tag = f'  ← 异常见 0 覆盖点'
            ok = False
        print(f'  {y}: {n:2d}/{len(matrix)} 点有数据{tag}')

    if lat_all:
        arr = np.array(lat_all)
        print(f'\n单点读延迟: 中位 {np.median(arr):.2f}s  P90 {np.percentile(arr,90):.2f}s '
              f'最大 {arr.max():.2f}s  (n={len(arr)})')
        print('★ 每点要拉 1 个内块（64×256×256 int8 = 4.19 MB），故延迟受限于块下载而非带宽。')

    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'endpoint': url, 'matrix': matrix,
                       'latency_median_s': float(np.median(lat_all))}, f,
                      ensure_ascii=False, indent=2)
        print(f'已写 {a.json}')

    print(f'\n判定: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
