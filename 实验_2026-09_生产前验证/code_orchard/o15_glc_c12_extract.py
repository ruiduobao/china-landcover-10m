# -*- coding: utf-8 -*-
"""o15_glc_c12_extract.py — GLC 官方样本库(1°网格 CSV)抽取 class==12 乔灌园地样本

* 输入：E:/data/全球土地覆盖数据/数据/样本数据/GLC_Samples_Grid_2x2/Sample_Lon{lon}_Lat{lat}.csv
        （64,800 个 1°×1° 格文件，表头 lon,lat,class,year,tile_id；格名=左下角整数度）
* 规则源：lc_conf.py GLC 码表（GLC_FCS30D 原始码 12 = Tree/shrub cropland 乔灌园地）
* 门槛：中国 bbox = lon∈[73,136] × lat∈[18,54]（严格按点过滤）
* 输出：
    data/orchard/glc_samples_china_c12.csv      中国 bbox 内 class==12 点（lon,lat,class,year,tile_id,srcfile）
    data/orchard/glc_samples_c12_global.parquet 全球 class==12 点（诊断用，判断类是否真的存在及分布）
    results/orchard/c12_extract.json            计数/耗时/中国格 class 直方图
* 用法：python o15_glc_c12_extract.py [--workers 12]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

SRC = r'E:/data/全球土地覆盖数据/数据/样本数据/GLC_Samples_Grid_2x2'
WORK = r'F:/lc_work/v31_exp'
OUTD = os.path.join(WORK, 'data', 'orchard')
RESD = os.path.join(WORK, 'results', 'orchard')

CN_LON = (73.0, 136.0)     # 中国 bbox（含邻国边缘，报告须声明）
CN_LAT = (18.0, 54.0)
TARGET = 12                # GLC_FCS30D 原始码 12 = Tree/shrub cropland（乔灌园地）
NAME_RE = re.compile(r'^Sample_Lon(-?\d+)_Lat(-?\d+)\.csv$')

USECOLS = ['lon', 'lat', 'class', 'year', 'tile_id']
DTYPES = {'lon': 'float64', 'lat': 'float64', 'class': 'int16', 'year': 'int16', 'tile_id': 'str'}


def one_file(name):
    """读一个格文件 → (n_rows, n_c12, c12_df, cn_class_hist 或 None, lon/lat 边界, tile_id)"""
    m = NAME_RE.match(name)
    lon0, lat0 = int(m.group(1)), int(m.group(2))
    path = os.path.join(SRC, name)
    try:
        df = pd.read_csv(path, usecols=USECOLS, dtype=DTYPES)
    except Exception as e:  # 空文件 / 坏文件
        return (name, lon0, lat0, 0, 0, None, None,
                {'err': f'{type(e).__name__}: {e}'})

    n = len(df)
    hist = None
    # 中国格：记录全类直方图（诊断用）
    if (CN_LON[0] - 1 <= lon0 <= CN_LON[1]) and (CN_LAT[0] - 1 <= lat0 <= CN_LAT[1]):
        hist = df['class'].value_counts().to_dict()

    sub = df[df['class'] == TARGET]
    c12 = None
    if len(sub):
        c12 = sub[['lon', 'lat', 'class', 'year', 'tile_id']].copy()
        c12['srcfile'] = name
    bounds = (float(df['lon'].min()), float(df['lon'].max()),
              float(df['lat'].min()), float(df['lat'].max())) if n else None
    return (name, lon0, lat0, n, int(len(sub)), c12, bounds, hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0, help='仅前 N 个文件（冒烟测试用）')
    args = ap.parse_args()

    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(RESD, exist_ok=True)

    t0 = time.time()
    names = [n for n in os.listdir(SRC) if NAME_RE.match(n)]
    names.sort()
    if args.limit:
        names = names[:args.limit]
    print(f'[scan] 目录文件数 = {len(os.listdir(SRC))}，可解析格文件 = {len(names)}')
    t_list = time.time() - t0

    c12_parts, hist_total, errs = [], Counter(), []
    n_files_data = 0
    n_rows_total = 0
    out_of_cell = 0
    done = 0
    t1 = time.time()
    with Pool(args.workers) as pool:
        for (name, lon0, lat0, n, k, c12, bounds, hist) in pool.imap_unordered(
                one_file, names, chunksize=64):
            done += 1
            if isinstance(hist, dict) and 'err' in hist:
                errs.append((name, hist['err']))
            elif hist is not None:
                for kk, vv in hist.items():
                    hist_total[int(kk)] += int(vv)
            if n:
                n_files_data += 1
                n_rows_total += n
            if bounds:
                # 校验：点是否落在名义格内（容差 1e-6）
                if not (lon0 - 1e-6 <= bounds[0] <= bounds[1] <= lon0 + 1 + 1e-6 and
                        lat0 - 1e-6 <= bounds[2] <= bounds[3] <= lat0 + 1 + 1e-6):
                    out_of_cell += 1
            if c12 is not None:
                c12_parts.append(c12)
            if done % 10000 == 0:
                el = time.time() - t1
                print(f'[scan] {done}/{len(names)}  {el:.0f}s  ({done/el:.0f} files/s)', flush=True)
    t_scan = time.time() - t1
    print(f'[scan] 完成 {done} 文件，{t_scan:.0f}s，{done/t_scan:.0f} files/s，'
          f'有数据文件 {n_files_data}，总行数 {n_rows_total:,}，越界格 {out_of_cell}，错误 {len(errs)}')

    glob = pd.concat(c12_parts, ignore_index=True) if c12_parts else pd.DataFrame(
        columns=USECOLS + ['srcfile'])
    print(f'[global] class=={TARGET} 全球点数 = {len(glob):,}')

    # ---- 中国 bbox ----
    if len(glob):
        cn = glob[(glob.lon >= CN_LON[0]) & (glob.lon <= CN_LON[1]) &
                  (glob.lat >= CN_LAT[0]) & (glob.lat <= CN_LAT[1])].copy()
    else:
        cn = glob.copy()
    cn = cn.sort_values(['lat', 'lon']).reset_index(drop=True)
    out_csv = os.path.join(OUTD, 'glc_samples_china_c12.csv')
    cn.to_csv(out_csv, index=False, float_format='%.10g')
    print(f'[china] bbox={CN_LON}x{CN_LAT} 内 class=={TARGET} 点数 = {len(cn):,} → {out_csv}')

    out_pq = os.path.join(OUTD, 'glc_samples_c12_global.parquet')
    glob.to_parquet(out_pq, index=False)
    print(f'[global] 全球 class=={TARGET} → {out_pq}')

    # ---- 诊断：中国格全类直方图 ----
    hist_sorted = dict(sorted(hist_total.items(), key=lambda kv: -kv[1]))
    res = {
        'src': SRC,
        'n_files_total': len(os.listdir(SRC)),
        'n_files_parsed': len(names),
        'n_files_with_data': n_files_data,
        'n_rows_total': n_rows_total,
        'n_cells_out_of_nominal_bounds': out_of_cell,
        'n_errors': len(errs),
        'err_sample': errs[:5],
        't_listdir_s': round(t_list, 1),
        't_scan_s': round(t_scan, 1),
        't_total_s': round(time.time() - t0, 1),
        'target_class': TARGET,
        'n_c12_global': int(len(glob)),
        'n_c12_china_bbox': int(len(cn)),
        'china_bbox': {'lon': CN_LON, 'lat': CN_LAT},
        'china_all_class_hist': {str(k): int(v) for k, v in hist_sorted.items()},
        'workers': args.workers,
    }
    with open(os.path.join(RESD, 'c12_extract.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'[done] 总耗时 {res["t_total_s"]}s')
    print('[china class hist] ' + ', '.join(f'{k}:{v}' for k, v in list(hist_sorted.items())[:20]))


if __name__ == '__main__':
    main()
