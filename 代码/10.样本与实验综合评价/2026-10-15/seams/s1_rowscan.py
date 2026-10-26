# -*- coding: utf-8 -*-
"""s1_rowscan — 指定栅格行区间逐行统计（非零计数 / 预期覆盖掩膜内零值 / 最长零游程）

* 输入：任意 uint8 类别 GeoTIFF（mosaic2 总幅 / 省件 / rasters_batch 瓦片）
* 规则源：AGENTS 硬约束 3/14（nearest、类码 0=nodata）；预期覆盖掩膜由 plan placements 给出
* 门槛：只读；不做任何写入到成品目录；CSV/JSON 落 reports/quality_review_2026-10-15/seams/
* 输出：逐行 CSV
* 用法：python s1_rowscan.py --file <tif> --r0 111280 --r1 111340 [--plan <plan.json>] [--tag mos]
"""
import os
import sys
import json
import csv
import argparse

import numpy as np
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
OUTDIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_ROWS = 32


def expected_union_cols(placements, r, global_row0=0):
    """给定全局行 r，返回 placements 覆盖的列区间列表（已按 col 排序合并）。"""
    iv = []
    for p in placements:
        if p['row0'] <= r < p['row0'] + p['height']:
            iv.append((p['col0'], p['col0'] + p['width']))
    if not iv:
        return []
    iv.sort()
    out = [list(iv[0])]
    for c0, c1 in iv[1:]:
        if c0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], c1)
        else:
            out.append([c0, c1])
    return [tuple(x) for x in out]


def longest_zero_run(a):
    """最长连续 0 游程长度（bool 数组，True=零）。"""
    if not a.any():
        return 0
    b = np.concatenate(([False], a, [False]))
    d = np.diff(b.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def scan(fp, r0, r1, plan=None, tag='', write=True):
    placements = []
    frame = None
    if plan:
        d = json.load(open(plan, encoding='utf-8'))
        placements = d['placements']
        frame = d['frame']
    rows_out = []
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
        r0 = max(0, r0)
        r1 = min(H, r1)
        for c0 in range(r0, r1, CHUNK_ROWS):
            c1 = min(r1, c0 + CHUNK_ROWS)
            buf = ds.read(1, window=Window(0, c0, W, c1 - c0))
            nz = buf > 0
            for i in range(c1 - c0):
                r = c0 + i
                row = nz[i]
                n_tot = int(row.sum())
                rec = dict(file=os.path.basename(fp), tag=tag, row=r,
                           lat_top=ds.transform.f - r * abs(ds.transform.e),
                           lat_bot=ds.transform.f - (r + 1) * abs(ds.transform.e),
                           nonzero=n_tot)
                # 预期覆盖（plan 口径）内
                if placements:
                    exp = np.zeros(W, dtype=bool)
                    for a, b in expected_union_cols(placements, r):
                        exp[a:b] = True
                    n_exp = int(exp.sum())
                    miss = int((exp & ~row).sum())
                    rec.update(exp_cols=n_exp, exp_miss=miss,
                               exp_miss_pct=(100.0 * miss / n_exp) if n_exp else 0.0,
                               exp_internal_zero_run=longest_zero_run(exp & ~row) if n_exp else 0)
                # 全宽最长零游程（仅统计行内有数据的区间内部）
                if n_tot:
                    idx = np.where(row)[0]
                    seg = row[idx[0]:idx[-1] + 1]
                    rec['full_span'] = int(idx[-1] - idx[0] + 1)
                    rec['full_internal_zero_run'] = longest_zero_run(~seg)
                    rec['col_first'] = int(idx[0])
                    rec['col_last'] = int(idx[-1])
                else:
                    rec['full_span'] = 0
                    rec['full_internal_zero_run'] = 0
                    rec['col_first'] = -1
                    rec['col_last'] = -1
                rows_out.append(rec)
    if write and rows_out:
        nm = tag or os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(OUTDIR, 'rowscan_%s_%d_%d.csv' % (nm, r0, r1))
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print('WROTE', out)
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--r0', type=int, required=True)
    ap.add_argument('--r1', type=int, required=True)
    ap.add_argument('--plan', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--print', action='store_true')
    a = ap.parse_args()
    rows = scan(a.file, a.r0, a.r1, a.plan, a.tag)
    if a.print:
        for r in rows:
            print(r)


if __name__ == '__main__':
    main()
