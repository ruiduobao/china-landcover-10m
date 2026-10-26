# -*- coding: utf-8 -*-
"""oe2_sample_points — 按点表批量抽取 OlmoEarth 嵌入（本机直连版）
* 输入：parquet/csv 点表，须含 lon / lat 列
* 规则源：oe0_conf（原生 S3、反量化、覆盖矩阵）
* 门槛：自动拒绝 COVERAGE_CN 非 full 的年份（2017–2020 中国无数据、2021 仅部分）
* 输出：<out>.parquet（原表 + oe_000..oe_127 float32 + oe_valid bool + oe_zone）
* 用法：
    python oe2_sample_points.py --pts pts.parquet --year 2024 --out out.parquet --workers 16
* ★ 关键实测（2026-10-26，见 技术文档/87）：
    - 读 1 个像元必须拉 1 个内块 = 64×256×256 int8 = 4.19 MB。逐点读＝每点 4.19 MB。
    - 母库 2,060,577 点散落在 1,070,666 个内块里（平均 1.9 点/块）
      → 逐点读 8.64 TB，按块聚簇 4.49 TB，整幅下载 4.80 TB。
    - 结论：**本项目这种稀疏样本，点抽取不比整幅下载省**。
      真要全国样本特征，优先考虑 oe3（整幅下 shard）后本地取样。
      本脚本用于点数少（<5 万）或点数高度聚集的场景。
    - 已改为**按 shard 聚合读取**（一个 shard 一次请求取里面所有点），
      比逐点读省，但仍受"点太散"限制。
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe0_conf import (YEARS, N_DIM, NODATA, SHARD_PX, CHUNK_PX,
                      open_root, lonlat_to_zone, COVERAGE_CN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pts', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8, help='并发 shard 数（实测 8–16 最佳）')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--nband', type=int, default=N_DIM, choices=[64, 128])
    ap.add_argument('--endpoint', choices=['s3', 'cf'], default='s3')
    ap.add_argument('--mode', choices=['shard', 'chunk'], default='shard',
                    help='shard=按 2048² 聚合（默认，省请求）; chunk=按 256² 聚合')
    a = ap.parse_args()

    cov = COVERAGE_CN.get(a.year)
    if cov != 'full':
        print(f'[拒绝] {a.year} 中国覆盖率={cov}。可用年份: '
              f'{[y for y,v in COVERAGE_CN.items() if v=="full"]}')
        return 2

    df = (pd.read_parquet(a.pts) if a.pts.endswith('.parquet')
          else pd.read_csv(a.pts, encoding='utf-8-sig'))
    if a.limit:
        df = df.head(a.limit).copy()
    if not {'lon', 'lat'} <= set(df.columns):
        print('[错误] 点表缺 lon/lat'); return 2
    print(f'读入 {len(df):,} 点 ← {a.pts}')

    from oe0_conf import S3_URL_HTTP, S3_URL_CF
    root = open_root(S3_URL_HTTP if a.endpoint == 's3' else S3_URL_CF)

    # ── 1. 全部点映射到 (zone, ix, iy) ────────────────────────────────
    df['oe_zone'] = [lonlat_to_zone(v) for v in df['lon'].values]
    ix_all = np.full(len(df), -1, np.int64)
    iy_all = np.full(len(df), -1, np.int64)
    for z in np.unique(df['oe_zone'].values):
        m = (df['oe_zone'].values == z)
        g = root[f'utm{z:02d}']
        xa, ya = g['x'][:], g['y'][:]
        tr = Transformer.from_crs('EPSG:4326', f'EPSG:326{z:02d}', always_xy=True)
        x, y = tr.transform(df['lon'].values[m], df['lat'].values[m])
        ix_all[m] = np.clip(np.searchsorted(xa, x), 0, len(xa) - 1)
        iy_all[m] = np.clip(np.searchsorted(-ya, -y), 0, len(ya) - 1)

    # ── 2. 按 shard / chunk 分组 ─────────────────────────────────────
    blk = SHARD_PX if a.mode == 'shard' else CHUNK_PX
    groups = defaultdict(list)
    for i in range(len(df)):
        z = int(df['oe_zone'].iat[i])
        groups[(z, int(iy_all[i]) // blk, int(ix_all[i]) // blk)].append(i)
    print(f'聚合为 {len(groups):,} 个 {blk}×{blk} 块 '
          f'(平均 {len(df)/max(1,len(groups)):.1f} 点/块)')

    ti = a.year - YEARS[0]
    arr = np.full((len(df), a.nband), np.nan, np.float32)

    def fetch(key):
        z, by, bx = key
        idx = groups[key]
        y0, y1 = by * blk, min((by + 1) * blk, 1867776)
        x0, x1 = bx * blk, min((bx + 1) * blk, 98304)
        try:
            b = root[f'utm{z:02d}']['embeddings'][ti, 0:a.nband, y0:y1, x0:x1]
        except Exception:
            return idx, None
        return idx, (b, by * blk, bx * blk)

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); done = 0; nbytes = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for idx, res in ex.map(fetch, list(groups.keys()), timeout=None):
            done += 1
            if res is None:
                continue
            b, y0, x0 = res
            nbytes += b.nbytes
            ys = iy_all[idx] - y0
            xs = ix_all[idx] - x0
            v = b[:, ys, xs].T          # (n_pts, nband)
            m = v[:, 0] != NODATA
            arr[idx[m]] = v[m].astype(np.float32)
            if done % 200 == 0 or done == len(groups):
                el = time.time() - t0
                print(f'  {done:,}/{len(groups):,} 块  {el/60:.1f}min  '
                      f'已取 {nbytes/1e9:.1f} GB', flush=True)

    cols = [f'oe_{k:03d}' for k in range(a.nband)]
    out = df.drop(columns=['oe_zone']).copy()
    out[cols] = arr
    out['oe_valid'] = ~np.isnan(arr[:, 0])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    out.to_parquet(a.out, index=False)

    el = time.time() - t0
    ok = out['oe_valid'].sum()
    print(f'\n完成 {ok:,}/{len(df):,} 有效 ({ok/len(df)*100:.2f}%)  '
          f'用时 {el/60:.1f}min  传输 {nbytes/1e9:.2f} GB')
    print(f'写 {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
