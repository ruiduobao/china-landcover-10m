# -*- coding: utf-8 -*-
"""
ng6_build_samples.py — 把各家族过门槛样本拼成"可入库层"（本地，零 EECU）

做四件事
  1. 读 `ng_gate/ng_<spec>.parquet`（每家族一份）+ 试点留存 `_pilot_archive/*`（可选合并）；
  2. 归一化 → 母库同构元数据：
       row_id 用 **12,000,000+ 独立号段**（9,000,000+ 已被稀有类补样占用，勿冲突）
       src='ng_<family>'，src_conf 按"源数"给分（3 源 0.85 / 2 源 0.75）
       tier='external'，sample_type='core'，weight=1.0
       valid_from=2020 / valid_to=2022（源产品基准年 2021 的 ±1 年保守窗；不做跨 8 年外推，
         正是 FCS10 跨年毒性的教训）
  3. 与母库同类点做 500 m 近邻去重（保留新点：其来源更独立）；
  4. 产出：
       ng_train.parquet          —— 归一化样本层（供入库与断点续跑）
       ng_chunks_index.parquet   —— 年度嵌入提取索引（row_id/lon/lat/class_new/year/chunk_id）
       ng_build_summary.json     —— 各家族计数、去重删除数、年度展开数

用法: python ng6_build_samples.py [--keep-dups] [--years 2020,2021,2022]
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

PROJ = P.PROJ
MB = os.path.join(P.KB, '数据/本地处理/全国清洗训练')
RARE = os.path.join(MB, '稀有类补样/r7_rare_samples.parquet')
MOTHER = os.path.join(P.PROJ, '数据/本地处理/样本重建/r7_train.parquet')

ROW0_MIN = 12_000_000    # 下一代补样 row_id 号段下限（避开 0..2.2M 母库与 9,000,000+ 稀有类）
# ⚠️ 起点必须动态取"母库最大 row_id + 1"：第二轮并入若仍从 12,000,000 起会与第一轮撞号
#    （2026-09-14 实测：ng8 以 row_id 冲突中止，属预期保护）
CHUNK = 3000
DEDUP_M = 500
CONF = {3: 0.85, 2: 0.75, 4: 0.85, 5: 0.9}      # 按家族源数给 src_conf


def mother_xy():
    m = pd.read_parquet(MOTHER, columns=['lon', 'lat', 'class_new'])
    return m


def dedup_against(mother, df):
    """与母库同类点做 500 m 近邻去重（保留新点）"""
    out = []
    for cls, g in df.groupby('class_new'):
        mg = mother[mother.class_new == cls]
        if not len(mg):
            out.append(g)
            continue
        lat0 = np.radians(g.lat.mean())
        kx = 111.32 * np.cos(lat0)
        mx = np.c_[mg.lon.to_numpy() * kx, mg.lat.to_numpy() * 111.32]
        gx = np.c_[g.lon.to_numpy() * kx, g.lat.to_numpy() * 111.32]
        tree = cKDTree(mx)
        d, _ = tree.query(gx, k=1)
        out.append(g[d > DEDUP_M / 1000.0])
    return pd.concat(out, ignore_index=True) if out else df.iloc[:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep-dups', action='store_true', help='不做母库去重（默认去重）')
    ap.add_argument('--include-pilot', action='store_true', default=True)
    ap.add_argument('--years', default='2020,2021,2022')
    a = ap.parse_args()
    years = [int(v) for v in a.years.split(',')]
    t0 = time.time()

    parts, stat = [], {}
    for spec in list(S.SPECS):
        files = sorted(glob.glob(os.path.join(P.GATE, f'ng_{spec}.parquet')))
        if a.include_pilot:
            files += sorted(glob.glob(os.path.join(P.GATE, '_pilot_archive', f'ng_{spec}_pilot.parquet')))
        if not files:
            continue
        fs = []
        for f in files:
            d = pd.read_parquet(f)
            if not len(d):
                continue
            d = d[['lon', 'lat', 'class_new']].copy()
            d['src'] = f'ng_{spec}'
            d['_file'] = os.path.basename(f)
            fs.append(d)
        if not fs:
            continue
        d = pd.concat(fs, ignore_index=True)
        n_src = len(S.SPECS[spec].get('sources', [])) or 2
        d['src_conf'] = CONF.get(n_src, 0.8)
        d['spec'] = spec
        stat[spec] = {'gate_files': len(fs), 'rows_raw': int(len(d))}
        parts.append(d)
    if not parts:
        raise SystemExit('ng_gate 下没有任何成品，先跑批量采样与 ng9_qa')
    df = pd.concat(parts, ignore_index=True)
    df['lon'] = df.lon.astype(float).round(6)
    df['lat'] = df.lat.astype(float).round(6)

    # 跨家族/跨文件去重（同坐标同类别只留一条）
    n0 = len(df)
    df = df.drop_duplicates(subset=['lon', 'lat', 'class_new']).reset_index(drop=True)
    n_dup = n0 - len(df)

    # 母库去重
    n_before_mb = len(df)
    if not a.keep_dups:
        mb = mother_xy()
        df = dedup_against(mb, df).reset_index(drop=True)
    n_dedup_mb = n_before_mb - len(df)

    df = df.sort_values(['class_new', 'lon', 'lat']).reset_index(drop=True)
    row0 = max(ROW0_MIN, int(pd.read_parquet(MOTHER, columns=['row_id']).row_id.max()) + 1)
    df.insert(0, 'row_id', np.arange(row0, row0 + len(df), dtype=np.int64))
    df['tier'] = 'external'
    df['sample_type'] = 'core'
    df['weight'] = np.float32(1.0)
    df['valid_from'] = 2020
    df['valid_to'] = 2022
    df['qc_scope'] = 'ng'
    df['src'] = df['src'].astype(str)
    cols = ['row_id', 'lon', 'lat', 'class_new', 'src', 'src_conf', 'tier',
            'sample_type', 'weight', 'valid_from', 'valid_to', 'qc_scope']
    ng = df[cols].copy()

    # 年度嵌入提取索引
    # ⚠️ 必须**按点分块**（同一 chunk 内包含该点的全部年份），不能按 (year,row_id) 排序后分块 ——
    # 否则每个 chunk 只剩单一年份，提取器只会拉那一年，后续按 row_id 匹配嵌入块也会串号
    # （2026-09-14 实测踩到：9 个块全部只含 2020，且 row_id 范围互相重复）。
    base = ng[['row_id', 'lon', 'lat', 'class_new']].copy()
    base['chunk_id'] = np.arange(len(base)) // CHUNK
    idx = []
    for y in years:
        f = base[(ng.valid_from.to_numpy() <= y) & (ng.valid_to.to_numpy() >= y)].copy()
        f['year'] = y
        idx.append(f)
    if idx:
        index = pd.concat(idx, ignore_index=True)
    else:
        index = base.assign(year=years[0])

    P.ensure_all(list(S.SPECS))
    ng.to_parquet(os.path.join(P.NG, 'ng_train.parquet'), index=False)
    index.to_parquet(os.path.join(P.NG, 'ng_chunks_index.parquet'), index=False)

    for spec in stat:
        sub = ng[ng.src == f'ng_{spec}']
        stat[spec].update(rows_final=int(len(sub)),
                          classes=sorted(int(v) for v in sub.class_new.unique()))
    summ = {'time': time.strftime('%Y-%m-%d %H:%M'), 'row0': int(row0),
            'rows_raw': int(n0), 'dup_cross_family': int(n_dup),
            'dedup_vs_mother': int(n_dedup_mb), 'rows_final': int(len(ng)),
            'years': years, 'point_years': int(len(index)),
            'chunks': int(index.chunk_id.max()) + 1 if len(index) else 0,
            'by_family': stat,
            'by_class': {int(k): int(v) for k, v in ng.class_new.value_counts().items()},
            'src_conf': {str(k): v for k, v in CONF.items()},
            'validity': '2020-2022（源基准年 2021 的 ±1 年保守窗）'}
    json.dump(summ, open(os.path.join(P.NG, 'ng_build_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'原始 {n0:,} → 跨家族去重 -{n_dup:,} → 母库 500m 去重 -{n_dedup_mb:,} '
          f'→ 可入库 {len(ng):,} 点')
    print(f'年度索引: {len(index):,} point-years / {summ["chunks"]} 块（{CHUNK}/块, 年份 {years}）')
    print('按家族:', {k: v['rows_final'] for k, v in stat.items()})
    print('按类别:', summ['by_class'])
    print('输出:', os.path.join(P.NG, 'ng_train.parquet'))
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
