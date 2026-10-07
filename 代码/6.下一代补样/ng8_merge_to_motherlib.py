# -*- coding: utf-8 -*-
"""
ng8_merge_to_motherlib.py — 把下一代补样层并入母库与年度索引（**可回滚，先快照**）

设计要点（为什么不另起一套目录）
  e2b 的输入 = 母库 validity + chunks_index_r7 + emb_parts_r7/chunkr7_*.parquet。
  只要把下一代点**按同一约定**补进这三处（新 row_id 号段 + 新 chunk_id + 同名嵌入块），
  后续 e2b → e7 → e6 → e8 **一行代码不改**就能把新样本纳入全链，且口径完全一致。
  这比"再做一个平行目录"更不容易出现两套口径漂移。

动作
  0. 快照（默认开启）：母库/validity/索引/QC 表/年度子集 → 数据/备份/母库_并入下一代前_<日期>/
  1. r7_train.parquet          追加 ng 行（列对齐，缺列补 NaN）
  2. r7_train_validity.parquet 追加 ng 行（带 valid_from/valid_to/qc_scope='ng'）
  3. chunks_index_r7.parquet   追加 ng point-years，chunk_id 从 (max+1) 起连续编
  4. emb_parts_r7/chunkr7_<新cid>.parquet ← ng_emb/<前缀><cid>.parquet（重命名对齐）
  5. 校验：row_id 唯一、chunk 数与嵌入块数一致、母库行数 = 原行数 + ng 行数
  6. 写 merge_report.json（去重、跳过、校验结果）

用法: python ng8_merge_to_motherlib.py [--no-snapshot] [--dry]
"""
import os
import re
import sys
import glob
import json
import time
import shutil
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P

PROJ = P.PROJ
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
MB = os.path.join(P.KB, '数据/本地处理/全国清洗训练')
TRAIN = os.path.join(WORKR, 'r7_train.parquet')
VALID = os.path.join(WORKR, 'r7_train_validity.parquet')
IDX = os.path.join(MB, 'chunks_index_r7.parquet')
PARTS = os.path.join(MB, 'emb_parts_r7')
QC = os.path.join(MB, 'sample_year_qc.parquet')
BACKUP = os.path.join(PROJ, '数据', '备份',
                      '母库_并入下一代前_' + time.strftime('%Y%m%d'))
CN_COLS = ['class_raw', 'tier', 'year', 'src', 'src_conf', 'agree_n', 'border',
           'stab_years', 'wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0',
           'province', 'urban', '_fcs10_match', '_crop_built_frac', '_prio',
           'train_weight']


def snapshot():
    os.makedirs(BACKUP, exist_ok=True)
    done = []
    for src in (TRAIN, VALID, IDX, QC):
        if os.path.isfile(src):
            dst = os.path.join(BACKUP, os.path.basename(src))
            shutil.copy2(src, dst)
            done.append((os.path.basename(src), os.path.getsize(dst)))
    print(f'快照 {len(done)} 个文件 → {BACKUP}')
    for n, s in done:
        print(f'  {s/1e6:>10.1f} MB  {n}')
    if not done:
        print('  ⚠️ 没有任何文件被快照（路径可能不对）——中止以策安全')
        raise SystemExit(2)
    return BACKUP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-snapshot', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    ng_fp = os.path.join(P.NG, 'ng_train.parquet')
    ng_idx_fp = os.path.join(P.NG, 'ng_chunks_index.parquet')
    for f in (ng_fp, ng_idx_fp):
        if not os.path.isfile(f):
            raise SystemExit(f'缺少 {f}（先跑 ng6_build_samples.py）')
    ng = pd.read_parquet(ng_fp)
    ng_idx = pd.read_parquet(ng_idx_fp)
    emb_dir = os.path.join(P.NG, 'ng_emb')
    emb_files = sorted(glob.glob(os.path.join(emb_dir, '*.parquet')))
    print(f'待并入: {len(ng):,} 点 / {len(ng_idx):,} point-years / 嵌入块 {len(emb_files)}')
    if not emb_files:
        raise SystemExit('ng_emb 下没有嵌入块（先跑 ng7_extract.py）')

    rep = {'time': time.strftime('%Y-%m-%d %H:%M'),
           'ng_points': int(len(ng)), 'ng_point_years': int(len(ng_idx)),
           'ng_emb_blocks': len(emb_files)}

    if not a.dry and not a.no_snapshot:
        rep['backup_dir'] = snapshot()

    tr = pd.read_parquet(TRAIN)
    va = pd.read_parquet(VALID)
    idx = pd.read_parquet(IDX)
    rep['before'] = {'train': int(len(tr)), 'validity': int(len(va)),
                     'index': int(len(idx)), 'max_chunk': int(idx.chunk_id.max())}

    # row_id 冲突检查（新号段必须与母库/旧索引无交集）
    overlap = np.intersect1d(ng.row_id.to_numpy(), tr.row_id.to_numpy())
    if len(overlap):
        raise SystemExit(f'row_id 与新号段冲突 {len(overlap)} 条（前 5: {overlap[:5]}），中止')
    dup_in_ng = ng.row_id.duplicated().sum()
    if dup_in_ng:
        raise SystemExit(f'ng 层内部 row_id 重复 {int(dup_in_ng)} 条，中止')
    rep['checks'] = {'row_id_overlap': 0, 'dup_in_ng': 0}

    # ---- 1/2) 母库 + validity ----
    for c in CN_COLS:
        if c not in ng.columns:
            ng[c] = np.nan
    tr2 = pd.concat([tr, ng[list(tr.columns)]], ignore_index=True)
    for c in ('valid_from', 'valid_to', 'qc_scope'):
        if c not in ng.columns:
            ng[c] = np.nan
    va2 = pd.concat([va, ng[list(va.columns)]], ignore_index=True)

    # ---- 3) 年度索引：chunk_id 从旧 max+1 起连续编 ----
    base = int(idx.chunk_id.max()) + 1
    # 直接沿用 ng6 的分块（**按点分块**：一个 chunk 含该点全部年份），只做 id 偏移。
    # 切勿在这里重新按行序切块 —— 那会把点-年份打散，同一 chunk 只剩单年（2026-09-14 踩过）。
    ni = ng_idx.sort_values(['chunk_id', 'year', 'row_id']).reset_index(drop=True)
    ni['chunk_id'] = base + ni['chunk_id'].to_numpy(np.int64)
    rep['new_chunk_range'] = [int(ni.chunk_id.min()), int(ni.chunk_id.max())]
    idx2 = pd.concat([idx[list(idx.columns)], ni[list(idx.columns)]], ignore_index=True)

    # ---- 4) 嵌入块重命名对齐 ----
    #  ng7 产出文件名形如 <前缀><cid>.parquet，其中 cid 是 ng_chunks_index 的 chunk_id；
    #  这里按 row_id 集合做**内容匹配**重命名，避免依赖文件名前缀。
    want = {int(c): set(ni[ni.chunk_id == c].row_id.tolist())
            for c in sorted(ni.chunk_id.unique())}
    # 账号专属提取器把列名也做了差异化（<词干>_rid / _embyr / _cid），
    # 复制进 emb_parts_r7 前必须**归一化**回 row_id/emb_year/chunk_id + A00..A63，
    # 否则 e2b 读不了（它按母库既定列名解析）。
    import pyarrow.parquet as pq
    FE = [f'A{i:02d}' for i in range(64)]

    def idcol_of(f):
        names = pq.ParquetFile(f).schema_arrow.names
        if 'row_id' in names:
            return 'row_id', ('emb_year' if 'emb_year' in names else None),                    ('chunk_id' if 'chunk_id' in names else None)
        rid = next(c for c in names if c.endswith('_rid'))
        embyr = next((c for c in names if c.endswith('_embyr')), None)
        cid = next((c for c in names if c.endswith('_cid')), None)
        return rid, embyr, cid

    plan = []
    for f in emb_files:
        ic, _, _ = idcol_of(f)
        d = pd.read_parquet(f, columns=[ic])
        plan.append((f, set(int(v) for v in d[ic].tolist())))
    assigned, missing = {}, []
    for cid, ids in want.items():
        hit = [f for f, fs in plan if fs & ids]
        if hit:
            assigned[cid] = hit[0]
        else:
            missing.append(cid)
    rep['emb_blocks_assigned'] = len(assigned)
    rep['emb_blocks_missing'] = len(missing)
    if missing:
        print(f'⚠️ {len(missing)} 个块没有对应嵌入文件（其点不会进入年度子集）')

    if a.dry:
        print('dry-run：未写任何文件')
        json.dump(rep, open(os.path.join(P.NG, 'merge_report.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        return

    tr2.to_parquet(TRAIN, index=False)
    va2.to_parquet(VALID, index=False)
    idx2.to_parquet(IDX, index=False)
    os.makedirs(PARTS, exist_ok=True)
    for cid, f in assigned.items():
        ic, iy, icd = idcol_of(f)
        d = pd.read_parquet(f)
        d = d.rename(columns={ic: 'row_id'})
        if iy:
            d = d.rename(columns={iy: 'emb_year'})
        if icd:
            d = d.rename(columns={icd: 'chunk_id'})
        missing = [c for c in FE if c not in d.columns]
        if missing:
            raise SystemExit(f'{os.path.basename(f)} 缺嵌入波段 {missing[:4]}…')
        d['chunk_id'] = int(cid)
        d[['row_id', 'emb_year', 'chunk_id'] + FE].to_parquet(
            os.path.join(PARTS, f'chunkr7_{cid:04d}.parquet'), index=False)

    rep['after'] = {'train': int(len(tr2)), 'validity': int(len(va2)),
                    'index': int(len(idx2)), 'max_chunk': int(idx2.chunk_id.max())}
    rep['checks'].update({
        'train_delta': int(len(tr2) - len(tr)),
        'validity_delta': int(len(va2) - len(va)),
        'index_delta': int(len(idx2) - len(idx)),
        'train_rowid_unique': bool(not tr2.row_id.duplicated().any()),
        'index_rowid_year_unique': bool(not idx2.duplicated(['row_id', 'year']).any()),
        'emb_blocks_expected': int(len(want)),
        'emb_blocks_copied': len(assigned)})
    rep['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(rep, open(os.path.join(P.NG, 'merge_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ('before', 'after', 'checks')},
                     ensure_ascii=False, indent=1))
    print('报告:', os.path.join(P.NG, 'merge_report.json'))
    print(f'({rep["elapsed_min"]} min)  → 现在可以跑 e2b → e7 → e6 → e8')


if __name__ == '__main__':
    main()
