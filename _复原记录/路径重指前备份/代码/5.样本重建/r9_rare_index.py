# -*- coding: utf-8 -*-
"""
r9_rare_index.py — P3.3 稀有类补样候选合并 + 嵌入提取索引
* 输入: r8_rare_candidates.parquet（3×3 纯净核心）+ r8b_rare_boundary.parquet（边界/CW）
* 处理: 统一列 → 与 r7_train 同类 100m 去重 → 每类按 class_balance_design.json 缺口封顶
        → 赋 row_id（9,000,000 起，避开 r7 的 0..2,238,646）→ 按年份有效期切块
* 有效期（P1.1 先验）: FCS10-2023 系 → 2022-2024；GMW/CW-2020 系 → 2019-2021
* 输出:
    数据/本地处理/全国清洗训练/稀有类补样/r7_rare_samples.parquet（含 row_id/weight）
    数据/本地处理/全国清洗训练/chunks_index_rare.parquet（row_id/lon/lat/class_new/year/chunk_id）
* 嵌入提取（主提取 4758 块完成后执行）:
    E1_IDX=数据/本地处理/全国清洗训练/chunks_index_rare.parquet \
    E1_OUT=数据/本地处理/全国清洗训练/emb_parts_rare \
    python 代码/4.全国清洗训练/e1c_worker_generic.py <账号>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
BASE = os.path.join(PROJ, '数据/本地处理/全国清洗训练')
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
OUT_DIR = os.path.join(BASE, '稀有类补样')
os.makedirs(OUT_DIR, exist_ok=True)
CHUNK = 3000
ROW0 = 9_000_000
GRID = 0.001        # ≈100m 去重

FCS10_SRC = ('fcs10_2023_rare', 'fcs10_2023_rare_boundary')


def load_any():
    fs = [os.path.join(WORKR, 'r8_rare_candidates.parquet'),
          os.path.join(WORKR, 'r8b_rare_boundary.parquet')]
    frames = []
    for f in fs:
        if os.path.isfile(f):
            d = pd.read_parquet(f)
            if 'sample_type' not in d:
                d['sample_type'] = 'core'
            if 'weight' not in d:
                d['weight'] = np.where(d.sample_type == 'core', 1.0, 0.6)
            frames.append(d)
            print('载入', os.path.basename(f), len(d), flush=True)
    if not frames:
        raise SystemExit('无候选文件')
    return pd.concat(frames, ignore_index=True)


def main():
    t0 = time.time()
    design = json.load(open(os.path.join(BASE, '样本诊断/class_balance_design.json'),
                            encoding='utf-8'))
    gaps = {int(r['class']): int(r['gap']) for r in design['E3_稀有类专项池']}
    print('缺口:', gaps, flush=True)

    df = load_any()
    df['class_new'] = df.class_new.astype(int)
    # 与 r7_train 同类 100m 去重（避免与现有训练点重叠）
    tr = pd.read_parquet(os.path.join(WORKR, 'r7_train.parquet'),
                         columns=['lon', 'lat', 'class_new'])
    tr['g'] = (np.floor(tr.lon / GRID).astype(np.int64) * 10_000_000 +
               np.floor(tr.lat / GRID).astype(np.int64))
    trk = set(zip(tr.class_new.astype(int), tr.g))
    df['g'] = (np.floor(df.lon / GRID).astype(np.int64) * 10_000_000 +
               np.floor(df.lat / GRID).astype(np.int64))
    before = len(df)
    df = df[~pd.Series(list(zip(df.class_new, df.g)), index=df.index).isin(trk)]
    print(f'与 r7_train 同类100m去重: {before:,} -> {len(df):,}', flush=True)

    # 每类封顶到缺口（核心优先于边界）
    keep = []
    for c, g in df.groupby('class_new'):
        g = g.assign(_p=np.where(g.sample_type == 'core', 0, 1))
        g = g.sample(frac=1.0, random_state=42).sort_values('_p', kind='stable')
        n = gaps.get(int(c), 2000)
        keep.append(g.head(n))
    df = pd.concat(keep, ignore_index=True).drop(columns=['_p', 'g'], errors='ignore')
    df = df.drop_duplicates(['lon', 'lat', 'class_new']).reset_index(drop=True)
    print('封顶后:', dict(df.class_new.value_counts()), flush=True)
    print('按类型:', dict(df.sample_type.value_counts()), flush=True)

    # row_id 与有效期
    df['row_id'] = (ROW0 + np.arange(len(df))).astype(np.int64)
    src = df.src.fillna('')
    is_fcs10 = src.str.startswith('fcs10')
    df['valid_from'] = np.where(is_fcs10, 2022, 2019)
    df['valid_to'] = np.where(is_fcs10, 2024, 2021)
    df = df[['row_id', 'lon', 'lat', 'class_new', 'class_raw', 'src', 'src_conf',
             'year', 'tier', 'agree_n', 'sample_type', 'weight',
             'valid_from', 'valid_to']]
    df.to_parquet(os.path.join(OUT_DIR, 'r7_rare_samples.parquet'), index=False)

    # 切块索引（按年独立切块，保证每块单一年份——worker 按块取年份影像）
    parts = []
    cid0 = 0
    for y in range(2017, 2025):
        m = (df.valid_from <= y) & (df.valid_to >= y)
        g = df[m][['row_id', 'lon', 'lat', 'class_new']].copy()
        if len(g) == 0:
            continue
        g['year'] = y
        g['chunk_id'] = cid0 + (np.arange(len(g)) // CHUNK)
        cid0 = int(g.chunk_id.max()) + 1
        parts.append(g)
    idx = pd.concat(parts, ignore_index=True)
    idx = idx[['row_id', 'lon', 'lat', 'class_new', 'year', 'chunk_id']]
    idx.to_parquet(os.path.join(BASE, 'chunks_index_rare.parquet'), index=False)
    print(f'point-years {len(idx):,}，块 {idx.chunk_id.nunique()} '
          f'(每块 ≤{CHUNK})', flush=True)
    assert idx.groupby('chunk_id').year.nunique().max() == 1, '块内混年！'
    print('按年:', dict(idx.year.value_counts().sort_index()), flush=True)
    print('输出:', os.path.join(OUT_DIR, 'r7_rare_samples.parquet'), flush=True)
    print('输出:', os.path.join(BASE, 'chunks_index_rare.parquet'), flush=True)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
