# -*- coding: utf-8 -*-
"""
e1b_prep_chunks.py — P2 前置：为 r7_train 构建年度嵌入提取索引
* 每个样本点只在其 valid_from..valid_to 窗口内的年份提取嵌入（P1.1 窗口复用）
* 块大小 ~12000 点/块（与旧 709 块实测稳定规模一致）；row_id 全局唯一，
  按 valid_from 窗口生成后顺序切块；三账号按 chunk_id % 3 静态分配
* 输出: 数据/本地处理/全国清洗训练/chunks_index_r7.parquet
* 用法: python e1b_prep_chunks.py
"""
import os, sys
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
OUT = os.path.join(PROJ, '数据/本地处理/全国清洗训练/chunks_index_r7.parquet')
CHUNK = 3000
YEARS = list(range(2017, 2025))

def main():
    tr = pd.read_parquet(os.path.join(WORKR, 'r7_train_validity.parquet'),
                         columns=['lon', 'lat', 'class_new', 'year', 'src',
                                  'valid_from', 'valid_to'])
    tr['pt_id'] = np.arange(len(tr))
    frames = []
    for y in YEARS:
        sel = tr[(tr.valid_from <= y) & (tr.valid_to >= y)]
        f = sel[['pt_id', 'lon', 'lat', 'class_new']].copy()
        f['year'] = y
        frames.append(f)
        print(f'y{y}: {len(f):,}', flush=True)
    idx = pd.concat(frames, ignore_index=True)
    idx = idx.rename(columns={'pt_id': 'row_id'})
    idx['chunk_id'] = np.arange(len(idx)) // CHUNK
    idx.to_parquet(OUT, index=False)
    n_chunk = int(idx.chunk_id.max()) + 1
    print(f'合计 point-years {len(idx):,} → {n_chunk} 块（~{CHUNK}/块）')
    print('按年块数:', idx.groupby("year").chunk_id.nunique().to_dict())
    print('输出:', OUT)

if __name__ == '__main__':
    main()

