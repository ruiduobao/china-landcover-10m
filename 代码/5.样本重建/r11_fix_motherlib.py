# -*- coding: utf-8 -*-
"""
r11_fix_motherlib.py — 根治母库错位点问题（2026-09-09）
* 问题：r7_train/r7_train_validity 采用"行号即 row_id"的隐式约定，错位点不能删行。
* 根治：
  1. 给两个母库文件写入**显式 row_id 列**（0..N-1）；
  2. 删除 159 个省级白名单违规点（基线），row_id 显式保留 → 删行后编号不再漂移；
  3. 消费方（e2b/e6/e1b_prep/r10/e9）改为读显式列（本次同步修改）。
* 影响后：母库 2,241,785 → 2,241,626；chunks_index/嵌入/年度子集/QC 表不动（row_id 未变）。
"""
import os, sys, time
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
BASE = os.path.join(PROJ, '数据/本地处理/全国清洗训练')
TRAIN = os.path.join(WORKR, 'r7_train.parquet')
VALID = os.path.join(WORKR, 'r7_train_validity.parquet')
LIST_OUT = os.path.join(BASE, '错位样本剔除清单.parquet')


def main():
    t0 = time.time()
    p = pd.read_parquet(LIST_OUT)
    ids = np.sort(p.loc[p.origin == 'base_r7_train', 'row_id'].to_numpy(np.int64))
    print('基线错位点:', len(ids), flush=True)

    t = pd.read_parquet(TRAIN)
    v = pd.read_parquet(VALID)
    assert len(t) == len(v) and (t.lon.to_numpy() == v.lon.to_numpy()).all(), '行不对齐!'
    assert (t.class_new.to_numpy()[ids] ==
            p[p.origin == 'base_r7_train'].class_new.to_numpy()).all(), 'row_id 映射校验失败!'

    # 1) 显式 row_id（当前行号）
    t.insert(0, 'row_id', np.arange(len(t), dtype=np.int64))
    v.insert(0, 'row_id', np.arange(len(v), dtype=np.int64))

    # 2) 删除错位行（row_id 显式保留，编号不漂移）
    t2 = t[~t.row_id.isin(ids)].reset_index(drop=True)
    v2 = v[~v.row_id.isin(ids)].reset_index(drop=True)
    assert len(t2) == len(t) - len(ids) and len(v2) == len(v) - len(ids)
    t2.to_parquet(TRAIN, index=False)
    v2.to_parquet(VALID, index=False)
    print(f'母库: {len(t):,} → {len(t2):,}（-{len(ids)}）；validity 同步', flush=True)
    print('剩余 140/183/81:',
          {int(c): int(n) for c, n in
           t2[t2.class_new.isin([140, 183, 81])].class_new.value_counts().items()}, flush=True)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
