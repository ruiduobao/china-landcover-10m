# -*- coding: utf-8 -*-
"""
lc15_aggregate_v2.py — 聚合五产品采样 → 样本底座 v2（置信分层 + 边界带 + 稳定性）
* 输入: v2_parts/{A_wc21,B_esri20,C_th17,D_clcd20,E_clcd100,F_cn30}.parquet
* 步骤:
  1) 各产品按 row_id 去重（窗口重叠致重复；同 id 多值时取"众数/首个非255"）
  2) raw → level0 (9类) 查表
  3) 教师口径对齐: 教师 T_L0 来自 class_new（lc_conf LEVEL0）
  4) 投票: 6 个票源 = WC21, ESRI20, TH17, CLCD30_20, CN30_20, (CLCD100_20≈CLCD30 合并参考不重复计)
     一致度 agree_n = 与教师 l0 相同的票数（0..5）
  5) 置信分层:
       金 gold   : agree_n==5 且全部 div<=2（10m 票源非边界）
       银 silver : agree_n>=4
       铜 bronze : agree_n==3
       剔除      : agree_n<=2
  6) 边界带标记: 任一 10m 票源 div>=3 → border=1（金标要求 div<=2 自动排除重边界）
  7) CLCD 2000-2024 稳定列 stab_years (0..24)
  8) 输出 cn_samples_v2.parquet（含全部原始列+新列）与分层统计
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_conf import CLASSES, LEVEL0, CODE_MAP

PARTS = '数据/本地处理/样本底座/v2_parts'
OUT = '数据/本地处理/样本底座/cn_samples_v2.parquet'
BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v1.parquet')
N = len(BASE)
T_L0 = np.array([LEVEL0[CLASSES[int(c)][3]] for c in BASE['class_new']], dtype=np.int8)
idx_of_rowid = pd.Series(np.arange(N), index=BASE.index)   # row_id -> 行号

def load_dedup(pattern, cols_raw):
    """读取并按 row_id 去重；同 id 多行时优先 raw!=255 的第一行"""
    fs = glob.glob(os.path.join(PARTS, pattern))
    if not fs:
        print(f'[warn] 缺少 {pattern}')
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df[df.row_id.isin(idx_of_rowid.index)]
    df['miss'] = (df[cols_raw[0]] == 255)
    # 优先保留非缺失；同组内取第一
    df = df.sort_values(['row_id', 'miss']).drop_duplicates('row_id', keep='first')
    df = df.set_index('row_id').reindex(BASE.index)
    return df

print('1) 读取与去重 …', flush=True)
A = load_dedup('A_wc21.parquet', ['wc_raw'])
B = load_dedup('B_esri20.parquet', ['esri_raw'])
C = load_dedup('C_th17.parquet', ['th_raw'])
D = load_dedup('D_clcd20.parquet', ['clcd_raw'])
F = load_dedup('F_cn30.parquet', ['cn30_raw'])
E = None
if os.path.exists(os.path.join(PARTS, 'E_clcd100.parquet')):
    E = pd.read_parquet(os.path.join(PARTS, 'E_clcd100.parquet'))
    E = E.drop_duplicates('row_id').set_index('row_id').reindex(BASE.index)
    print('E 稳定列:', E['stab_years'].notna().sum(), flush=True)

print('2) 查表 raw→l0 …', flush=True)
from lc14_sample_products import LUT   # 复用码表
def to_l0(df, rawcol, key):
    if df is None: return np.full(N, -1, dtype=np.int8)
    return LUT[key][df[rawcol].fillna(255).astype(np.uint8).to_numpy()]

wc_l0  = to_l0(A, 'wc_raw', 'WC')
es_l0  = to_l0(B, 'esri_raw', 'ESRI')
th_l0  = to_l0(C, 'th_raw', 'TH')
cl_l0  = to_l0(D, 'clcd_raw', 'CLCD')
cn_l0  = to_l0(F, 'cn30_raw', 'CN30')

print('3) 投票 …', flush=True)
votes = np.stack([wc_l0, es_l0, th_l0, cl_l0, cn_l0], axis=1)   # N×5
agree = (votes == T_L0[:, None]).sum(axis=1).astype(np.int8)

print('4) 边界带/多样性 …', flush=True)
def divcol(df, col, key):
    if df is None: return np.full(N, 0, dtype=np.uint8)
    return df[col].fillna(0).to_numpy(dtype=np.uint8)
divs = np.stack([divcol(A, 'wc_div', 'WC'), divcol(B, 'esri_div', 'ESRI'),
                 divcol(C, 'th_div', 'TH')], axis=1)
border = ((divs >= 3).any(axis=1)).astype(np.uint8)

print('5) 置信分层 …', flush=True)
tier = np.full(N, 'reject', dtype=object)
tier[(agree >= 3)] = 'bronze'
tier[(agree >= 4)] = 'silver'
gold = (agree == 5) & (~border.astype(bool))
tier[gold] = 'gold'
# 缺票点(比如 ESRI 覆盖外)降级处理：票源有效数
nvote = (votes >= 0).sum(axis=1)
tier[(nvote <= 3) & (tier == 'reject')] = 'uncovered'   # 产品覆盖不足，非矛盾

stab = E['stab_years'].to_numpy() if E is not None else np.full(N, -1, np.int16)

v2 = BASE.copy()
v2['wc_l0'] = wc_l0; v2['esri_l0'] = es_l0; v2['th_l0'] = th_l0
v2['clcd_l0'] = cl_l0; v2['cn30_l0'] = cn_l0
v2['agree_n'] = agree
v2['border'] = border
v2['stab_years'] = stab
v2['tier'] = tier
v2.to_parquet(OUT, index=False)

print('\n=== 分层统计 ===')
print(v2['tier'].value_counts())
print('\n各层 agree_n 分布:')
print(v2.groupby('tier')['agree_n'].value_counts().sort_index())
print('\n金标样本的类别覆盖（level0）:')
g = v2[v2.tier == 'gold']
print(g.groupby('class_new').size().sort_values(ascending=False).head(32))
print('\n金标+20年稳定(<= -1 或 >=18):', ((g.stab_years >= 18) | (g.stab_years == -1)).sum())
print('输出:', OUT)
