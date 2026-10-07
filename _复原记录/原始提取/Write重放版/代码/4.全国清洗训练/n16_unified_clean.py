# -*- coding: utf-8 -*-
"""
n16_unified_clean.py — 统一样本清洗（生态合理性 + 500m 同类抽稀 + 格网配额）
* 输入: cn_samples_v4.parquet（10,223,338 点）
* 三重过滤:
  1. 生态合理性规则: 按地类-气候/地理常识设规则，违规点标记 ecol_flag
  2. 同类500m最小间距: 消除成片矩形（每个类在局部只保留代表性点）
  3. 格网配额: 每 0.25° 格每类 ≤100 点（硬上限防堆积）
* 输出: cn_samples_v5.parquet（训练级，含可靠度得分）+ v5_summary.json
* 资源: 单进程、≤10GB
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely import contains_xy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
BOUND = r'数据/边界/china_100000_full.json'
OUT = '数据/本地处理/样本底座/cn_samples_v5.parquet'
LOG = '数据/本地处理/全国清洗训练/n16_run.log'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---------- 生态合理性规则（硬规则=剔除；软规则=降权0.3） ----------
# 规则基于: 中国植被/土地覆盖地理分布常识
ECO_RULES = {
    # class: [(condition_fn(lon,lat)->bool, 'hard'|'soft', reason)]
    120: [(lambda x, y: y > 42, 'hard', '常绿灌丛在42°N以北无分布'),
          (lambda x, y: (x > 80) & (x < 95) & (y > 32), 'hard', '藏北高原无常绿灌丛'),
          (lambda x, y: (x > 80) & (x < 90) & (y > 36), 'hard', '塔里木腹地无常绿灌丛')],
    121: [(lambda x, y: (x > 80) & (x < 92) & (y > 37), 'hard', '塔里木/藏北核心落叶灌丛不可靠')],
    51:  [(lambda x, y: y < 18, 'hard', '常绿阔叶林不在南海'),
          (lambda x, y: y > 34, 'soft', '常绿阔叶林北界')],
    52:  [(lambda x, y: y < 18, 'hard', '疏闭常绿阔叶林不在南海'),
          (lambda x, y: y > 35, 'soft', '疏闭常绿阔叶林北界')],
    61:  [(lambda x, y: y > 53, 'soft', '落叶阔叶林北界')],
    130: [(lambda x, y: y < 18, 'soft', '草地南界')],
    220: [(lambda x, y: y < 30, 'hard', '冰雪只在高山/高纬')],
}

# 每类最小间距（米），控制空间自相关
MIN_DIST = {
    'default': 500,       # 默认同类点间距
    10: 500, 11: 1000, 12: 500,
    51: 500, 52: 1000, 61: 500, 62: 1000,
    71: 500, 72: 1000, 81: 500, 82: 500,
    91: 500, 92: 1000,
    120: 1000, 121: 1000,
    130: 500, 140: 2000, 150: 500,
    180: 500, 181: 500, 182: 500, 183: 500, 184: 1000, 185: 500, 186: 500,
    190: 200, 200: 300, 201: 500, 202: 500, 220: 2000,
}

# 格网配额: 每 0.25° 格每类最大点数
CELL_QUOTA = 100

def main():
    t0 = time.time()
    df = pd.read_parquet(V4)
    print(f'v4 输入: {len(df):,} 点', flush=True)

    # ========== 1) 生态合理性 ==========
    print('Step 1: 生态合理性规则…', flush=True)
    ecol_hard = np.zeros(len(df), dtype=bool)
    ecol_soft = np.zeros(len(df), dtype=bool)
    lon = df.lon.to_numpy(); lat = df.lat.to_numpy()
    cls = df.class_new.to_numpy(int)
    for c, rules in ECO_RULES.items():
        m = cls == c
        for fn, severity, reason in rules:
            bad = fn(lon[m], lat[m])
            if severity == 'hard':
                ecol_hard[m] |= bad
            else:
                ecol_soft[m] |= bad
    print(f'  生态硬违规: {ecol_hard.sum():,}  软违规: {ecol_soft.sum():,}', flush=True)

    # ========== 2) 同类最小间距抽稀 ==========
    print('Step 2: 同类空间抽稀（最小间距）…', flush=True)
    from scipy.spatial import cKDTree
    # 按类分桶处理
    keep_dist = np.ones(len(df), dtype=bool)
    for c in np.unique(cls):
        m = cls == c
        idx = np.flatnonzero(m)
        if len(idx) < 10:
            continue
        dist_km = MIN_DIST.get(c, MIN_DIST['default']) / 1000.0
        xy = np.column_stack([lon[idx] * 111.32 * np.cos(np.radians(lat[idx])),
                              lat[idx] * 110.57])
        tree = cKDTree(xy)
        pairs = tree.query_pairs(dist_km)
        # 贪心：src_conf 高的优先保留
        conf = df['src_conf'].to_numpy()[idx] if 'src_conf' in df.columns else np.full(len(idx), 0.8)
        order = np.argsort(-conf)
        adj = {}
        for i, j in pairs:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        removed = set()
        for i in order:
            if i in removed:
                continue
            for j in adj.get(i, []):
                if j not in removed:
                    removed.add(j)
        # 写回：removed 点设为 False
        rem_local = np.array(sorted(removed), dtype=int)
        if len(rem_local):
            keep_dist[idx[rem_local]] = False
    print(f'  距离抽稀后剩余: {keep_dist.sum():,}', flush=True)

    # ========== 3) 格网配额 ==========
    print('Step 3: 0.25° 格网配额（每格每类 ≤100）…', flush=True)
    cell_q = (np.floor(lon * 4).astype(int) * 10000 +
              np.floor(lat * 4).astype(int))
    pair_key = cell_q.astype(np.int64) * 100 + cls
    conf_arr = df['src_conf'].fillna(0.5).to_numpy() if 'src_conf' in df.columns else np.full(len(df), 0.5)
    tmp = pd.DataFrame({'pk': pair_key, 'conf': conf_arr, 'keep': keep_dist})
    tmp = tmp.sort_values(['pk', 'conf'], ascending=[True, False])
    tmp['rank'] = tmp.groupby('pk').cumcount()
    quota_ok = (tmp['rank'] < CELL_QUOTA).to_numpy()
    keep_dist = keep_dist & quota_ok
    print(f'  配额后剩余: {keep_dist.sum():,}', flush=True)

    # ========== 汇总 ==========
    keep_final = keep_dist & (~ecol_hard)
    df_out = df[keep_final].copy()
    df_out['ecol_hard'] = ecol_hard[keep_final]
    df_out['ecol_soft'] = ecol_soft[keep_final]
    df_out['keep_reason'] = np.where(
        ecol_hard[keep_final], 'eco_violation',
        np.where(keep_dist[keep_final], 'ok', 'quota_exceeded'))
    df_out.to_parquet(OUT, index=False)

    removed_n = len(df) - keep_final.sum()
    summary = {
        'input': int(len(df)),
        'ecol_hard_removed': int(ecol_hard.sum()),
        'dist_thinned': int((~keep_dist & ~ecol_hard).sum()),
        'final': int(keep_final.sum()),
        'by_tier': dict(df_out.tier.value_counts()),
        'by_class': {str(k): int(v) for k, v in df_out.class_new.value_counts().items()},
        'elapsed_min': round((time.time() - t0) / 60, 1),
    }
    json.dump(summary, open(OUT.replace('.parquet', '_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    print('输出:', OUT)

if __name__ == '__main__':
    main()
