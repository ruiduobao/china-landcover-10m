# -*- coding: utf-8 -*-
"""
s6_clean_train.py — r1 步骤5：训练集清洗 → r1_train.parquet
* 统一清洗定义（消灭旧链 n16/n18 双版本冲突）：
  1) tier 门槛: gold/silver/bronze 全留；external 需 agree_n ≥ 3（旧链外部点无投票门槛）
  2) 生态硬规则: ECO_LAT 区间 + ECO_EXTRA_HARD 联合规则（s0_conf 唯一定义）
  3) 同类最小间距: 每类 MIN_DIST（0.25km 桶贪心，优先级=TIER_W×src_conf，桶=间距防漏判）
  4) 0.25°格×类配额 ≤200（优先级截断，确定性）
  5) 国界终检（应恒为 0 剔除，仅作门禁）
* 全程固定种子，输出含 cell_q/province。
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

def to_km(lon, lat):
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])

def greedy_thin(df, dist_m):
    """同类桶贪心抽稀：bucket=dist_km，保证任意 <dist 点对必在相邻桶内"""
    dist_km = dist_m / 1000.0
    km = to_km(df.lon.to_numpy(), df.lat.to_numpy())
    bx = np.floor(km[:, 0] / dist_km).astype(np.int64)
    by = np.floor(km[:, 1] / dist_km).astype(np.int64)
    kept = {}
    keep = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        kx, ky = bx[i], by[i]
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((kx + dx, ky + dy), ()):
                    if (km[i, 0] - km[j, 0]) ** 2 + (km[i, 1] - km[j, 1]) ** 2 < dist_km * dist_km:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok:
            kept.setdefault((kx, ky), []).append(i)
            keep[i] = True
    return keep

# 专题产品源前缀：稀缺类由专项高精度产品确认，通用五产品本不识别其亚类，
# 不适用 agree≥3 门槛（生态/国界/间距/配额约束照常）
THEMATIC_FREE = r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|gwl_fcs30_2020_raster)'
# 投票盲区类：五产品中 ESRI/清华/CLCD 无湿地与灌丛细分图例、CN30(G30) 亦无，
# 这些类的票天然缺失，uncovered（覆盖不足）不代表标签错误。
# 对盲区类保留 uncovered，仅剔除主动反对（reject）。
BLIND_CLASSES = {120, 121, 180, 181, 182, 183, 184, 185, 186, 140, 91, 92}

def main():
    t0 = time.time()
    df = pd.read_parquet(C.R1_POOL)
    print(f'r1_pool 输入: {len(df):,}', flush=True)
    n_all = len(df)

    # ---- 1) tier 门槛 ----
    src = df['src'].fillna('').astype(str)
    thematic = src.str.contains(THEMATIC_FREE, regex=True)
    blind = df['class_new'].isin(BLIND_CLASSES)
    keep_tier = df['tier'].isin(['gold', 'silver', 'bronze']) | (
        (df['tier'] == 'external') &
        ((df['agree_n'] >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)) | (
        blind & (df['tier'] == 'uncovered'))
    n0 = len(df)
    df = df[keep_tier].reset_index(drop=True)
    print(f"1) tier 门槛: {n0:,} -> {len(df):,}（盲区类保留 uncovered "
          f"{int((blind & (df['tier'] == 'uncovered')).sum()):,}）", flush=True)

    # ---- 2) 生态硬规则 ----
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    n0 = len(df)
    df = df[~bad].reset_index(drop=True)
    print(f'2) 生态硬规则: 剔 {n0-len(df):,} -> {len(df):,}', flush=True)

    # ---- 3) 同类最小间距（桶贪心，优先级高者先占位） ----
    tw = df['tier'].map(C.TIER_W).fillna(0.5).to_numpy()
    conf = df['src_conf'].fillna(0.5).to_numpy()
    rng = np.random.default_rng(C.SEED)
    prio = tw * 10 + conf + rng.random(len(df)) * 1e-6
    order = np.argsort(-prio, kind='stable')
    keep_all = np.zeros(len(df), dtype=bool)
    cls_arr = df['class_new'].to_numpy()
    lon_arr = df['lon'].to_numpy(); lat_arr = df['lat'].to_numpy()
    for c in np.unique(cls_arr):
        idx = np.flatnonzero(cls_arr == c)
        if len(idx) <= 1:
            keep_all[idx] = True
            continue
        sub = df.iloc[idx]
        dist_m = C.MIN_DIST.get(c, C.MIN_DIST['default'])
        # 桶贪心要求按优先级序遍历：把 idx 按 prio 排序后重排
        sub_order = idx[np.argsort(-prio[idx], kind='stable')]
        sub_df = df.iloc[sub_order]
        keep_sub = greedy_thin(sub_df, dist_m)
        keep_all[sub_order[keep_sub]] = True
    n0 = len(df)
    df = df[keep_all].reset_index(drop=True)
    prio = prio[keep_all]
    print(f'3) 同类最小间距: 剔 {n0-len(df):,} -> {len(df):,}', flush=True)

    # ---- 4) 0.25°格×类配额 ----
    lon = df.lon.to_numpy(); lat = df.lat.to_numpy(); cls = df.class_new.to_numpy()
    cell_q = np.floor(lon * 4).astype(np.int64) * 10000 + np.floor(lat * 4).astype(np.int64)
    pk = cell_q.astype(np.int64) * 1000 + cls
    tmp = pd.DataFrame({'pk': pk, 'prio': prio})
    tmp = tmp.sort_values(['pk', 'prio'], ascending=[True, False])
    rank = tmp.groupby('pk').cumcount().to_numpy()
    keep_q = np.zeros(len(df), dtype=bool)
    keep_q[tmp.index[rank < C.CELL_QUOTA]] = True   # rank 已按原索引对齐
    n0 = len(df)
    df = df[keep_q].reset_index(drop=True)
    df['cell_q'] = cell_q[keep_q]
    print(f'4) 0.25°格配额≤{C.CELL_QUOTA}: 剔 {n0-len(df):,} -> {len(df):,}', flush=True)

    # ---- 5) 国界终检 ----
    inside = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    n0 = len(df)
    df = df[inside].reset_index(drop=True)
    print(f'5) 国界终检: 剔 {n0-len(df):,}（应≈0）-> {len(df):,}', flush=True)

    df.to_parquet(C.R1_TRAIN, index=False)
    print(f'\n最终训练集: {len(df):,} 点, {df.class_new.nunique()} 类', flush=True)
    print('tier:', dict(df['tier'].value_counts()))
    print('类别分布:')
    for c in sorted(df.class_new.unique()):
        print(f'  {c:>4} {C.class_name(c):<14} {int((df.class_new == c).sum()):>9,}')

    summary = {'pool_total': int(n_all), 'train_total': int(len(df)),
               'by_tier': {k: int(v) for k, v in df['tier'].value_counts().items()},
               'by_class': {str(k): int(v) for k, v in df['class_new'].value_counts().sort_index().items()},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(C.R1_TRAIN.replace('.parquet', '_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('\n输出:', C.R1_TRAIN, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
