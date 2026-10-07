# -*- coding: utf-8 -*-
"""
n15_fix_v4.py — 修复 v4：GLC_FCS10 采样抽稀 + 国界裁剪 + ESRI 青藏耕地降权
* 问题回顾:
  1) glc_fcs10_samples.parquet 是 n8 多次运行叠加（每 (1°格,类) 达 25-36 遍 × 15 点）→ 抽稀至 ≤15
  2) GLC_FCS10/其他采样 bbox 裁剪不彻底 → 按国界再清一次
  3) ESRI 在青藏高原的耕地为已知产品噪声 → src_conf 降至 0.3（lat>30 & lon<104 的 class 10）
* 输出: cn_samples_v4.parquet（覆盖）+ 修复报告
"""
import os, sys, json
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
FCS = '数据/外部样本源/glc_fcs10_samples.parquet'
BOUND = r'数据/边界/china_100000_full.json'

def load_polys():
    d = json.load(open(BOUND, encoding='utf-8'))
    ps = []
    for f in d['features']:
        sg = shapely.geometry.shape(f['geometry'])
        ps += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return ps

def china_mask(xs, ys, polys):
    from shapely import contains_xy
    keep = np.zeros(len(xs), dtype=bool)
    m = (xs >= 70) & (xs <= 138) & (ys >= 15) & (ys <= 56)
    km = np.zeros(m.sum(), dtype=bool)
    sx, sy = xs[m], ys[m]
    for p in polys:
        km |= contains_xy(p, sx, sy)
    keep[m] = km
    return keep

def thin_fcs10():
    """glc_fcs10 样本: 国界裁剪 + (1°格,类) 抽稀 ≤15"""
    df = pd.read_parquet(FCS)
    n0 = len(df)
    polys = load_polys()
    keep = china_mask(df.lon.to_numpy(), df.lat.to_numpy(), polys)
    df = df[keep].reset_index(drop=True)
    print(f'glc_fcs10: {n0:,} -> 国界内 {len(df):,}')
    df['cell'] = (np.floor(df.lon).astype(int) * 1000 +
                  np.floor(df.lat).astype(int))
    rng = np.random.default_rng(15)
    kept = []
    for (cell, cls), g in df.groupby(['cell', 'class_new']):
        take = min(15, len(g))
        kept.append(g.sample(take, random_state=int(cell) ^ int(cls) ^ 15))
    out = pd.concat(kept, ignore_index=True)
    print(f'  (1°格,类) 抽稀后: {len(out):,}  组合数: {len(kept)}')
    return out

def main():
    v4 = pd.read_parquet(V4)
    print('v4 当前:', f'{len(v4):,}')
    # 1) 剔除旧 glc_fcs10 点
    mask_old = v4.src == 'glc_fcs10_2023'
    v4 = v4[~mask_old].reset_index(drop=True)
    print(f'剔除旧 glc_fcs10 点: {int(mask_old.sum()):,}')
    # 2) 并入抽稀修复后的样本
    fcs = thin_fcs10()
    frames = [v4, fcs[v4.columns.intersection(fcs.columns).tolist()]]
    v4 = pd.concat(frames, ignore_index=True)
    # 3) ESRI 青藏耕地降权（lat>30 & lon<104 的 class 10 = 高原假耕地噪声区）
    m_esri_tibet = (v4.src.str.startswith('esri_nat', na=False)) & \
                   (v4.class_new == 10) & (v4.lat > 30) & (v4.lon < 104)
    v4.loc[m_esri_tibet, 'src_conf'] = 0.3
    print(f'ESRI 青藏耕地降权点数: {int(m_esri_tibet.sum()):,}')
    v4.to_parquet(V4, index=False)
    print('v4 修正后:', f'{len(v4):,}')
    summary = {'before': int(n0 := 0), 'glc_fcs10_kept': int(len(fcs)),
               'esri_tibet_downweighted': int(m_esri_tibet.sum()),
               'v4_total': int(len(v4))}
    del summary['before']
    json.dump(summary, open(V4.replace('.parquet', '_fix_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
