# -*- coding: utf-8 -*-
"""o14_gem_final.py — GEM 版乔灌园地样本交付装配（候选+判读真值+分层样本→ 目标目录）

* 输入：data/orchard/gem_cn.parquet + gem/feats_y{2020,2023}.parquet + gem_cross.csv
        data/orchard/gem_judge_merged.csv（208 判读）+ gem_overlay.csv（我方成品叠加）
* 规则源（**预注册分层**）：
  A_目视确认：GEM=2 且 208 判读中 Q1=是（22 点）
  B_多源一致：非 A，且 (判别器 p_orchard_2023≥0.30) 或（上轮判读池=是）或（P1_hi 命中）
  C_单源候选：其余 GEM=2 点
* 输出：F:/地理所/.../数据/单类样本的单独处理_2026.10.12/乔灌园地/GEM版/
        - 乔灌园地_GEM候选_全国.parquet（610 点，含 AEF64×2 年）
        - 乔灌园地_GEM候选_全国.csv（精简）
        - 乔灌园地_GEM判读真值.csv（208）
        - 乔灌园地_GEM样本_分层.csv（A/B/C）
        - GEM训练集中国子集.csv / GEM验证集中国子集.csv（Zenodo 公开数据参照）
* 用法：python o14_gem_final.py
"""
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/GEM版'
ZEN = r'F:/lc_work/外部样本/GEM_Forest'


def main():
    os.makedirs(DEST, exist_ok=True)
    cn = pd.read_parquet(os.path.join(ORD, 'gem_cn.parquet'))
    f20 = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2020.parquet'))
    f23 = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y2023.parquet'))
    # 合并 AEF（先按 lon/lat round6 连接）
    for d in (cn, f20, f23):
        d['k'] = d.lon.round(6).astype(str) + '_' + d.lat.round(6).astype(str)
    f20b = f20[['k'] + VC.FEATS].rename(columns={c: c + '_y2020' for c in VC.FEATS})
    f23b = f23[['k'] + VC.FEATS].rename(columns={c: c + '_y2023' for c in VC.FEATS})
    full = cn.merge(f20b, on='k', how='left').merge(f23b, on='k', how='left')
    # 判别器分
    cross = pd.read_csv(os.path.join(ORD, 'gem_cross.csv'), encoding='utf-8-sig')
    cross['k'] = cross.lon.round(6).astype(str) + '_' + cross.lat.round(6).astype(str)
    full = full.merge(cross[['k', 'p_orchard_2020', 'p_orchard_2023']], on='k', how='left')
    # 我方成品叠加
    ov = pd.read_csv(os.path.join(ORD, 'gem_overlay.csv'), encoding='utf-8-sig')
    ov['k'] = ov.lon.round(6).astype(str) + '_' + ov.lat.round(6).astype(str)
    full = full.merge(ov[['k', 'prov_gem', 'our_class', 'our_class_name']], on='k', how='left', suffixes=('', '_ov'))
    # 判读真值（208）
    jm = pd.read_csv(os.path.join(ORD, 'gem_judge_merged.csv'), encoding='utf-8-sig')
    jm['k'] = jm.lon.round(6).astype(str) + '_' + jm.lat.round(6).astype(str)
    full = full.merge(jm[['k', 'point_id', 'Q1', 'Q2', 'Q3', 'conf']], on='k', how='left')
    # P1_hi 命中
    sel = pd.read_csv(os.path.join(ORD, 'orchard_samples_2023.csv'), encoding='utf-8-sig')
    p1 = sel[sel.tier == 'P1_hi'].copy()
    p1['k'] = p1.lon.round(6).astype(str) + '_' + p1.lat.round(6).astype(str)
    full['in_p1'] = full.k.isin(set(p1.k))
    # 分层
    def tier_of(r):
        if r.Q1 == '是':
            return 'A_目视确认'
        if (pd.notna(r.p_orchard_2023) and r.p_orchard_2023 >= 0.30) or r.in_p1:
            return 'B_多源一致'
        return 'C_单源候选'
    full['tier'] = full.apply(tier_of, axis=1)
    # 输出
    keep_core = ['lon', 'lat', 'prov', 'tier', 'p_orchard_2020', 'p_orchard_2023',
                 'our_class_name', 'point_id', 'Q1', 'Q2', 'Q3', 'conf', 'aef_ok_x']
    keep_core = [c for c in keep_core if c in full.columns]
    slim = full[keep_core].rename(columns={'point_id': 'judge_id', 'aef_ok_x': 'aef_ok'})
    full.drop(columns=['k']).to_parquet(os.path.join(DEST, '乔灌园地_GEM候选_全国.parquet'), index=False)
    slim.to_csv(os.path.join(DEST, '乔灌园地_GEM候选_全国.csv'), index=False, encoding='utf-8-sig')
    jm.to_csv(os.path.join(DEST, '乔灌园地_GEM判读真值.csv'), index=False, encoding='utf-8-sig')
    full[keep_core].drop(columns=['aef_ok_x'], errors='ignore').to_csv(
        os.path.join(DEST, '乔灌园地_GEM样本_分层.csv'), index=False, encoding='utf-8-sig')
    print('分层:', full.tier.value_counts().to_dict())
    print('省级 top:', full.prov.value_counts().head(10).to_dict())
    # Zenodo 参照数据
    for src, dst in [('training_china.csv', 'GEM训练集中国子集.csv'),
                     ('plant_validation_china.csv', 'GEM验证集中国子集.csv')]:
        p = os.path.join(ZEN, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(DEST, dst))
    VC.jsave({'n_cn': int(len(full)), 'tiers': full.tier.value_counts().to_dict(),
              'prov_top': full.prov.value_counts().head(15).to_dict()},
             os.path.join(DEST, 'GEM版_stats.json'))
    print('→', DEST)


if __name__ == '__main__':
    main()
