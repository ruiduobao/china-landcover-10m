# -*- coding: utf-8 -*-
"""q2_pool_compare.py — 2023 交付 / 池 v2 / 首批池 v1 三方对照 + 池排除分解 + tier/src 可复原率

* 输入：
  - 交付瘦表 reports/.../delivery2023_slim.parquet（q1 产）
  - 池 v2 F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（只读必要列）
  - 首批池 F:/lc_work/prod5p_2023/data/train2023_clean.parquet（只读必要列）
  - 剔除键来源：5 个 eval 窗 + validation_pool_v2 + 错位样本剔除清单（与 s1_build_pool_v2.py 同文件）
* 输出（reports/quality_review_2026-10-15/samples/）：
  - pool_v2_vs_delivery_by_class.csv   逐类：交付 n / 池 n / 剔除 n / 剔除分解（id键/位置键/两清单命中）
  - pool_v2_tier_src_recover.csv       逐类：池内可回接交付 tier/src/ct_prov 的覆盖率
  - pool_v1_vs_v2_by_class.csv         逐类（24 类映射后）：v1 首批池 n vs v2 池 n / 位置重叠 / v1 tier 构成
  - pool_v1_src_distribution.csv       v1 真来源(src)/质量层(tier)分布（v2 已丢失的那部分）
* 用法：python q2_pool_compare.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SP = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'
V2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
V1 = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
EVALW = [r'F:/lc_work/样本修正/geetest/eval_w%d.parquet' % i for i in range(1, 5)] + \
        [r'F:/lc_work/v31_exp/data/eval_w5.parquet']
VP2 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本底座/validation_pool_v2.parquet'
BAD = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/错位样本剔除清单.parquet'
# 30类 -> 24类（v31_map.json：merges + identity）
M30_24 = {51: 4, 52: 4, 61: 5, 62: 5, 71: 6, 72: 6, 81: 7, 82: 7, 91: 8, 92: 8, 190: 21, 200: 21,
          10: 1, 11: 2, 12: 3, 120: 9, 121: 10, 130: 11, 140: 12, 150: 13, 180: 14, 181: 15,
          182: 16, 183: 17, 184: 18, 185: 19, 186: 20, 201: 22, 202: 23, 220: 24}


def main():
    os.makedirs(SP, exist_ok=True)
    deliv = pd.read_parquet(os.path.join(SP, 'delivery2023_slim.parquet'),
                            columns=['cl', 'point_id', 'lon', 'lat', 'tier', 'src', 'ct_prov'])
    v2 = pd.read_parquet(V2, columns=['point_id', 'cl', 'lon', 'lat', 'src']).rename(columns={'src': 'src_pool'})
    print('交付 %d ｜ 池v2 %d' % (len(deliv), len(v2)))

    # ---- 剔除键（与 s1 同构）----
    rid, loc = set(), set()
    win_hit = {}
    for i, f in enumerate(EVALW, 1):
        d = pd.read_parquet(f, columns=['row_id', 'lon', 'lat'])
        rid |= set(d.row_id.astype(str)); loc |= set(zip(d.lon.round(6), d.lat.round(6)))
        win_hit['eval_w%d' % i] = set(d.row_id.astype(str))
    vp = pd.read_parquet(VP2, columns=['lon', 'lat'])
    loc |= set(zip(vp.lon.round(6), vp.lat.round(6)))
    bd = pd.read_parquet(BAD, columns=['row_id', 'lon', 'lat'])
    rid |= set(bd.row_id.astype(str)); loc |= set(zip(bd.lon.round(6), bd.lat.round(6)))
    print('剔除键 rid %d ｜ loc %d' % (len(rid), len(loc)))

    # ---- 交付 -> 池v2 剔除分解 ----
    v2_key = set(zip(v2.cl, v2.point_id))
    deliv['in_pool'] = [ (c, p) in v2_key for c, p in zip(deliv.cl, deliv.point_id) ]
    drop = deliv[~deliv.in_pool].copy()
    drop['hit_id'] = drop.point_id.isin(rid)
    drop['hit_loc'] = [ (round(a, 6), round(b, 6)) in loc for a, b in zip(drop.lon, drop.lat) ]
    drop['hit_any'] = drop.hit_id | drop.hit_loc
    per = []
    for cl, g in deliv.groupby('cl'):
        n_all = len(g)
        d = drop[drop.cl == cl]
        per.append(dict(cl=cl, n_delivery=n_all, n_pool_v2=int(g.in_pool.sum()), n_dropped=len(d),
                        drop_hit_id=int(d.hit_id.sum()), drop_hit_loc_only=int((d.hit_loc & ~d.hit_id).sum()),
                        drop_unexplained=int((~d.hit_any).sum())))
    by = pd.DataFrame(per)
    # 池v2 中是否有交付之外的 id（应为 0，作对照）
    deliv_key = set(zip(deliv.cl, deliv.point_id))
    extra = sum(1 for c, p in v2_key if (c, p) not in deliv_key)
    print('池v2 中不在交付的 (cl,id) 数：', extra)
    by.to_csv(os.path.join(SP, 'pool_v2_vs_delivery_by_class.csv'), index=False, encoding='utf-8-sig')

    # ---- tier/src/ct_prov 可回接覆盖率 ----
    rec = deliv[['cl', 'point_id', 'tier', 'src', 'ct_prov']].rename(columns={'src': 'src_delivery'})
    m = v2.merge(rec, on=['cl', 'point_id'], how='left', suffixes=('', '_d'))
    rows = []
    for cl, g in m.groupby('cl'):
        rows.append(dict(cl=cl, n=len(g),
                         tier_null=int(g.tier.isna().sum()), tier_strong=int((g.tier == 'strong').sum()),
                         tier_weak=int((g.tier == 'weak').sum()),
                         src_deliv_null=int(g.src_delivery.isna().sum()),
                         ct_prov_null=int(g.ct_prov.isna().sum()),
                         src_pool_unique=int(g.src_pool.nunique())))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'pool_v2_tier_src_recover.csv'), index=False, encoding='utf-8-sig')
    # 交付内 src 分布（真实标签来源层）
    sd = deliv.groupby(['cl', 'src']).size().reset_index(name='n')
    sd.to_csv(os.path.join(SP, 'delivery2023_src_distribution.csv'), index=False, encoding='utf-8-sig')
    print('tier 可读：strong %d / weak %d / null %d' % ((m.tier == 'strong').sum(), (m.tier == 'weak').sum(), m.tier.isna().sum()))

    # ---- 首批池 v1 ----
    v1 = pd.read_parquet(V1, columns=['row_id', 'class_new', 'lon', 'lat', 'tier', 'src', 'qc_status'])
    v1['cl24'] = v1.class_new.map(M30_24)
    v1 = v1[v1.cl24.notna()].copy()
    v1['cl24'] = v1.cl24.astype(int)
    # v2 位置键与 v1 位置键重叠
    v2pos = set(zip(v2.cl, v2.lon.round(6), v2.lat.round(6)))
    v1['in_v2pos'] = [(c, round(a, 6), round(b, 6)) in v2pos for c, a, b in zip(v1.cl24, v1.lon, v1.lat)]
    # v2 id 为数字串时对 v1 row_id
    v2num = v2[v2.point_id.str.fullmatch(r'\d+', na=False)].copy()
    v2num['rid'] = v2num.point_id.astype('int64')
    v2rid = set(zip(v2num.cl, v2num.rid))
    v1['in_v2id'] = [(c, r) in v2rid for c, r in zip(v1.cl24, v1.row_id)]
    rows = []
    for cl, g in v1.groupby('cl24'):
        gv2 = v2[v2.cl == cl]
        rows.append(dict(cl=cl, n_v1=len(g), n_v2=len(gv2),
                         v1_pos_in_v2=int(g.in_v2pos.sum()), v1_id_in_v2=int(g.in_v2id.sum()),
                         v1_tier_gold=int((g.tier == 'gold').sum()), v1_tier_external=int((g.tier == 'external').sum()),
                         v1_reject=int((g.tier == 'reject').sum())))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'pool_v1_vs_v2_by_class.csv'), index=False, encoding='utf-8-sig')
    v1.groupby(['cl24', 'tier']).size().reset_index(name='n').to_csv(
        os.path.join(SP, 'pool_v1_tier_by_class.csv'), index=False, encoding='utf-8-sig')
    v1.groupby(['cl24', 'src']).size().reset_index(name='n').to_csv(
        os.path.join(SP, 'pool_v1_src_distribution.csv'), index=False, encoding='utf-8-sig')
    print('done')


if __name__ == '__main__':
    main()
