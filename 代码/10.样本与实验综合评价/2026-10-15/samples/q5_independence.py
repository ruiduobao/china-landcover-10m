# -*- coding: utf-8 -*-
"""q5_independence.py — 池 v2 独立性/风险统计：唯一位置、跨类同坐标、重复id、聚团、评估泄漏、tier构成、年份证据

* 输入：train2023_v2.parquet（必要列）、delivery2023_slim.parquet、5 eval 窗 + validation_pool_v2 + 错位清单、
        各交付类 逐年质检统计.csv
* 输出：
  - ind_unique_positions_by_class.csv   逐类：n / 唯一(lon6,lat6) / dup位置数 / 最大重复 / 0.025°格数与top-10占share / 100k子样本最近邻中位
  - ind_cross_class_conflicts.csv       同坐标跨类冲突（位置→类集合），top 类对计数
  - ind_id_risk.csv                     id 空间体检：重复id、数字/非数字、跨类id碰撞、评估清单 id/位置命中
  - ind_tier_in_pool.csv                池内 tier 语义构成（含 2/14/15 缺失）
  - ind_year_evidence.csv               逐类 × 2017–2024 年成员覆盖（来自交付逐年质检统计）
* 用法：python q5_independence.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
SP = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'
DEL = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
EVALW = [r'F:/lc_work/样本修正/geetest/eval_w%d.parquet' % i for i in range(1, 5)] + \
        [r'F:/lc_work/v31_exp/data/eval_w5.parquet']
VP2 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本底座/validation_pool_v2.parquet'
BAD = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/错位样本剔除清单.parquet'
NAME = {1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶林', 5: '落叶阔叶林', 6: '常绿针叶林', 7: '落叶针叶林',
        9: '常绿灌丛', 10: '落叶灌丛', 11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽',
        16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 21: '不透水面', 22: '裸地',
        23: '水体', 24: '永久冰雪'}


def main():
    v2 = pd.read_parquet(r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet',
                         columns=['point_id', 'cl', 'lon', 'lat'])
    n = len(v2)
    print('池 v2 n =', n)

    # ---------- 1. 逐类唯一位置 / 聚团 ----------
    rows = []
    for cl, g in v2.groupby('cl'):
        lo = g.lon.round(6).to_numpy(); la = g.lat.round(6).to_numpy()
        pos = pd.Series(list(zip(lo, la)))
        vc = pos.value_counts()
        # 0.025° 格
        cell = pd.Series(list(zip((g.lon / 0.025).astype(int), (g.lat / 0.025).astype(int))))
        cvc = cell.value_counts()
        top10 = float(cvc.head(10).sum() / len(g))
        # 最近邻（子样本 100k）
        rng = np.random.default_rng(7)
        idx = rng.choice(len(g), size=min(100000, len(g)), replace=False)
        pts = np.c_[lo[idx], la[idx]]
        if len(pts) > 2:
            tr = cKDTree(pts)
            d, _ = tr.query(pts, k=2)
            nn_med = float(np.median(d[:, 1]) * 111.32)  # 粗略 km（纬度向）
        else:
            nn_med = np.nan
        rows.append(dict(cl=int(cl), name=NAME[int(cl)], n=len(g), uniq_pos=int(vc.shape[0]),
                         dup_pos_extra=int(len(g) - vc.shape[0]), max_mult=int(vc.max()),
                         n_cell_0025=int(cvc.shape[0]), top10_cell_share=round(top10, 4),
                         nn_med_km=round(nn_med, 3)))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'ind_unique_positions_by_class.csv'), index=False, encoding='utf-8-sig')

    # ---------- 2. 跨类同坐标冲突 ----------
    v2['pkey'] = list(zip(v2.lon.round(6), v2.lat.round(6)))
    pk = v2.groupby('pkey').cl.agg(lambda s: tuple(sorted(set(s))))
    conflict = pk[pk.map(len) > 1]
    print('跨类同坐标位置数：%d（涉及 %d 点）' % (len(conflict), int(v2.pkey.isin(conflict.index).sum())))
    pairs = {}
    for tup in conflict:
        for i in range(len(tup)):
            for j in range(i + 1, len(tup)):
                pairs[(tup[i], tup[j])] = pairs.get((tup[i], tup[j]), 0) + 1
    pr = pd.DataFrame([dict(cl_a=a, name_a=NAME[a], cl_b=b, name_b=NAME[b], n_pos=n) for (a, b), n in
                       sorted(pairs.items(), key=lambda kv: -kv[1])])
    pr.to_csv(os.path.join(SP, 'ind_cross_class_conflicts.csv'), index=False, encoding='utf-8-sig')
    # 同位置跨类明细样例（前 20）
    conflict.head(20).to_frame('classes').to_csv(os.path.join(SP, 'ind_cross_class_conflict_examples.csv'),
                                                 encoding='utf-8-sig')

    # ---------- 3. id 风险 ----------
    v2['numeric_id'] = v2.point_id.str.fullmatch(r'\d+', na=False)
    idupe = v2.point_id.duplicated().sum()
    idupe_cl = v2.duplicated(['cl', 'point_id']).sum()
    # 跨类幂等（同 id 出现于多类）
    idcl = v2.groupby('point_id').cl.nunique()
    cross_id = int((idcl > 1).sum())
    # id 位数分布（数字 id）
    num = v2.loc[v2.numeric_id, 'point_id']
    idlen = num.str.len().value_counts().sort_index()
    # 评估清单命中（自证 s1 口径）
    rid = set(); loc = set()
    for f in EVALW:
        d = pd.read_parquet(f, columns=['row_id', 'lon', 'lat'])
        rid |= set(d.row_id.astype(str)); loc |= set(zip(d.lon.round(6), d.lat.round(6)))
    vp = pd.read_parquet(VP2, columns=['lon', 'lat']); loc |= set(zip(vp.lon.round(6), vp.lat.round(6)))
    bd = pd.read_parquet(BAD, columns=['row_id', 'lon', 'lat'])
    rid |= set(bd.row_id.astype(str)); loc |= set(zip(bd.lon.round(6), bd.lat.round(6)))
    hit_id = int(v2.point_id.isin(rid).sum())
    hit_loc = int(v2.pkey.isin(loc).sum())
    # 4 位小数格（≈11 m）与评估位置近似重合
    loc4 = set((round(a, 4), round(b, 4)) for a, b in loc)
    hit_loc4 = int(sum(1 for a, b in v2.pkey if (round(a, 4), round(b, 4)) in loc4))
    pd.DataFrame([dict(metric='dup_point_id_total', value=int(idupe)),
                  dict(metric='dup_point_id_within_class', value=int(idupe_cl)),
                  dict(metric='point_id_in_multi_class', value=cross_id),
                  dict(metric='numeric_id_share', value=round(float(v2.numeric_id.mean()), 4)),
                  dict(metric='eval_id_hit', value=hit_id),
                  dict(metric='eval_pos6_hit', value=hit_loc),
                  dict(metric='eval_pos4_hit_11m', value=hit_loc4)]).to_csv(
        os.path.join(SP, 'ind_id_risk.csv'), index=False, encoding='utf-8-sig')
    idlen.to_frame('n').to_csv(os.path.join(SP, 'ind_idlen.csv'), encoding='utf-8-sig')
    # 多类 id 冲突的类对构成
    multi = v2.groupby('point_id').cl.agg(lambda s: tuple(sorted(set(s))))
    multi = multi[multi.map(len) > 1]
    pdict = {}
    for tup in multi:
        for i in range(len(tup)):
            for j in range(i + 1, len(tup)):
                pdict[(tup[i], tup[j])] = pdict.get((tup[i], tup[j]), 0) + 1
    pd.DataFrame([dict(cl_a=a, name_a=NAME[a], cl_b=b, name_b=NAME[b], n_ids=n)
                  for (a, b), n in sorted(pdict.items(), key=lambda kv: -kv[1])]).to_csv(
        os.path.join(SP, 'ind_id_collision_pairs.csv'), index=False, encoding='utf-8-sig')
    # 数字 id 的跨类冲突（row_id 空间）
    num2 = v2[v2.numeric_id]
    multi_n = num2.groupby('point_id').cl.nunique()
    print('数字 id 跨类冲突个数：%d' % int((multi_n > 1).sum()))
    print('id: dup=%d dup_in_class=%d multi_class_ids=%d numeric=%.1f%% eval_id=%d eval_pos=%d pos4=%d' % (
        idupe, idupe_cl, cross_id, 100 * v2.numeric_id.mean(), hit_id, hit_loc, hit_loc4))

    # ---------- 4. tier 池内构成 ----------
    slim = pd.read_parquet(os.path.join(SP, 'delivery2023_slim.parquet'), columns=['cl', 'point_id', 'tier'])
    m = v2.merge(slim, on=['cl', 'point_id'], how='left')
    t = m.groupby(['cl', 'tier']).size().reset_index(name='n')
    t['name'] = t.cl.map(NAME)
    t.to_csv(os.path.join(SP, 'ind_tier_in_pool.csv'), index=False, encoding='utf-8-sig')
    print('tier 池内：', t.groupby('tier').n.sum().to_dict())

    # ---------- 5. 年份证据覆盖 ----------
    yrows = []
    for d in sorted(glob.glob(DEL + '/*/逐年质检统计.csv')):
        cls_dir = os.path.basename(os.path.dirname(d))
        df = pd.read_csv(d, encoding='utf-8-sig')
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            if 'n_in_folder' in df.columns:
                sw = pd.to_numeric(r.get('strong'), errors='coerce'); wk = pd.to_numeric(r.get('weak'), errors='coerce')
                yrows.append(dict(cls_dir=cls_dir, year=int(r['year']), n=int(r['n_in_folder']),
                                  n_pool=int(r['n_pool']),
                                  strong=-1 if pd.isna(sw) else int(sw),
                                  weak=-1 if pd.isna(wk) else int(wk), removed=-1))
            else:
                yrows.append(dict(cls_dir=cls_dir, year=int(r['year']), n=int(r['n']),
                                  n_pool=int(r['n']), strong=-1, weak=-1, removed=int(r['removed'])))
    ye = pd.DataFrame(yrows)
    ye.to_csv(os.path.join(SP, 'ind_year_evidence.csv'), index=False, encoding='utf-8-sig')
    piv = ye.pivot_table(index='cls_dir', columns='year', values='n')
    piv.to_csv(os.path.join(SP, 'ind_year_evidence_pivot.csv'), encoding='utf-8-sig')
    print(piv.fillna(0).astype(int).to_string())
    print('done')


if __name__ == '__main__':
    main()
