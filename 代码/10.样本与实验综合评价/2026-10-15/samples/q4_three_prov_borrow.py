# -*- coding: utf-8 -*-
"""q4_three_prov_borrow.py — 川/粤/宁（+闽1瓦）二次批：省域类别构成 + 每瓦 ±2°窗候选/当地vs借样/20k期望

* 借样口径（代码实证，c3_driver_prod2.py submit_tile）：
    boxp = 瓦片 box 各向外扩 2°；fc = SRC.filterBounds(boxp).randomColumn('rc',7).sort('rc').limit(20000)
  → 每瓦训练样本 = 该 ±2° 窗内全池随机 2 万点。
  ⚠ 本脚本只算"候选窗计数"与"均匀抽样期望"（n_class × min(1, 20000/N_win)），
     **不是 GEE 实际选中点数**（服务端 randomColumn 无法本地复现；无 selected ids 落盘）。
* 当地/借样：当地 = 点 ct_prov == 瓦片所属省；借样 = 在窗内但 ct_prov != 所属省。
* 输入：reports/.../delivery2023_slim.parquet、train2023_v2.parquet、plan/tiles_secondrun.csv
* 输出：
  - tw_prov_class_counts.csv        省域（ct_prov）逐类池点数/占比/交付 src 构成
  - tw_tile_window_candidates.csv   逐瓦 × 类：窗内候选 n / 当地 / 借样 / 均匀期望选中
  - tw_prov_window_summary.csv      省级（union 窗）逐类：候选 unique / 当地 / 借样 / ≥30 的瓦数
* 用法：python q4_three_prov_borrow.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SP = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'
TILES = r'F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv'
CAP = 20000


def main():
    t = pd.read_csv(TILES)
    slim = pd.read_parquet(os.path.join(SP, 'delivery2023_slim.parquet'),
                           columns=['cl', 'point_id', 'lon', 'lat', 'ct_prov', 'tier', 'src'])
    v2 = pd.read_parquet(r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet', columns=['cl', 'point_id'])
    p = v2.merge(slim, on=['cl', 'point_id'], how='left')
    assert p.ct_prov.notna().all(), '池点 ct_prov 缺失 %d' % p.ct_prov.isna().sum()
    # 省域逐类构成（对三省+闽）
    provs = ['四川省', '广东省', '宁夏回族自治区', '福建省']
    SHORT = {'四川省': '川', '广东省': '粤', '宁夏回族自治区': '宁', '福建省': '闽'}
    rows = []
    for pv in provs:
        g = p[p.ct_prov == pv]
        tot = len(g)
        for cl, gg in g.groupby('cl'):
            rows.append(dict(ct_prov=pv, cl=int(cl), n=len(gg), share=round(len(gg) / tot, 5) if tot else 0,
                             tier_strong=int((gg.tier == 'strong').sum()),
                             tier_other=int(gg.tier.notna().sum()) - int((gg.tier == 'strong').sum()),
                             tier_null=int(gg.tier.isna().sum())))
        rows.append(dict(ct_prov=SHORT.get(pv, pv), cl=0, n=tot, share=1.0, tier_strong=-1, tier_other=-1, tier_null=-1))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'tw_prov_class_counts.csv'), index=False, encoding='utf-8-sig')

    # 逐瓦窗内候选
    lon = p.lon.to_numpy(); lat = p.lat.to_numpy(); clv = p.cl.to_numpy(); provv = p.ct_prov.to_numpy()
    rows = []
    for _, r in t.iterrows():
        x0, y0, x1, y1 = r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2
        m = (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
        nwin = int(m.sum())
        if nwin == 0:
            continue
        subc = clv[m]; subp = provv[m]
        local = (subp == {'川': '四川省', '粤': '广东省', '宁': '宁夏回族自治区', '闽': '福建省'}.get(r.prov, r.prov))
        s = pd.DataFrame(dict(cl=subc, local=local)).groupby(['cl', 'local']).size().unstack(fill_value=0)
        for cl, row in s.iterrows():
            nl = int(row.get(True, 0)); nb = int(row.get(False, 0)); n = nl + nb
            rows.append(dict(tile=r.tile, prov=r.prov, n_win=nwin, cl=int(cl), n=n, n_local=nl, n_borrow=nb,
                             borrow_share=round(nb / n, 4) if n else 0,
                             exp_kept_20k=round(n * min(1.0, CAP / nwin), 1)))
    tw = pd.DataFrame(rows)
    tw.to_csv(os.path.join(SP, 'tw_tile_window_candidates.csv'), index=False, encoding='utf-8-sig')

    # 省级 union 汇总
    rows = []
    for pv in provs:
        short = SHORT[pv]
        tt = t[t.prov == short]
        x0 = (tt.x0 - 2).min(); x1 = (tt.x1 + 2).max(); y0 = (tt.y0 - 2).min(); y1 = (tt.y1 + 2).max()
        m = (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
        agg = {}
        sub = pd.DataFrame(dict(cl=clv[m], pt_prov=provv[m]))
        for cl, g in sub.groupby('cl'):
            nl = int((g.pt_prov == pv).sum()); nb = len(g) - nl
            agg[int(cl)] = (len(g), nl, nb)
        # 逐瓦期望汇总
        twp = tw[tw.prov == short]
        for cl, g in twp.groupby('cl'):
            n_ge30 = int((g.exp_kept_20k >= 30).sum()); n_lt30 = int((g.exp_kept_20k < 30).sum())
            n_zero_local = int((g.n_local == 0).sum())
            T = agg.get(int(cl), (0, 0, 0))
            rows.append(dict(prov=short, cl=int(cl), union_n=T[0], union_local=T[1], union_borrow=T[2],
                             tiles=int(len(g)), tiles_exp_lt30=n_lt30, tiles_local0=n_zero_local,
                             exp_kept_med=float(g.exp_kept_20k.median()),
                             exp_kept_min=float(g.exp_kept_20k.min())))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'tw_prov_window_summary.csv'), index=False, encoding='utf-8-sig')

    # 控制台：三省逐类省域 n（前 25 类）
    for pv in ['四川省', '广东省', '宁夏回族自治区']:
        g = p[p.ct_prov == pv]
        top = g.groupby('cl').size().sort_values(ascending=False)
        print(pv, 'n=', len(g), dict(top.head(25)))
    print('done')


if __name__ == '__main__':
    main()
