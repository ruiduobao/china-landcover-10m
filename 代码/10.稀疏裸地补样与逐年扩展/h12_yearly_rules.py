# -*- coding: utf-8 -*-
"""h12_yearly_rules.py — 补样点逐年可用性判定（2017–2024，锚定 2023）

* 输入：final_2023_v2.parquet（7,098 点）、data/evidence（2023 静态：wc/occurrence）、
        data/evidence_yearly/<year>/（h11 逐年动态证据）
* 判据（与 2023 预注册口径一致，逐年应用）：
    稀疏 keep ⇔ (NDVI≥60 或 shrub≥0.25) 且 NDVI<400 且 water<0.40 且 built<0.30
              且 NDVI≥30 且 WC∉{70,80,100}
    裸地 keep ⇔ NDVI<200 且 water<0.30 且 built<0.30 且 snow<0.30
              且 GSW occurrence<0.30 且 crops<0.35
    NDVI 取值：夏(6–9) → 宽窗(4–11) → 冬(上年12–2) 依次兜底
* 缺证据年份（如 2017 冬季 S2 未开档）：**继承最近有证据年份**的判定，标记 evidence='inherit'
  （沿用 doc63 灌木逐年面板的"锚年+决断级证据、其余继承"原则）
* 输出：
    data/final/yearly_panel.parquet（点×年，status/ndvi/evidence/used_band）
    data/final/yearly_membership.csv（年×类 的 keep 计数）
    data/final/yearly_valid_window.parquet（pid, valid_from, valid_to，含 2023 的连续区间）
* 用法：python h12_yearly_rules.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
YEARS = list(range(2017, 2025))
SENT = -9999.0


def valid(x):
    return np.isfinite(x) & (x > -9000)


def main():
    fin = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'))
    fin = fin[['pid', 'lon', 'lat', 'fam', 'class']].copy()
    # 2023 静态证据
    ev23 = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(os.path.join(BASE, 'data', 'evidence', 'ev_p*_b*.parquet')))],
                     ignore_index=True).drop_duplicates('pid')
    fin = fin.merge(ev23[['pid', 'wc', 'occurrence']], on='pid', how='left')
    cvp = os.path.join(BASE, 'data', 'yearly', 'clcd_values.parquet')
    clcd = {}
    if os.path.isfile(cvp):
        c = pd.read_parquet(cvp)
        for y, g in c.groupby('year'):
            clcd[int(y)] = g[['pid', 'clcd']]
    print('点 %d ｜ 2023 静态证据 wc 非空 %d' % (len(fin), int(fin.wc.notna().sum())), flush=True)

    rows = []
    per_year_counts = {}
    for y in YEARS:
        fs = sorted(glob.glob(os.path.join(BASE, 'data', 'evidence_yearly', str(y), '*.parquet')))
        if not fs:
            print('  !! 缺 %d 证据' % y, flush=True)
            continue
        ev = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates('pid')
        d = fin.merge(ev.drop(columns=[c for c in ('lon', 'lat') if c in ev.columns]),
                      on='pid', how='left')
        for c in ['water', 'trees', 'grass', 'crops', 'built', 'bare', 'snow_and_ice',
                  'flooded_vegetation', 'shrub']:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(-1).clip(0, 1)
        for c in ['s2_ndvi_xia', 's2_ndwi_xia', 's2_ndvi_dong', 's2_ndvi_wide']:
            d[c] = pd.to_numeric(d[c], errors='coerce').where(lambda x: x > -9000)
        d['ndvi'] = d.s2_ndvi_xia.fillna(d.s2_ndvi_wide).fillna(d.s2_ndvi_dong)
        d['used_band'] = np.where(d.s2_ndvi_xia.notna(), 'summer',
                                  np.where(d.s2_ndvi_wide.notna(), 'wide',
                                           np.where(d.s2_ndvi_dong.notna(), 'winter', 'none')))
        sp = d.fam == 'sparse'
        keep_sp = ((d['ndvi'] >= 60) | (d['shrub'] >= 0.25)) & (d['ndvi'] < 400) & \
                  (d['water'] < 0.40) & (d['built'] < 0.30) & (d['ndvi'] >= 30) & \
                  (~d['wc'].isin([70, 80, 100])) & (d['crops'] < 0.35)   # 2026-10-25 追加：农田护栏
        ba = d.fam == 'bare'
        keep_ba = (d['ndvi'] < 200) & (d['water'] < 0.30) & (d['built'] < 0.30) & \
                  (d['snow_and_ice'] < 0.30) & (d['occurrence'] < 0.30) & (d['crops'] < 0.35) & (d['bare'] >= 0.25)   # 2026-10-25：DW 裸地正向门槛
        d['status'] = np.where(sp & keep_sp, 'keep', np.where(ba & keep_ba, 'keep', 'drop'))
        d['evidence'] = np.where(d['ndvi'].notna(), 'direct', 'none')
        # settlement guard（2026-10-25 追加）：CLCD 当年=不透水 且 DW built≥0.25 → 剔（裸地/稀疏与村居工矿混淆）
        cv = clcd.get(y)
        if cv is not None:
            d = d.merge(cv, on='pid', how='left')
            guard = (d['clcd'] == 8)      # CLCD 当年判不透水：村居/工矿裸面，剔
            d.loc[guard, 'status'] = 'drop'
            d = d.drop(columns=['clcd'])
        d['year'] = y
        rows.append(d[['pid', 'fam', 'year', 'status', 'evidence', 'used_band', 'ndvi',
                       'water', 'built', 'crops', 'snow_and_ice', 'shrub', 'bare']])
        per_year_counts[y] = {'n': int(len(d)),
                              'keep': int((d.status == 'keep').sum()),
                              'no_ev': int((d.evidence == 'none').sum()),
                              'by_fam_keep': {k: int(v) for k, v in d[d.status == 'keep'].fam.value_counts().items()},
                              'band': {k: int(v) for k, v in d.used_band.value_counts().items()}}
        print('%d: keep %d/%d ｜ 直接证据 %d ｜ band %s' % (
            y, per_year_counts[y]['keep'], len(d), int((d.evidence == 'direct').sum()),
            per_year_counts[y]['band']), flush=True)

    panel = pd.concat(rows, ignore_index=True)
    # 缺证据年份继承最近有证据年份（按年份距离，平手取较晚）
    for pid, g in panel.groupby('pid'):
        g = g.sort_values('year')
        direct = g[g.evidence == 'direct']
        if direct.empty:
            continue
        d_years = direct.year.to_numpy(); d_stat = direct.status.to_numpy()
        for i, r in g.iterrows():
            if r.evidence == 'direct':
                continue
            k = int(np.argmin(np.abs(d_years - r.year)))
            panel.loc[i, 'status'] = d_stat[k]
            panel.loc[i, 'evidence'] = 'inherit(%d)' % d_years[k]
    panel.to_parquet(os.path.join(BASE, 'data', 'final', 'yearly_panel.parquet'), index=False)

    # 逐年 membership（继承后）
    mem = []
    for y in YEARS:
        g = panel[panel.year == y]
        mem.append({'year': y, 'n': int(len(g)), 'keep': int((g.status == 'keep').sum()),
                    'sparse_keep': int(((g.fam == 'sparse') & (g.status == 'keep')).sum()),
                    'bare_keep': int(((g.fam == 'bare') & (g.status == 'keep')).sum()),
                    'inherit': int(g.evidence.astype(str).str.startswith('inherit').sum())})
    mem = pd.DataFrame(mem)
    mem.to_csv(os.path.join(BASE, 'data', 'final', 'yearly_membership.csv'), index=False,
               encoding='utf-8-sig')
    print(mem.to_string(index=False), flush=True)

    # valid window（含 2023 的连续 keep 区间）
    keep_map = panel.pivot_table(index='pid', columns='year', values='status',
                                 aggfunc='first')
    win = []
    for pid, row in keep_map.iterrows():
        ks = {y: (row.get(y) == 'keep') for y in YEARS}
        if not ks.get(2023, False):
            win.append((pid, 0, 0)); continue
        lo = hi = 2023
        while lo - 1 in ks and ks[lo - 1]:
            lo -= 1
        while hi + 1 in ks and ks[hi + 1]:
            hi += 1
        win.append((pid, lo, hi))
    w = pd.DataFrame(win, columns=['pid', 'valid_from', 'valid_to'])
    w.to_parquet(os.path.join(BASE, 'data', 'final', 'yearly_valid_window.parquet'), index=False)
    print('valid window 分布：', w.groupby(['valid_from', 'valid_to']).size().sort_values(ascending=False).head(8).to_dict(), flush=True)
    json.dump({str(k): v for k, v in per_year_counts.items()},
              open(os.path.join(BASE, 'data', 'final', 'yearly_counts.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
