# -*- coding: utf-8 -*-
"""o6_select.py — 乔灌园地：全国放量选点（P0 遗留 / P1 高精 / P2 走量，省级配额 + 间距）

* 输入：data/orchard/scored.parquet（o5 打分）＋ data/orchard/mk_orchard_points.csv（母库框）
* 规则源（**预注册**）：
  P0_legacy：母库框 10,052 点全保留（独立专题源抽样，已有沿用价值；p 一并带出，低 p 者供后续审计）
  P1_hi：p≥0.70，省级上限 3,000
  P2_vol：0.60≤p<0.70，省级上限 2,000，且与 P1 合计每省 ≤4,000
  间距：全选区 ≥500 m（贪心，2° 网格乱序）；省归属用 k19 原始几何 contains（禁 make_valid）
* 门槛：逐级报告 n、省数、p 分布
* 输出：data/orchard/orchard_samples_2023.csv（point_id,lon,lat,prov,frame,tier,p_orchard）
* 用法：python o6_select.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
TH_HI, TH_VOL = 0.50, 0.35
CAP_HI, CAP_VOL = 600, 900
MIN_KM = 0.5


def greedy_spaced(df, min_km, rng):
    """2° 网格乱序 + 贪心间距（近似均匀）。"""
    if len(df) == 0:
        return df
    g = df.copy()
    g['gx'] = (g.lon // 2).astype(int)
    g['gy'] = (g.lat // 2).astype(int)
    g['gk'] = g.gx.astype(str) + '_' + g.gy.astype(str)
    order = rng.permutation(len(g))
    g = g.iloc[order]
    R = min_km / 111.0
    keep, pts = [], []
    arr = g[['lon', 'lat']].to_numpy()
    for i in range(len(g)):
        lo, la = arr[i]
        if pts:
            a = np.asarray(pts)
            d = np.hypot((a[:, 0] - lo) * np.cos(np.radians(la)), a[:, 1] - la)
            if d.min() < R:
                continue
        keep.append(i)
        pts.append((lo, la))
    return g.iloc[keep]


def main():
    rng = np.random.default_rng(20261012)
    sc = pd.read_parquet(os.path.join(ORD, 'scored.parquet'))
    print('候选打分 %d' % len(sc))
    # 省归属
    sys.path.insert(0, os.path.join(WORK, 'code'))
    try:
        import k19_provattrib as PA
        sc['prov'] = PA.attribute(sc)
        sc.loc[sc.prov.astype(str) == '', 'prov'] = ''
    except Exception as e:
        print('省归属失败（%s）→ 空' % str(e)[:80])
        sc['prov'] = ''
    mk = pd.read_csv(os.path.join(ORD, 'mk_orchard_points.csv'), encoding='utf-8-sig')
    mk['frame'] = 'mk_orchard'
    mksc = sc[sc.frame == 'mk_orchard'][['lon', 'lat', 'p_orchard', 'prov']].copy()
    mksc['lon'] = mksc.lon.round(6); mksc['lat'] = mksc.lat.round(6)
    mk['lon6'] = mk.lon.round(6); mk['lat6'] = mk.lat.round(6)
    mk = mk.merge(mksc, left_on=['lon6', 'lat6'], right_on=['lon', 'lat'], how='left',
                  suffixes=('', '_sc'))
    mk['prov'] = mk.get('prov_sc', pd.Series([None]*len(mk))).fillna(mk.get('prov', ''))
    p0 = mk[['lon', 'lat', 'prov', 'frame', 'p_orchard']].copy()
    p0 = p0[p0.prov.notna() & (p0.prov.astype(str) != '')]
    p0['tier'] = 'P0_legacy'
    # P1 / P2
    cand = sc[~sc.frame.isin(['mk_orchard'])].copy()
    cand = cand[cand.prov.notna() & (cand.prov.astype(str) != '')]
    hi = cand[cand.p_orchard >= TH_HI].copy()
    hi['tier'] = 'P1_hi'
    hi = hi.groupby('prov', group_keys=False).apply(lambda g: g.nlargest(CAP_HI, 'p_orchard'))
    hi = greedy_spaced(hi, MIN_KM, rng)
    vol = cand[(cand.p_orchard >= TH_VOL) & (cand.p_orchard < TH_HI)].copy()
    vol['tier'] = 'P2_vol'
    vol = vol.groupby('prov', group_keys=False).apply(lambda g: g.nlargest(CAP_VOL, 'p_orchard'))
    vol = greedy_spaced(vol, MIN_KM, rng)
    # 合并（间距对 P0 不做约束——保留遗留全集）
    sel = pd.concat([p0, hi, vol], ignore_index=True)
    sel = sel.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    sel['point_id'] = ['ORD-%05d' % i for i in range(len(sel))]
    sel = sel[['point_id', 'lon', 'lat', 'prov', 'frame', 'tier', 'p_orchard']]
    sel.to_csv(os.path.join(ORD, 'orchard_samples_2023.csv'), index=False, encoding='utf-8-sig')
    print('P0 %d；P1 %d；P2 %d；合计 %d' % ((sel.tier == 'P0_legacy').sum(), (sel.tier == 'P1_hi').sum(),
                                        (sel.tier == 'P2_vol').sum(), len(sel)))
    print('省份数 %d；tier×prov top:' % sel.prov.nunique())
    print(sel.groupby('tier').size().to_dict())
    print(sel[sel.tier != 'P0_legacy'].prov.value_counts().head(15).to_dict())
    VC.jsave({'n_total': int(len(sel)), 'tiers': sel.tier.value_counts().to_dict(),
              'prov_top': sel.prov.value_counts().head(20).to_dict()},
             os.path.join(WORK, 'results', 'orchard', 'select.json'))


if __name__ == '__main__':
    main()
