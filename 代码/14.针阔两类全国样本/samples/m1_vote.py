# -*- coding: utf-8 -*-
"""m1_vote.py — 方法A：多产品针阔投票 / 一致性筛选
* 输入：labels/methods/work_base.parquet（466,135 林点；含池标签/MODIS/FCS10 三叶型源）
        labels/methods/gedi_cgls.parquet（m3 产出的 CGLS forest_type / Hansen / GEDI，若存在）
* 规则：叶型二值 N=1(针) / B=0(阔)。每源独立给出 N/B/非决断。
        strict = 至少 2 个**外部源**（MODIS / FCS10 / CGLS）同向且无外部源反向；
        clean  = 外部源中同向数 > 反向数（多数票，至少 1 外部源）；
        fcs10only = 仅 FCS10 单源。
* 产物：labels/methods/strict_vote.parquet、labels/methods/m1_vote.json
* 用法：python m1_vote.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
BASE = os.path.join(WORK, 'work_base.parquet')
GEDI = os.path.join(WORK, 'gedi_cgls.parquet')


def mh_or(df, a_col, b_col, strat='stratum', min_cell=1):
    """Mantel-Haenszel OR：a=分子类, b=分母类（用 sun=='shade'/'sun' 计数）。
    返回 raw OR、MH OR、层数。"""
    d = df[df['sun'].isin(['shade', 'sun'])].copy()
    if len(d) == 0:
        return None
    a = d[a_col].sum(); b = d[b_col].sum()
    rawh = None
    if a > 0 and b > 0:
        rawh = float((a / max(d[a_col].sum() + d[b_col].sum(), 1)))
    num = den = 0.0; ns = 0
    for _, g in d.groupby(strat, observed=True):
        na_s = int(((g[a_col]) & (g['sun'] == 'shade')).sum())
        na_u = int(((g[a_col]) & (g['sun'] == 'sun')).sum())
        nb_s = int(((g[b_col]) & (g['sun'] == 'shade')).sum())
        nb_u = int(((g[b_col]) & (g['sun'] == 'sun')).sum())
        n = na_s + na_u + nb_s + nb_u
        if min(na_s, na_u, nb_s, nb_u) < min_cell or n == 0:
            continue
        num += na_s * nb_u / n
        den += na_u * nb_s / n
        ns += 1
    return dict(enf_shade=na_s if False else None,
                mh_or=round(num / den, 4) if den > 0 else None, n_strata=ns)


def or_simple(df, a_col, b_col):
    """raw OR（不分层）+ 分层 MH-OR + 层数"""
    d = df[df['sun'].isin(['shade', 'sun'])]
    out = dict(n=len(d),
               n_a=int(d[a_col].sum()), n_b=int(d[b_col].sum()))
    a_s = int((d[a_col] & (d['sun'] == 'shade')).sum()); a_u = int((d[a_col] & (d['sun'] == 'sun')).sum())
    b_s = int((d[b_col] & (d['sun'] == 'shade')).sum()); b_u = int((d[b_col] & (d['sun'] == 'sun')).sum())
    out.update(a_shade=a_s, a_sun=a_u, b_shade=b_s, b_sun=b_u)
    raw = None
    if min(a_s, a_u, b_s, b_u) > 0:
        raw = (a_s / a_u) / (b_s / b_u)
    out['or_raw'] = round(raw, 4) if raw else None
    num = den = 0.0; ns = 0
    for _, g in d.groupby('stratum', observed=True):
        xa_s = int((g[a_col] & (g['sun'] == 'shade')).sum()); xa_u = int((g[a_col] & (g['sun'] == 'sun')).sum())
        xb_s = int((g[b_col] & (g['sun'] == 'shade')).sum()); xb_u = int((g[b_col] & (g['sun'] == 'sun')).sum())
        n = xa_s + xa_u + xb_s + xb_u
        if min(xa_s, xa_u, xb_s, xb_u) <= 0 or n == 0:
            continue
        num += xa_s * xb_u / n; den += xa_u * xb_s / n; ns += 1
    out['mh_or'] = round(num / den, 4) if den > 0 else None
    out['n_strata'] = ns
    return out


def main():
    df = pd.read_parquet(BASE)
    res = {}

    # 可选：并入 CGLS / Hansen / GEDI（若 m3 已产出）
    if os.path.exists(GEDI):
        g = pd.read_parquet(GEDI)
        df = df.merge(g, on='stable_uid', how='left')
        print('merged gedi_cgls', g.shape)

    # ---- 1. 一致性矩阵（叶型轴）
    srcs = [('label', 'leaf_label_s'), ('modis', 'leaf_modis_s'), ('fcs10', 'leaf_fcs10_s')]
    if 'leaf_cgls_s' in df.columns:
        srcs.append(('cgls', 'leaf_cgls_s'))
    if 'leaf_hansen_s' in df.columns:
        srcs.append(('hansen', 'leaf_hansen_s'))

    pairwise = {}
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            n1, c1 = srcs[i]; n2, c2 = srcs[j]
            m = df[[c1, c2]].dropna()
            if len(m) == 0:
                continue
            agree = float((m[c1] == m[c2]).mean())
            pairwise[f'{n1}__{n2}'] = dict(n_pairs=int(len(m)),
                                           agree_rate=round(agree, 4),
                                           both_needle=int(((m[c1] == 1) & (m[c2] == 1)).sum()),
                                           both_broad=int(((m[c1] == 0) & (m[c2] == 0)).sum()))
    res['pairwise_leaf_agreement'] = pairwise

    # ---- 2. 外部源投票（不含池标签；池标签非独立）
    ext = [(n, c) for n, c in srcs if n != 'label']
    E = df[[c for _, c in ext]].copy()
    E.columns = [n for n, _ in ext]
    n_ext_dec = E.notna().sum(axis=1)
    n_ext_N = (E == 1).sum(axis=1)
    n_ext_B = (E == 0).sum(axis=1)
    df['ext_dec'] = n_ext_dec
    df['ext_N'] = n_ext_N
    df['ext_B'] = n_ext_B

    # strict: ≥2 外部源同向，且无外部源反向
    df['vote_strict'] = np.where((n_ext_dec >= 2) & (n_ext_N >= 2) & (n_ext_B == 0), 1,
                                 np.where((n_ext_dec >= 2) & (n_ext_B >= 2) & (n_ext_N == 0), 0, np.nan))
    # clean: 多数票
    df['vote_clean'] = np.where(n_ext_dec >= 1,
                                np.where(n_ext_N > n_ext_B, 1,
                                         np.where(n_ext_B > n_ext_N, 0, np.nan)), np.nan)
    # fcs10 单源
    df['fcs10_only'] = df['leaf_fcs10_s']

    for nm in ['vote_strict', 'vote_clean', 'fcs10_only']:
        v = df[nm]
        cnt = v.value_counts(dropna=True)
        res[nm] = dict(n_decided=int(v.notna().sum()),
                       needle=int((v == 1).sum()), broad=int((v == 0).sum()),
                       pct_of_all=round(v.notna().mean() * 100, 2))

    # ---- 3. 与池标签（cl_24）的一致率
    def agree_with_label(v):
        m = df[[v, 'cl_24']].copy()
        m = m[m[v].notna()]
        lab = np.where(m['cl_24'].isin([6, 7]), 1, 0)
        return dict(n=int(len(m)), agree=round(float((m[v] == lab).mean()), 4))

    for nm in ['vote_strict', 'vote_clean', 'fcs10_only']:
        res[f'{nm}_vs_pool_label'] = agree_with_label(nm)

    # ---- 4. 分层：按气候带 z / 大区
    by_cz = {}
    for nm in ['vote_strict', 'vote_clean', 'fcs10_only']:
        sub = df[df[nm].notna()]
        by_cz[nm] = {str(k): dict(n=int(len(g)),
                                    needle=int((g[nm] == 1).sum()),
                                    needle_pct=round(float((g[nm] == 1).mean()) * 100, 2))
                     for k, g in sub.groupby('cz', observed=True)}
    res['by_climate_zone'] = by_cz

    # ---- 5. 坡向 OR（对 strict / clean / fcs10only / 池标签 各算一遍）
    ors = {}
    df['is_enf_pool'] = (df['cl_24'] == 6)
    df['is_ebf_pool'] = (df['cl_24'] == 4)
    df['is_dnf_pool'] = (df['cl_24'] == 7)
    df['is_dbf_pool'] = (df['cl_24'] == 5)

    ors['pool_ENF_vs_EBF'] = or_simple(df, 'is_enf_pool', 'is_ebf_pool')

    for nm in ['vote_strict', 'vote_clean', 'fcs10_only']:
        d2 = df.copy()
        if nm == 'fcs10_only':
            # 只在 FCS10 有值处比
            pass
        d2['a'] = (d2[nm] == 1)
        d2['b'] = (d2[nm] == 0)
        ors[f'{nm}_N_vs_B'] = or_simple(d2, 'a', 'b')
        # 同时按「真针 vs 真阔」在 strict 内做两两对照（ENF vs EBF 近似：用 pheno 轴 E 且 leaf N/B）
        dE = d2[(d2['pheno_axis'] == 'E')]
        ors[f'{nm}_ENF_vs_EBF_phenoE'] = or_simple(dE, 'a', 'b')

    res['aspect_or'] = ors

    # ---- 6. 与 FCS10 单源的差异
    both = df[df['vote_strict'].notna() & df['leaf_fcs10_s'].notna()]
    if len(both):
        res['strict_minus_fcs10'] = dict(
            n_both=int(len(both)),
            same=int((both['vote_strict'] == both['leaf_fcs10_s']).sum()),
            same_pct=round(float((both['vote_strict'] == both['leaf_fcs10_s']).mean()) * 100, 2),
            strict_only_n=int((both['vote_strict'].notna() & both['leaf_fcs10_s'].isna()).sum()))
    # strict 中有多少是 FCS10 未覆盖的（靠 MODIS+CGLS 达成 2 源）
    s = df[df['vote_strict'].notna()]
    res['strict_fcs10_missing_share'] = round(float((~s['fcs10_covered']).mean()) * 100, 2)

    # ---- 7. 产出 strict 点表
    cols = ['stable_uid', 'lon', 'lat', 'cl_24', 'cl_24_orig', 'tile', 'prov_cta', 'cz',
            'dem_nasadem', 'slope', 'aspect', 'sun', 'stratum',
            'leaf_label_s', 'leaf_modis_s', 'leaf_fcs10_s',
            'ext_dec', 'ext_N', 'ext_B', 'vote_strict', 'vote_clean',
            'amp2019', 'amp2023', 'pheno_axis', 'pheno_2019_s', 'pheno_2023_s', 'pheno_modis_s',
            'n_years']
    if 'leaf_cgls_s' in df.columns:
        cols.insert(15, 'leaf_cgls_s')
    st = df[df['vote_strict'].notna()][cols].copy()
    st['leaf_final'] = np.where(st['vote_strict'] == 1, 'N', 'B')
    st['pheno_final'] = st['pheno_axis']
    st['cl_final'] = np.where(st['vote_strict'] == 1,
                              np.where(st['pheno_axis'] == 'E', 6, 7),
                              np.where(st['pheno_axis'] == 'E', 4, 5))
    st.loc[st['pheno_axis'] == '?', 'cl_final'] = 0
    out = os.path.join(WORK, 'strict_vote.parquet')
    st.to_parquet(out, index=False)
    res['strict_vote_file'] = dict(path=out, n=int(len(st)))
    res['strict_class_counts'] = {int(k): int(v) for k, v in st['cl_final'].value_counts().sort_index().items()}

    # 清点
    res['n_total'] = int(len(df))
    js = os.path.join(WORK, 'm1_vote.json')
    json.dump(res, open(js, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ('by_climate_zone',)}, ensure_ascii=False, indent=1, default=str)[:6000])
    print('WROTE', js, out)


if __name__ == '__main__':
    main()
