# -*- coding: utf-8 -*-
"""m5_anchor_boot.py — 方法G：高置信锚点自举（弱监督）+ 判别器 + 坡向 OR 检验
* 输入：labels/methods/work_base.parquet、m2_flags.parquet、m4_flags.parquet
        labels/methods/gee_struct_sample.parquet（若 m3 已完成；可选 CGLS 第三源）
        F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet（AEF64）
* 锚点四条件（预注册）：
    A1 多产品一致：≥2 外部源同向且无外部源反向（等价 m1 vote_strict）
    A2 物候自洽：不命中 m2 的 B_any（无明显矛盾）
    A3 地形生态合理：不命中 m4 的 D_implausible
    A4 多年稳定：n_years>=7 且 AEF 2017~2024 余弦 >= p20
* 判别器：LDA / 逻辑回归（sklearn），特征 = AEF64；空间块 CV（2°瓦 5 折）
* 扩增：对非锚点全部林点预测 P(needle)，仅取 P>=0.90 或 <=0.10 且 ext 冲突（ext_N>0 且 ext_B>0）的低冲突点
* 关键检验：扩增前后的 ENF vs EBF 坡向 MH-OR 是否漂移
* 输出：labels/methods/anchor_boot.parquet、labels/methods/m5_anchor.json
* 用法：python m5_anchor_boot.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
ANN = r'F:/lc_work/年度子集_含稀有类'
AEF = [f'A{i:02d}' for i in range(64)]


def mh_or(d, a_mask, b_mask):
    """MH OR over stratum（2°瓦×高程带），a/b 为布尔掩码"""
    num = den = 0.0; ns = 0
    for s, g in d.groupby('stratum', observed=True):
        a_s = int(((a_mask.reindex(g.index)) & (g['sun'] == 'shade')).sum())
        a_u = int(((a_mask.reindex(g.index)) & (g['sun'] == 'sun')).sum())
        b_s = int(((b_mask.reindex(g.index)) & (g['sun'] == 'shade')).sum())
        b_u = int(((b_mask.reindex(g.index)) & (g['sun'] == 'sun')).sum())
        n = a_s + a_u + b_s + b_u
        if min(a_s, a_u, b_s, b_u) <= 0 or n == 0:
            continue
        num += a_s * b_u / n; den += a_u * b_s / n; ns += 1
    return (round(num / den, 4) if den > 0 else None), ns


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def main():
    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'))
    df['stable_uid'] = df['stable_uid'].astype(str)
    res = {}

    # --- 并入可选层
    pf = os.path.join(WORK, 'm2_flags.parquet')
    if os.path.exists(pf):
        f2 = pd.read_parquet(pf)
        f2['stable_uid'] = f2['stable_uid'].astype(str)
        df = df.merge(f2[['stable_uid', 'B_any', 'B1', 'B2', 'B3', 'B4']], on='stable_uid', how='left')
    qf = os.path.join(WORK, 'm4_flags.parquet')
    if os.path.exists(qf):
        f4 = pd.read_parquet(qf)
        f4['stable_uid'] = f4['stable_uid'].astype(str)
        df = df.merge(f4[['stable_uid', 'D_implausible', 'D5_eco_ok']], on='stable_uid', how='left')
    sf = os.path.join(WORK, 'm6_stability.parquet')
    if os.path.exists(sf):
        f6 = pd.read_parquet(sf)
        f6['stable_uid'] = f6['stable_uid'].astype(str)
        df = df.merge(f6[['stable_uid', 'cos_1724']], on='stable_uid', how='left')
    gf = os.path.join(WORK, 'gee_struct_sample.parquet')
    if os.path.exists(gf):
        fg = pd.read_parquet(gf)
        fg['stable_uid'] = fg['stable_uid'].astype(str)
        df = df.merge(fg[['stable_uid', 'leaf_cgls_s', 'cgls_ft', 'cgls_tcd',
                          'hansen_tc2000', 'hansen_lossyr']], on='stable_uid', how='left')
        print('merged CGLS/Hansen n=', df['leaf_cgls_s'].notna().sum())

    # --- 重新计算外部投票（含 CGLS 若可用）
    ext_cols = ['leaf_modis_s', 'leaf_fcs10_s']
    if 'leaf_cgls_s' in df.columns:
        ext_cols.append('leaf_cgls_s')
    E = df[ext_cols]
    df['ext_dec'] = E.notna().sum(axis=1)
    df['ext_N'] = (E == 1).sum(axis=1)
    df['ext_B'] = (E == 0).sum(axis=1)
    df['vote_strict'] = np.where((df['ext_dec'] >= 2) & (df['ext_N'] >= 2) & (df['ext_B'] == 0), 1,
                                 np.where((df['ext_dec'] >= 2) & (df['ext_B'] >= 2) & (df['ext_N'] == 0), 0, np.nan))
    res['vote_strict_with_optional_sources'] = dict(
        n=int(df['vote_strict'].notna().sum()),
        needle=int((df['vote_strict'] == 1).sum()),
        broad=int((df['vote_strict'] == 0).sum()),
        sources=ext_cols)

    # --- 锚点四条件
    A1 = df['vote_strict'].notna()
    A2 = ~df['B_any'].fillna(False).astype(bool)
    A3 = ~df['D_implausible'].fillna(False).astype(bool)
    A4 = (df['n_years'] >= 7) & (df['cos_1724'].fillna(0) >= 0.9557)
    df['anchor'] = A1 & A2 & A3 & A4
    df['y_needle'] = df['vote_strict']
    for nm, m in [('A1', A1), ('A2', A2), ('A3', A3), ('A4', A4), ('anchor', df['anchor'])]:
        res[nm] = dict(n=int(m.sum()), pct=round(float(m.mean()) * 100, 2))
    an = df[df['anchor']]
    res['anchor_by_class'] = {int(c): int(v) for c, v in
                              np.where(an['y_needle'] == 1,
                                       np.where(an['pheno_axis'] == 'E', 6, 7),
                                       np.where(an['pheno_axis'] == 'E', 4, 5)).tolist() and
                              pd.Series(np.where(an['y_needle'] == 1,
                                                 np.where(an['pheno_axis'] == 'E', 6, 7),
                                                 np.where(an['pheno_axis'] == 'E', 4, 5))).value_counts().items()}
    res['anchor_by_prov'] = an['prov_cta'].value_counts().head(10).to_dict()

    # --- 读 AEF64（2023 年，作为特征）
    cols = ['row_id'] + AEF
    a = pd.read_parquet(os.path.join(ANN, 'r7_train_2023.parquet'), columns=cols)
    a['stable_uid'] = a['row_id'].astype(str)
    a = a.drop(columns=['row_id']).drop_duplicates('stable_uid')
    df = df.merge(a, on='stable_uid', how='left')
    res['aef_missing'] = int(df[AEF[0]].isna().sum())

    # --- 空间块 CV（2°瓦 5 折）
    tiles = df['tile'].astype(str).unique()
    rng = np.random.RandomState(42)
    tfold = {t: i % 5 for i, t in enumerate(rng.permutation(tiles))}
    df['fold'] = df['tile'].map(tfold)

    has = df[AEF].notna().all(axis=1)
    lab = df['y_needle'].notna() & has
    res['n_trainable'] = int(lab.sum())

    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    models = {
        'lda': lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        'logit': lambda: make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')),
    }
    cv = {}
    preds = {}
    for nm, mk in models.items():
        p = np.full(len(df), np.nan)
        aucs = []
        for f in range(5):
            tr = lab & (df['fold'] != f)
            te = lab & (df['fold'] == f)
            m = mk()
            m.fit(df.loc[tr, AEF].to_numpy(np.float32), df.loc[tr, 'y_needle'].to_numpy())
            p[te.values] = m.predict_proba(df.loc[te, AEF].to_numpy(np.float32))[:, 1]
            aucs.append(auc(df.loc[te, 'y_needle'].to_numpy(), p[te.values]))
        cv[nm] = dict(auc_folds=[round(x, 4) if x else None for x in aucs],
                      auc_mean=round(float(np.nanmean(aucs)), 4))
        preds[nm] = p
    res['cv'] = cv
    df['p_needle'] = preds['lda']
    # 非标签行（unlab）无折外预测 → 用全量标签重训一个模型，对全体打分（扩增用）
    mfull = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    mfull.fit(df.loc[lab, AEF].to_numpy(np.float32), df.loc[lab, 'y_needle'].to_numpy())
    pall = np.full(len(df), np.nan)
    idx_has = has.to_numpy()
    pall[idx_has] = mfull.predict_proba(df.loc[has, AEF].to_numpy(np.float32))[:, 1]
    df['p_needle_full'] = pall

    # --- 与池标签口径对照（说明 CV 是label-fit而非真可分）
    res['cv_note'] = ('CV 的 y 来自多产品投票（外部源），不是人工真值；AUC 高说明 AEF 能拟合投票规则，'
                      '不等于能分真针阔（doc54 独立判读 AUC 0.724）。')

    # --- 扩增：低冲突 + 高置信
    unlab = (~lab) & has
    conflict = (df['ext_N'] > 0) & (df['ext_B'] > 0)
    low_conf = ~conflict
    hi = ((df['p_needle_full'] >= 0.90) | (df['p_needle_full'] <= 0.10))
    df['boot'] = np.where(unlab & low_conf & hi,
                          (df['p_needle_full'] >= 0.90).astype(int), np.nan)
    res['boot'] = dict(n=int(pd.Series(df['boot']).notna().sum()),
                       needle=int((df['boot'] == 1).sum()),
                       broad=int((df['boot'] == 0).sum()),
                       candidates_unlab=int(unlab.sum()),
                       conflict_share_pct=round(float(conflict.mean()) * 100, 2))

    # --- 坡向 OR：锚点 / 扩增 / 合并
    df['is_anchor_enf'] = (df['anchor']) & (df['y_needle'] == 1) & (df['pheno_axis'] == 'E')
    df['is_anchor_ebf'] = (df['anchor']) & (df['y_needle'] == 0) & (df['pheno_axis'] == 'E')
    df['is_boot_enf'] = (df['boot'] == 1) & (df['pheno_axis'] == 'E')
    df['is_boot_ebf'] = (df['boot'] == 0) & (df['pheno_axis'] == 'E')
    df['is_all_enf'] = df['is_anchor_enf'] | df['is_boot_enf']
    df['is_all_ebf'] = df['is_anchor_ebf'] | df['is_boot_ebf']
    ors = {}
    for k, (a_, b_) in {'anchor': ('is_anchor_enf', 'is_anchor_ebf'),
                        'boot': ('is_boot_enf', 'is_boot_ebf'),
                        'anchor+boot': ('is_all_enf', 'is_all_ebf')}.items():
        o, ns = mh_or(df, df[a_], df[b_])
        ors[k] = dict(mh_or=o, n_strata=ns,
                      n_enf=int(df[a_].sum()), n_ebf=int(df[b_].sum()),
                      enf_shade_pct=round(float((df.loc[df[a_], 'sun'] == 'shade').mean()) * 100, 2)
                      if df[a_].sum() else None,
                      ebf_shade_pct=round(float((df.loc[df[b_], 'sun'] == 'shade').mean()) * 100, 2)
                      if df[b_].sum() else None)
    # 池标签对照
    o, ns = mh_or(df, df['cl_24'] == 6, df['cl_24'] == 4)
    ors['pool_label_6_vs_4'] = dict(mh_or=o, n_strata=ns)
    res['aspect_or'] = ors

    # --- 产出
    keep = ['stable_uid', 'lon', 'lat', 'cl_24', 'tile', 'prov_cta', 'cz', 'dem_nasadem',
            'slope', 'aspect', 'sun', 'stratum', 'leaf_label_s', 'leaf_modis_s', 'leaf_fcs10_s',
            'ext_dec', 'ext_N', 'ext_B', 'vote_strict', 'y_needle', 'p_needle', 'anchor', 'boot',
            'pheno_axis', 'amp2019', 'amp2023', 'cos_1724', 'n_years', 'fold']
    if 'leaf_cgls_s' in df.columns:
        keep.insert(13, 'leaf_cgls_s')
    out = df[df['anchor'] | df['boot'].notna()][keep].copy()
    out['leaf_final'] = np.where(out['anchor'], out['y_needle'], out['boot'])
    out['source'] = np.where(out['anchor'], 'anchor', 'boot')
    op = os.path.join(WORK, 'anchor_boot.parquet')
    out.to_parquet(op, index=False)
    res['file'] = dict(path=op, n=int(len(out)),
                       n_anchor=int((out['source'] == 'anchor').sum()),
                       n_boot=int((out['source'] == 'boot').sum()))

    json.dump(res, open(os.path.join(WORK, 'm5_anchor.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:7000])


if __name__ == '__main__':
    main()
