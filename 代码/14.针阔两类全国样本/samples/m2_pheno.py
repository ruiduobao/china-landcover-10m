# -*- coding: utf-8 -*-
"""m2_pheno.py — 方法B：S2 物候"合理性筛查"（只剔除明显矛盾，不判针阔）+ 方法F：多年稳定性
* 输入：labels/methods/work_base.parquet（含 amp2019/amp2023/djf2019/djf2023/pheno_*）
        F:/lc_work/年度子集_含稀有类/r7_train_YYYY.parquet（逐年 AEF64，用于"逐年稳定性"）
* 规则 B（预注册，写死）：
    B1 常绿针叶(ENF) 但两个物候年都"强落叶"（amp >= 落叶类 z 带阈 + margin 且 DJF>=150）→ 矛盾
    B2 落叶针叶(DNF)/落叶阔叶(DBF) 但物候明确常绿（amp <= 常绿阈 - margin）→ 矛盾  [仅落叶针叶做，DBF 不属针阔轴]
    B3 冬季完全落叶（DJF<200）却标"常绿"（ENF/EBF）→ 矛盾
  一律用"两年都矛盾"才剔除（保守），单年矛盾只 flag。
* 规则 F：n_years>=7 且 8 年 class_new 恒定 → stable；用 AEF 逐年向量余弦/欧氏距离作为"影像稳定性"。
* 输出：labels/methods/m2_pheno.json、labels/methods/m2_flags.parquet、labels/methods/m6_stable.json
* 用法：python m2_pheno.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
ANN = r'F:/lc_work/年度子集_含稀有类'
CAL = r'F:/lc_work/v31_exp/data/forest_qc/out/calibration.json'


def main():
    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'))
    res = {}
    cal = json.load(open(CAL, encoding='utf-8'))

    # z 带阈值（常绿 amp 上界）
    thr = {z: v['thr'] for z, v in cal.items()}
    gray = {z: v['gray'] for z, v in cal.items()}
    df['thr_amp'] = df['cz'].map(thr).fillna(326.1)
    df['gray_amp'] = df['cz'].map(gray).fillna(56.5)

    # --- B1: ENF 被两个物候年判为强落叶
    en = df['cl_24'] == 6
    amp19_d = (df['amp2019'] >= df['thr_amp'] + df['gray_amp'])
    amp23_d = (df['amp2023'] >= df['thr_amp'] + df['gray_amp'])
    djf_ok19 = df['djf2019'] >= 150
    djf_ok23 = df['djf2023'] >= 150
    b1 = en & amp19_d & amp23_d & djf_ok19 & djf_ok23
    b1_1yr = en & ((amp19_d & djf_ok19) | (amp23_d & djf_ok23))

    # --- B2: DNF 被两个物候年判为明确常绿
    dn = df['cl_24'] == 7
    amp19_e = df['amp2019'] <= df['thr_amp'] - df['gray_amp']
    amp23_e = df['amp2023'] <= df['thr_amp'] - df['gray_amp']
    b2 = dn & amp19_e & amp23_e & djf_ok19 & djf_ok23
    b2_1yr = dn & ((amp19_e & djf_ok19) | (amp23_e & djf_ok23))

    # --- B3: 冬季完全落叶（DJF<200）却标常绿（两年都如此）
    ever_all = df['cl_24'].isin([4, 6])
    b3 = ever_all & (df['djf2019'] < 200) & (df['djf2023'] < 200)
    b3_1yr = ever_all & ((df['djf2019'] < 200) | (df['djf2023'] < 200))

    # 反向：标落叶但两年都是"冬季满绿"（DJF>=700 且 amp<150）
    deci_all = df['cl_24'].isin([5, 7])
    b4 = deci_all & (df['djf2019'] >= 700) & (df['djf2023'] >= 700) & \
        (df['amp2019'] < 150) & (df['amp2023'] < 150)

    flags = pd.DataFrame(dict(
        stable_uid=df['stable_uid'], cl_24=df['cl_24'], tile=df['tile'],
        prov=df['prov_cta'], cz=df['cz'], dem=df['dem_nasadem'],
        amp2019=df['amp2019'], amp2023=df['amp2023'], djf2019=df['djf2019'], djf2023=df['djf2023'],
        B1=b1.values, B1_1yr=b1_1yr.values, B2=b2.values, B2_1yr=b2_1yr.values,
        B3=b3.values, B3_1yr=b3_1yr.values, B4=b4.values))
    flags['B_any'] = flags[['B1', 'B2', 'B3', 'B4']].any(axis=1)
    flags['B_any_1yr'] = flags[['B1_1yr', 'B2_1yr', 'B3_1yr', 'B4']].any(axis=1)
    flags.to_parquet(os.path.join(WORK, 'm2_flags.parquet'), index=False)

    res['n_total'] = int(len(df))
    for k in ['B1', 'B1_1yr', 'B2', 'B2_1yr', 'B3', 'B3_1yr', 'B4', 'B_any', 'B_any_1yr']:
        v = flags[k]
        res[k] = dict(n=int(v.sum()), pct=round(float(v.mean()) * 100, 3))
        # 分类别拆解
        res[k]['by_class'] = {int(c): int(g[k].sum()) for c, g in flags.groupby('cl_24')}

    # --- 空间分布：B1/B2 命中的省/瓦片
    for k in ['B1', 'B2', 'B3', 'B4']:
        sub = flags[flags[k]]
        res[k]['top_prov'] = sub['prov'].value_counts().head(8).to_dict()
        res[k]['top_tile'] = sub['tile'].value_counts().head(8).to_dict()
        res[k]['dem_median'] = float(sub['dem'].median()) if len(sub) else None

    # --- 与外部产品对照：B 命中点在 FCS10 里是什么
    b1u = set(flags.loc[flags['B1'], 'stable_uid'])
    d = df[df['stable_uid'].isin(b1u)]
    res['B1_fcs10_crosstab'] = {str(k): int(v) for k, v in d['fcs10_code'].value_counts().head(12).items()}
    res['B1_leaf_fcs10'] = {str(k): int(v) for k, v in d['leaf_fcs10'].value_counts(dropna=False).items()}
    res['B1_leaf_modis'] = {str(k): int(v) for k, v in d['leaf_modis'].value_counts(dropna=False).items()}

    # ============ 方法 F：多年稳定性 ============
    # (a) 标签层稳定性：n_years + yr_* 全 True + cl_24 跨年恒定（已知 row_id 跨年一致）
    all8 = df[['yr_2017', 'yr_2018', 'yr_2019', 'yr_2020', 'yr_2021', 'yr_2022', 'yr_2023', 'yr_2024']].all(axis=1)
    res['F_label_stable'] = dict(all8_years=int(all8.sum()),
                                 pct=round(float(all8.mean()) * 100, 2),
                                 n_years_ge7=int((df['n_years'] >= 7).sum()))

    # (b) 影像层稳定性：逐年 AEF64 的类内一致性
    uids = set(df['stable_uid'])
    clsmap = df.set_index('stable_uid')['cl_24'].to_dict()
    per_year = {}
    vecs = {}
    for y in range(2017, 2025):
        p = os.path.join(ANN, f'r7_train_{y}.parquet')
        d = pd.read_parquet(p, columns=['row_id'] + [f'A{i:02d}' for i in range(64)])
        d['row_id'] = d['row_id'].astype(str)
        d = d[d['row_id'].isin(uids)].drop_duplicates('row_id')
        per_year[y] = d.set_index('row_id')
    # 用 2019 与 2023（物候年）与 2017/2024 两端：余弦相似度
    keys = [('2017', '2024'), ('2019', '2023'), ('2017', '2019'), ('2022', '2024')]
    cos_stats = {}
    for a, b in keys:
        A = per_year[int(a)]; B = per_year[int(b)]
        common = A.index.intersection(B.index)
        va = A.loc[common, [f'A{i:02d}' for i in range(64)]].to_numpy(dtype=np.float32)
        vb = B.loc[common, [f'A{i:02d}' for i in range(64)]].to_numpy(dtype=np.float32)
        na = np.linalg.norm(va, axis=1); nb = np.linalg.norm(vb, axis=1)
        ok = (na > 0) & (nb > 0)
        cos = np.full(len(common), np.nan, dtype=np.float32)
        cos[ok] = (va[ok] * vb[ok]).sum(1) / (na[ok] * nb[ok])
        eu = np.linalg.norm(va - vb, axis=1)
        cos_stats[f'{a}-{b}'] = dict(n=int(len(common)),
                                     cos_median=float(np.nanmedian(cos)),
                                     cos_p10=float(np.nanpercentile(cos, 10)),
                                     euclid_median=float(np.nanmedian(eu)))
        if (a, b) == ('2017', '2024'):
            df['cos_1724'] = pd.Series(cos, index=common).reindex(df['stable_uid']).values
            df['eu_1724'] = pd.Series(eu, index=common).reindex(df['stable_uid']).values
    res['F_aef_cos'] = cos_stats

    # 稳定 vs 不稳定子集的针阔一致率（用 FCS10 vs 池标签，作为"标签是否可信"的代理）
    q = df['cos_1724'].quantile([0.2, 0.8]).values
    df['aef_stable'] = df['cos_1724'] >= q[1]
    df['aef_unstable'] = df['cos_1724'] <= q[0]
    fcs = df['leaf_fcs10_s'].notna() & df['leaf_label_s'].notna()
    def agree(sub):
        s = sub[fcs.reindex(sub.index).fillna(False)] if False else sub
        m = s[['leaf_fcs10_s', 'leaf_label_s']].dropna()
        return dict(n=int(len(m)), agree=round(float((m.iloc[:, 0] == m.iloc[:, 1]).mean()), 4)) if len(m) else None
    res['F_leaf_agree_by_stability'] = dict(
        stable_20pct=agree(df[df['aef_stable']]),
        unstable_20pct=agree(df[df['aef_unstable']]),
        all=agree(df))
    res['F_stability_thresholds'] = dict(cos_p20=float(q[0]), cos_p80=float(q[1]))

    # 每类 AEF 稳定性
    res['F_cos_by_class'] = {int(c): dict(n=int(g['cos_1724'].notna().sum()),
                                          cos_median=float(g['cos_1724'].median()),
                                          cos_p25=float(g['cos_1724'].quantile(.25)))
                             for c, g in df.groupby('cl_24')}

    json.dump(res, open(os.path.join(WORK, 'm2_pheno.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:8000])
    df[['stable_uid', 'cos_1724', 'eu_1724', 'aef_stable', 'aef_unstable']].to_parquet(
        os.path.join(WORK, 'm6_stability.parquet'), index=False)


if __name__ == '__main__':
    main()
