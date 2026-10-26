# -*- coding: utf-8 -*-
"""m4_eco_rules.py — 方法D：区域化"生态不可能组合"规则（只标记复核区，不改标签）
* 输入：labels/methods/work_base.parquet
* 规则（预注册，写死；命中只打 flag，不删点）：
    D1 热带低地针叶（纬度<25° 且 dem<500m 且 biome 属湿热）——华南/滇南自然林不应有大片针叶
    D2 南亚热带常绿阔叶核心区针叶（18-25°N, 105-120°E, dem<800m, bio12>1400）——广东伪影区
    D3 极干旱针叶（bio12<200 且 dem<1500）——沿用 doc65 R3
    D4 冷区低海拔常绿阔叶（bio11<=0 且 dem<2000）——沿用 doc65 R1
    D5 高寒针叶合理区（dem>=3000 或 bio06<=-100）→ 反向"高证据"标记（供 G 锚点用）
* 输出：labels/methods/m4_eco.json、labels/methods/m4_flags.parquet
* 用法：python m4_eco_rules.py
"""
import os, sys, json
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'


def main():
    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'))
    res = dict(n_total=int(len(df)))

    lat, lon = df['lat'], df['lon']
    dem, bio12, bio11, bio06 = df['dem_nasadem'], df['bio12'], df['bio11'], df['bio06']
    isN = df['cl_24'].isin([6, 7])
    isB = df['cl_24'].isin([4, 5])
    isE = df['cl_24'].isin([4, 6])

    # D1 热带低地针叶
    d1 = isN & (lat < 25) & (dem < 500)
    # D2 南亚热带常绿阔叶核心区低地针叶
    d2 = isN & lat.between(18, 25) & lon.between(105, 120) & (dem < 800) & (bio12 > 1400)
    # D3 极干旱针叶
    d3 = isN & (bio12 < 200) & (dem < 1500)
    # D4 冷区低海拔常绿阔叶
    d4 = (df['cl_24'] == 4) & (bio11 <= 0) & (dem < 2000)
    # D5 高寒针叶合理（不做剔除，做"生态合理"正证据）
    d5 = isN & ((dem >= 3000) | (bio06 <= -100))
    # D6 温带阔叶被标针叶的反向检查：北方针叶带（>45N, dem<800）标阔叶 → 提示
    d6 = isB & (lat > 46) & (dem < 800) & (bio06 < -180)

    flags = pd.DataFrame(dict(stable_uid=df['stable_uid'], cl_24=df['cl_24'],
                              lat=lat, lon=lon, dem=dem, bio12=bio12, bio11=bio11, bio06=bio06,
                              region=df['region'], prov=df['prov_cta'], tile=df['tile'],
                              D1=d1.values, D2=d2.values, D3=d3.values, D4=d4.values,
                              D5_eco_ok=d5.values, D6=d6.values))
    flags['D_implausible'] = flags[['D1', 'D2', 'D3', 'D4', 'D6']].any(axis=1)
    flags.to_parquet(os.path.join(WORK, 'm4_flags.parquet'), index=False)

    for k in ['D1', 'D2', 'D3', 'D4', 'D5_eco_ok', 'D6', 'D_implausible']:
        v = flags[k]
        res[k] = dict(n=int(v.sum()), pct_of_all=round(float(v.mean()) * 100, 3))
        sub = flags[v]
        res[k]['by_class'] = {int(c): int(g[k].sum()) for c, g in flags.groupby('cl_24')}
        if len(sub):
            res[k]['top_prov'] = {str(a): int(b) for a, b in sub['prov'].value_counts().head(6).items()}
            res[k]['dem_median'] = float(sub['dem'].median())

    # --- 与外部产品一致性：规则命中点 vs FCS10 / MODIS
    for k in ['D1', 'D2', 'D6']:
        u = set(flags.loc[flags[k], 'stable_uid'])
        s = df[df['stable_uid'].isin(u)]
        res[k]['fcs10_leaf'] = {str(a): int(b) for a, b in s['leaf_fcs10'].value_counts(dropna=False).items()}
        res[k]['modis_leaf'] = {str(a): int(b) for a, b in s['leaf_modis'].value_counts(dropna=False).items()}
        res[k]['fcs10_code'] = {str(a): int(b) for a, b in s['fcs10_code'].value_counts().head(8).items()}

    # --- 规则命中率 ÷ 该区总林点数（"按区看"更公平）
    for k in ['D1', 'D2', 'D6']:
        sub = flags[flags[k]]
        if not len(sub):
            continue
        # 同区域全部林点
        if k == 'D1':
            zone = flags[(flags['lat'] < 25) & (flags['dem'] < 500)]
        elif k == 'D2':
            zone = flags[flags['lat'].between(18, 25) & flags['lon'].between(105, 120) &
                         (flags['dem'] < 800) & (flags['bio12'] > 1400)]
        else:
            zone = flags[(flags['lat'] > 46) & (flags['dem'] < 800) & (flags['bio06'] < -180)]
        res[k]['zone_total'] = int(len(zone))
        res[k]['hit_rate_in_zone'] = round(float(len(sub) / max(len(zone), 1)) * 100, 2)

    # --- 生态规则的"可分性"价值：D5 正证据 vs D1/D2 负证据 的类别分布
    res['D5_vs_D12_class_dist'] = dict(
        D5={int(c): int(v) for c, v in flags[flags['D5_eco_ok']]['cl_24'].value_counts().items()},
        D12={int(c): int(v) for c, v in flags[flags['D1'] | flags['D2']]['cl_24'].value_counts().items()})

    json.dump(res, open(os.path.join(WORK, 'm4_eco.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:5000])


if __name__ == '__main__':
    main()
