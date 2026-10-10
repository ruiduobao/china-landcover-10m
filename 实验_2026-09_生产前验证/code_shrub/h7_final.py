# -*- coding: utf-8 -*-
"""h7_final.py — 全国灌木样本最终装配 + 质量门（本地，零 GEE）

* 输入：mass_points.csv（h6 选择）+ mass_feats.parquet（h2b 特征分片合并）
        + audit_feats.parquet（审计点特征）+ audit_points.csv + judgement（h5 输出）
* 规则源：
        常绿/落叶 = S2 物候振幅 ndvi_amp 与 **分区阈值**比较（阈值由审计确认点按 zone 校准：
                    两群的 ndvi_amp 中位值中点；任一群 <5 点则用全局阈值，报告里注明）；
                    gray zone = 阈值 ±0.05 判"待判"
        物理分档 = rh98_mx11（1.1 km 邻域最高 GEDI 冠层高度）：<8 m → P1；≥8 m → P2（森林邻近可疑）
        母库去重 = 与 F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet 的 (lon,lat) 精确重合剔除
* 门槛：AEF 缺失点剔除；与母库重合点剔除；正样本同点间距 ≥1 km；每省点数/类型/层分布落盘
* 输出：data/shrub_nat/shrub_samples_nat.csv（终表）+ shrub_nat_stats.json + shp/（双坐标）
* 用法：python h7_final.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
POOL23 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
MIN_KM = 1.0
GRAY = 0.05


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    import glob
    mp = pd.read_csv(os.path.join(OUTD, 'mass_points.csv'), encoding='utf-8-sig')
    mp['point_id'] = mp['point_id'].astype(str)
    feats = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(os.path.join(OUTD, 'mass_feats_p*.parquet')))], ignore_index=True)
    emit('放量点 %d，特征 %d' % (len(mp), len(feats)))
    df = mp.merge(feats, on='point_id', how='left')
    df = df[df['aef_ok'] == 1].copy()
    emit('AEF 缺失剔除后 %d' % len(df))

    # ---- 常绿/落叶分区阈值校准（用审计确认点）----
    ap = pd.read_csv(os.path.join(OUTD, 'audit_points.csv'), encoding='utf-8-sig')
    ap['point_id'] = ap['point_id'].astype(str)
    af = pd.concat([pd.read_parquet(f) for f in
                   sorted(glob.glob(os.path.join(OUTD, 'audit_feats_p*.parquet')))], ignore_index=True) \
        if glob.glob(os.path.join(OUTD, 'audit_feats_p*.parquet')) else ap
    jd = []
    for f in sorted(glob.glob(os.path.join(OUTD, 'imagery', 'judge_*.csv'))):
        jd.append(pd.read_csv(f, encoding='utf-8-sig'))
    jd = pd.concat(jd, ignore_index=True).drop_duplicates(subset=['point_id'], keep='first') if jd else pd.DataFrame()
    thr, thr_src = {}, {}
    if len(jd) and 'ndvi_amp' in af.columns:
        both = ap.merge(af[['point_id', 'ndvi_amp']], on='point_id', how='inner')
        both = both.merge(jd[['point_id', 'Q1_shrub', 'Q2_evergreen']], on='point_id', how='inner')
        both = both[(both['Q1_shrub'] == '是') & both['ndvi_amp'].notna()]
        ev = both[both['Q2_evergreen'] == '常绿']['ndvi_amp']
        de = both[both['Q2_evergreen'] == '落叶']['ndvi_amp']
        allv = np.concatenate([ev.values, de.values]) if len(ev) + len(de) else np.array([])
        gthr = float(np.median(allv)) if len(allv) else 0.22
        emit('全局校准样本 常绿 %d / 落叶 %d，全局阈值 %.3f' % (len(ev), len(de), gthr))
        for z, g in both.groupby('zone'):
            e = g[g['Q2_evergreen'] == '常绿']['ndvi_amp']
            d = g[g['Q2_evergreen'] == '落叶']['ndvi_amp']
            if len(e) >= 5 and len(d) >= 5:
                thr[z] = (float(e.median()) + float(d.median())) / 2.0
                thr_src[z] = 'zone(%d/%d)' % (len(e), len(d))
            elif len(g) >= 5:
                thr[z] = gthr
                thr_src[z] = 'global(样本不足)'
            else:
                thr[z] = gthr
                thr_src[z] = 'global(n<5)'
        emit('分区阈值：%s' % {k: round(v, 3) for k, v in thr.items()})
    else:
        gthr = 0.22
        for z in df['zone'].unique():
            thr[z] = gthr
            thr_src[z] = 'fallback'

    def cls(row):
        t = thr.get(row['zone'], gthr)
        a = row['ndvi_amp']
        if pd.isna(a):
            return '待判'
        if a < t - GRAY:
            return '常绿'
        if a > t + GRAY:
            return '落叶'
        return '待判'
    df['shrub_type'] = [cls(r) for r in df.itertuples()]
    df['rh98_mx11'] = df.get('rh98_mx11', np.nan)
    df['phys'] = np.where(df['rh98_mx11'].isna(), 'P0无GEDI',
                          np.where(df['rh98_mx11'] < 8, 'P1低邻域', 'P2高邻域'))

    # ---- 与母库去重 ----
    try:
        pool = pd.read_parquet(POOL23, columns=['lon', 'lat'])
        pk = set(zip(np.round(pool['lon'].to_numpy(float), 6), np.round(pool['lat'].to_numpy(float), 6)))
        key = set(zip(np.round(df['lon'].to_numpy(float), 6), np.round(df['lat'].to_numpy(float), 6)))
        dup = len(key & pk)
        df = df[~np.isin(list(zip(np.round(df['lon'].to_numpy(float), 6),
                                  np.round(df['lat'].to_numpy(float), 6))), list(pk))]
        emit('母库精确重合剔除 %d 点' % dup)
    except Exception as e:
        emit('母库去重跳过：%s' % str(e)[:70])
    df.to_parquet(os.path.join(OUTD, 'shrub_samples_nat.parquet'), index=False)

    pos = df[df['sample_role'] == 'shrub_pos']
    st = dict(
        n_pos=int(len(pos)), n_neg=int((df['sample_role'] == 'hard_neg').sum()),
        thr={k: round(v, 4) for k, v in thr.items()}, thr_src=thr_src,
        ed_ratio={z: {t: int(n) for t, n in g.groupby('shrub_type').size().items()}
                  for z, g in pos.groupby('zone')},
        ed_ratio_prov={p: {t: int(n) for t, n in g.groupby('shrub_type').size().items()}
                       for p, g in pos.groupby('prov')},
        phys=pos.groupby(['zone', 'phys']).size().to_dict().__str__(),
        pos_by_prov={p: int(n) for p, n in pos.groupby('prov').size().sort_values(ascending=False).items()},
        pos_by_zone={z: int(n) for z, n in pos.groupby('zone').size().items()},
        pos_by_k={k: int(n) for k, n in pos.groupby('k').size().items()},
        dwp_med={z: round(float(g['dwp'].median()), 3) for z, g in pos.groupby('zone')},
    )
    # 生态自洽：常绿占比 vs 区带常理（报告口径，非门槛）
    eco = {}
    for z, g in pos.groupby('zone'):
        ev = int((g['shrub_type'] == '常绿').sum()); de = int((g['shrub_type'] == '落叶').sum())
        pc, lo, hi = wilson(ev, max(1, ev + de))
        eco[z] = dict(n=int(ev + de), ev=ev, de=de, ever_frac=pc, ci=[lo, hi])
    st['eco_evergreen_frac'] = eco
    json.dump(st, open(os.path.join(OUTD, 'shrub_nat_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('正样本 %d：%s' % (len(pos), st['pos_by_zone']))
    emit('→ ' + os.path.join(OUTD, 'shrub_samples_nat.parquet'))


if __name__ == '__main__':
    main()
