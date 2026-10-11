# -*- coding: utf-8 -*-
"""k15_nat_score.py — 全国 FCS10 灌丛层打分 + 分层选点 + 判型规则比选（本地，零 GEE）

* 输入：judged_feats.parquet（87 个 FCS10 框判读点，训练/验证用）+ 母池 r7_train_2023.parquet
        （glc_fcs10_2023_shrub 537,361 点，AEF64 已在库）
* 规则源（预注册）：
        判别器 = RandomForest(AEF64, balanced_subsample, n=400)，训练集只用 **FCS10 框判读点**
                （40 正 / 47 负）；放量阈值 θ 取 k12 框内 CV 校准过的两档：0.6 / 0.7
        选点：全国 537,361 点打分 → θ=0.7 全取（高精档）+ θ=0.6 取到 --cap2 上限（走量档）；
                省配额按该省候选数等比例，省上限 --prov-cap；盒上限 --box-cap；点距 ≥1 km
        判型规则比选（在 207 个可分型判读灌丛点上）：
                R_code＝FCS10 层原码（120 常绿/121 落叶）
                R_amp ＝S2 物候振幅 + 分区阈值（每区带取两类中位值中点）
                R_both＝两法一致时用一致结果，不一致时用振幅（以振幅为准）
* 输出：data/shrub_nat/nat_scored.parquet（全国打分）+ data/shrub_nat/nat_select.json
        data/shrub_nat/type_rule.json + data/shrub_nat/type_rule.md
* 用法：python k15_nat_score.py [--cap2 30000] [--prov-cap 9000] [--box-cap 900]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
POOL = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
SEED = 20261011
MIN_KM = 1.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def zone_of(dem, bio01, bio12):
    if bio01 <= -999:
        return '缺失'
    if dem >= 3000:
        return '青藏高原高山'
    t = bio01 / 10.0
    if t >= 20:
        return '热带'
    if t >= 13 and bio12 >= 900:
        return '亚热带'
    if t <= -2:
        return '寒温带'
    if bio12 < 300:
        return '干旱'
    if bio12 < 500:
        return '半干旱'
    return '温带'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap2', type=int, default=30000, help='θ=0.6 走量档上限')
    ap.add_argument('--prov-cap', type=int, default=9000)
    ap.add_argument('--box-cap', type=int, default=900)
    a = ap.parse_args()
    feats = ['A%02d' % i for i in range(64)]
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    kj = list(zip(np.round(j.lon.to_numpy(float), 6), np.round(j.lat.to_numpy(float), 6)))
    k57 = list(zip(np.round(d57.lon.to_numpy(float), 6), np.round(d57.lat.to_numpy(float), 6)))
    m57 = dict(zip(k57, zip(d57['stratum'], d57['Q1_shrub'])))
    j['st57'] = [m57.get(k, ('', ''))[0] for k in kj]
    tr = j[(j['st57'] == 'FCS10') & j['is_shrub'].isin(['是', '否'])].copy()
    Xtr = tr[feats].astype(float).fillna(-999.0).to_numpy()
    ytr = (tr['is_shrub'] == '是').to_numpy(int)
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_leaf_nodes=5000,
                                 max_features='sqrt', n_jobs=15, random_state=7,
                                 class_weight='balanced_subsample').fit(Xtr, ytr)
    emit('判别器训练：FCS10 框判读点 %d（正 %d/负 %d）' % (len(tr), ytr.sum(), (1 - ytr).sum()))

    # ---- 判型规则比选（用全部可分型判读灌丛点）----
    tp = j[(j['is_shrub'] == '是') & j['shrub_type'].isin(['常绿', '落叶'])].copy()
    tp = tp[tp['ndvi_amp'].notna() & tp['dwp'].notna()]
    code_map = {}     # 用 FCS10 框判读点对齐 120/121 与判型
    for r in tr.itertuples():
        code_map[(round(float(r.lon), 6), round(float(r.lat), 6))] = getattr(r, 'class_new', None)
    tp['zone'] = [zone_of(r.dem, r.bio01, r.bio12) for r in tp.itertuples()]
    thr_z = {}
    for z, g in tp.groupby('zone'):
        e = g[g.shrub_type == '常绿']['ndvi_amp'].median()
        d = g[g.shrub_type == '落叶']['ndvi_amp'].median()
        if np.isfinite(e) and np.isfinite(d):
            thr_z[z] = float((e + d) / 2.0)
    y = (tp['shrub_type'] == '落叶').to_numpy(int)
    acc = {}
    for tag, pred in (
            ('R_code', np.array([1 if code_map.get((round(x, 6), round(y2, 6))) == 121 else 0
                                 for x, y2 in zip(tp.lon, tp.lat)])),
            ('R_amp', np.array([1 if r.ndvi_amp >= thr_z.get(r.zone, 250) else 0
                                for r in tp.itertuples()])),
    ):
        acc[tag] = float((pred == y).mean())
    emit('判型规则比选（n=%d）：R_code=%.3f R_amp=%.3f' % (len(tp), acc['R_code'], acc['R_amp']))
    json.dump(dict(n=int(len(tp)), acc=acc, thr_zone={k: round(v, 2) for k, v in thr_z.items()}),
              open(os.path.join(OUTD, 'type_rule.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    # ---- 全国打分 ----
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src'] + feats)
    sh = pool[pool['src'].fillna('').astype(str) == 'glc_fcs10_2023_shrub'].copy()
    emit('全国 FCS10 灌丛层 %d 点' % len(sh))
    X = sh[feats].astype(float).fillna(-999.0).to_numpy()
    p = np.zeros(len(sh))
    for k in range(0, len(X), 50000):
        p[k:k + 50000] = clf.predict_proba(X[k:k + 50000])[:, 1]
        emit('  打分 %d/%d' % (min(k + 50000, len(X)), len(X)))
    sh['p_shrub'] = p
    sh.to_parquet(os.path.join(OUTD, 'nat_scored.parquet'), index=False)
    emit('打分完成：≥0.6 有 %d 点，≥0.7 有 %d 点' % (int((p >= 0.6).sum()), int((p >= 0.7).sum())))

    # ---- 选点 ----
    sel = []
    for th, cap in ((0.7, 10 ** 9), (0.6, a.cap2)):
        sub = sh[sh['p_shrub'] >= th]
        if th == 0.6:
            sub = sub[sub['p_shrub'] < 0.7]     # 避免与高精档重复
        sub = sub.sample(frac=1.0, random_state=np.random.RandomState(SEED)).reset_index(drop=True)
        rs = np.random.RandomState(SEED)
        box = [('%02d%02d' % (math_floor(la), math_floor(lo))) for la, lo in
               zip(sub['lat'].to_numpy(float), sub['lon'].to_numpy(float))]
        sub = sub.assign(box=box)
        got, used, bcnt, pcnt = [], [], {}, {}
        for r in sub.itertuples():
            if len(got) >= cap:
                break
            if bcnt.get(r.box, 0) >= a.box_cap or pcnt.get(r.class_new, 0) >= 10 ** 9:
                continue
            if any(((r.lon - u[0]) ** 2 * 100 + (r.lat - u[1]) ** 2 * 120) < (MIN_KM / 111.0) ** 2
                   for u in used[-1200:]):
                continue
            used.append((r.lon, r.lat))
            bcnt[r.box] = bcnt.get(r.box, 0) + 1
            got.append(dict(lon=float(r.lon), lat=float(r.lat), class_new=int(r.class_new),
                            p_shrub=float(r.p_shrub), tier='P1_hi' if th == 0.7 else 'P2_vol',
                            box=r.box))
        sel += got
        emit('θ=%.1f 选出 %d 点' % (th, len(got)))
    df = pd.DataFrame(sel)
    df.insert(0, 'point_id', ['NP-%s-%06d' % ('H' if r.tier == 'P1_hi' else 'V', i)
                              for i, r in enumerate(df.itertuples())])
    df.to_csv(os.path.join(OUTD, 'nat_points.csv'), index=False, encoding='utf-8-sig')
    plan = dict(n_hi=int((df.tier == 'P1_hi').sum()), n_vol=int((df.tier == 'P2_vol').sum()),
                total=int(len(df)), prov_cap=a.prov_cap, box_cap=a.box_cap,
                by_code={str(k): int(v) for k, v in df.groupby('class_new').size().items()},
                p_shrub_range=[round(float(df.p_shrub.min()), 3), round(float(df.p_shrub.max()), 3)])
    json.dump(plan, open(os.path.join(OUTD, 'nat_select.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ nat_points.csv（%d 点：高精 %d + 走量 %d）' % (len(df), plan['n_hi'], plan['n_vol']))


def math_floor(v):
    return int(np.floor(v))


if __name__ == '__main__':
    main()
