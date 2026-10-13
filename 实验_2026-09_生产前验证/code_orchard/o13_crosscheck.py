# -*- coding: utf-8 -*-
"""o13_crosscheck.py — GEM-Forest × 我方样本/判别器 交叉验证（三重证据一致性）

* 输入：data/orchard/orchard_samples_2023.csv（P1_hi 111 + P2_vol 1268 + P0）
        data/orchard/judge2_merged.csv（上轮 480 判读点，52 正）
        data/orchard/gem_cn.parquet + gem/feats_y{2020,2023}.parquet（GEM 中国 610 候选）
* 规则源：
  A. GEE 采样 GEM b1 值于我方 P1/P2 与上轮判读正/负点 → 一致性表
  B. 用上轮判读（judge2）训练 AEF64 判别器（同 o5 配方）→ 给 GEM 610 候选打分
* 输出：results/orchard/gem_cross.json + data/orchard/gem_cross.csv
* 用法：python o13_crosscheck.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
RES = os.path.join(WORK, 'results', 'orchard')


def emit(m):
    import time
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def gem_at(ee, pts, tag):
    gem = ee.ImageCollection('projects/danielp-cuni/assets/GEM-Forest/GEM-Forest_2020').mosaic().select('b1')
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'i': i})
             for i, r in enumerate(pts.itertuples())]
    got = gem.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                            tileScale=4).getInfo()['features']
    d = pd.DataFrame([{'i': g['properties']['i'], 'gem': g['properties'].get('b1')} for g in got])
    out = pts.reset_index(drop=True).copy()
    out['i'] = np.arange(len(out))
    out = out.merge(d, on='i', how='left')
    emit('%s: GEM 取值 %s' % (tag, out.gem.value_counts(dropna=False).to_dict()))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    import ee
    VC.ensure_ctx('zixen8v8')
    out = {}
    # A) 我方 P1/P2
    sel = pd.read_csv(os.path.join(ORD, 'orchard_samples_2023.csv'), encoding='utf-8-sig')
    p12 = sel[sel.tier != 'P0_legacy'].copy()
    a = gem_at(ee, p12, 'P1P2')
    for t in ('P1_hi', 'P2_vol'):
        g = a[a.tier == t]
        out['gem_at_' + t] = {int(k): int(v) for k, v in g.gem.fillna(-1).astype(int).value_counts().items()}
    # B) 上轮判读正/负
    j2 = pd.read_csv(os.path.join(ORD, 'judge2_merged.csv'), encoding='utf-8-sig')
    j2 = j2[j2.Q1.isin(['是', '否'])].copy()
    b = gem_at(ee, j2[['lon', 'lat']].assign(point_id=j2.point_id), 'judge2')
    b = b.merge(j2[['point_id', 'Q1']], on='point_id')
    b.to_csv(os.path.join(ORD, 'gem_at_judge2.csv'), index=False, encoding='utf-8-sig')
    out['gem_at_judge2_pos'] = {int(k): int(v) for k, v in b[b.Q1 == '是'].gem.fillna(-1).astype(int).value_counts().items()}
    out['gem_at_judge2_neg'] = {int(k): int(v) for k, v in b[b.Q1 == '否'].gem.fillna(-1).astype(int).value_counts().items()}
    # C) 判别器打分 GEM 610（用 2023 特征；train 用 judge2 的 2023 特征）
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ORD, 'feats', 'feat_*_p*.parquet')))],
                      ignore_index=True)
    f2 = feats[['lon', 'lat'] + VC.FEATS].round({'lon': 6, 'lat': 6})
    jm = j2.round({'lon': 6, 'lat': 6}).merge(f2, on=['lon', 'lat'], how='inner')
    X = jm[VC.FEATS].to_numpy('float32'); y = (jm.Q1 == '是').to_numpy(int)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced_subsample', n_jobs=14, random_state=7).fit(X, y)
    gc = pd.read_parquet(os.path.join(ORD, 'gem_cn.parquet'))
    for yr in (2020, 2023):
        gf = pd.read_parquet(os.path.join(ORD, 'gem', 'feats_y%d.parquet' % yr))
        ok = gf.aef_ok
        p = np.full(len(gf), np.nan)
        p[ok.to_numpy()] = clf.predict_proba(gf.loc[ok, VC.FEATS].to_numpy('float32'))[:, 1]
        gc['p_orchard_%d' % yr] = p
    gc.to_csv(os.path.join(ORD, 'gem_cross.csv'), index=False, encoding='utf-8-sig')
    out['gem_p_2020'] = {t: int((gc.p_orchard_2020 >= t).sum()) for t in (0.3, 0.5, 0.6, 0.7)}
    out['gem_p_2023'] = {t: int((gc.p_orchard_2023 >= t).sum()) for t in (0.3, 0.5, 0.6, 0.7)}
    emit('GEM610 判别器 p≥0.5: 2020 %d / 2023 %d' % (out['gem_p_2020'][0.5], out['gem_p_2023'][0.5]))
    # 与判读合并
    jm2 = pd.read_csv(os.path.join(ORD, 'gem_judge_merged.csv'), encoding='utf-8-sig')
    g2 = gc.merge(jm2[['point_id', 'Q1', 'Q2', 'Q3']], left_index=True, right_index=True, how='left')
    VC.jsave(out, os.path.join(RES, 'gem_cross.json'))
    emit(json.dumps(out, ensure_ascii=False)[:600])


if __name__ == '__main__':
    main()
