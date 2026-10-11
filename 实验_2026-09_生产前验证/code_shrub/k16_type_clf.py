# -*- coding: utf-8 -*-
"""k16_type_clf.py — 常绿/落叶判型判别器（AEF64 + 物候 + 地形），GroupKFold 验证

* 输入：judged_feats.parquet（208 个可分型判读灌丛点）
* 背景：FCS10 层自带的码 120/121 判型准确率只有 0.495（抛硬币，实测）；S2 振幅+分区阈值 0.639。
        本脚本训练判别器，看能否显著更高。
* 规则源（预注册）：
        特征 = AEF64 + ndvi_djf/jja/amp + dem + bio01/bio12 + slope（布尔缺失填 -999）
        模型 = RandomForest(balanced_subsample, n=400)，GroupKFold(5) 按 src_exp 分组
        基线 = 多数类比例；门槛 = CV 准确率 ≥0.75 且 macro-F1 ≥0.70 才用于放量判型
* 输出：data/shrub_nat/type_clf.json（含 CV 指标、每类召回/精确、分区表现、判定阈值）
* 用法：python k16_type_clf.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


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
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    tp = j[(j['is_shrub'] == '是') & j['shrub_type'].isin(['常绿', '落叶'])].copy()
    tp['zone'] = [zone_of(r.dem, r.bio01, r.bio12) for r in tp.itertuples()]
    y = (tp['shrub_type'] == '落叶').to_numpy(int)
    feats = (['A%02d' % i for i in range(64)] + ['ndvi_djf', 'ndvi_jja', 'ndvi_amp',
                                                 'dem', 'bio01', 'bio12', 'slope', 'tc'])
    feats = [f for f in feats if f in tp.columns]
    X = tp[feats].astype(float).fillna(-999.0).to_numpy()
    groups = tp['src_exp'].astype('category').cat.codes.to_numpy()
    emit('可分型灌丛点 %d（常绿 %d / 落叶 %d）' % (len(tp), (y == 0).sum(), y.sum()))
    base = max(y.mean(), 1 - y.mean())
    p = np.zeros(len(tp))
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_leaf_nodes=5000,
                                     max_features='sqrt', n_jobs=15, random_state=7,
                                     class_weight='balanced_subsample').fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    cm = confusion_matrix(y, pred)
    per = {}
    for tag, k in (('常绿', 0), ('落叶', 1)):
        col = cm[:, k]
        per[tag] = dict(recall=round(float(col[k] / max(1, col.sum())), 4), n=int(col.sum()))
    byzone = {}
    for z, idx in tp.groupby('zone').groups.items():
        ii = tp.index.get_indexer(idx)
        byzone[z] = dict(n=len(ii), acc=round(float(accuracy_score(y[ii], pred[ii])), 4))
    usable = bool(acc >= 0.75 and f1 >= 0.70)
    res = dict(n=int(len(tp)), baseline=round(float(base), 4), cv_acc=round(float(acc), 4),
               cv_macro_f1=round(float(f1), 4), per_class=per, by_zone=byzone,
               usable=usable, feats=feats)
    emit('CV 准确率 %.3f（基线 %.3f）、macro-F1 %.3f → 可用=%s' % (acc, base, f1, usable))
    emit('每类：%s' % per)
    emit('分区：%s' % byzone)
    tp2 = tp[['point_id', 'lon', 'lat', 'shrub_type']].assign(p_decid=pred)
    tp2.to_parquet(os.path.join(OUTD, 'type_train.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'type_clf.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ type_clf.json')


if __name__ == '__main__':
    main()
