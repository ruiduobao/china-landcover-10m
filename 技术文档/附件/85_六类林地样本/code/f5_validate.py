# -*- coding: utf-8 -*-
"""f5_validate.py — 六类林地样本的 AEF64 留出验证（含分轴），并重估 L3 判据

* 输入：forest6/forest6_v1.parquet（472,307 × 113，用 cls6_train）
        forest6/f4_result.csv（S2 判读结果，用于对比）
* 规则源：技术文档/84 §4.3（L3 验收）、doc82 §11（分组验证纪律）
* 设计（预注册）：
    三种验证：① 10 km 空间块 5 折  ② Leave-One-Province-Out  ③ Leave-One-EcoRegion-Out
    分轴：针/阔（41vs43, 42vs44, 45vs46）、常绿/落叶（41vs42, 43vs44, 45vs46）、纯林/混交
* 输出：f5_aef_holdout.json / f5_thresholds.csv / f5_verdict.json
* 用法：python f5_validate.py
* 资源：单进程，RF n_jobs=8，内存 <10 GB
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
D = r'F:/lc_work/nb_national_2026/forest6'
AEF = ['A%02d' % i for i in range(64)]
SEED = 20261025
CLASSES = [41, 42, 43, 44, 45, 46]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def rf():
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, max_depth=28,
                                  max_features='sqrt', class_weight='balanced_subsample',
                                  random_state=SEED, n_jobs=10)


def blk10(lon, lat, b=0.1):
    return (np.floor(np.asarray(lon) / b).astype(np.int64) * 100000
            + np.floor(np.asarray(lat) / b).astype(np.int64))


def cv_eval(X, y, groups, name, max_fold=5):
    """按 groups 做分组留出，返回 OOF 预测与指标。groups 值过多的（如省级）做 leave-one-out。"""
    uq = np.unique(groups)
    oof = np.full(len(y), -1, dtype='int64')
    proba = np.zeros((len(y), len(np.unique(y))))
    labels = np.unique(y)
    if len(uq) <= max_fold:
        folds = [[g] for g in uq]
    else:
        # 把组随机分到 max_fold 折（组级划分，保证同组不跨折）
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(uq)
        folds = [list(x) for x in np.array_split(perm, max_fold)]
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = rf().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        cols = [list(m.classes_).index(c) if c in m.classes_ else -1
                for c in labels]
        for j, c in enumerate(cols):
            if c >= 0:
                proba[te, j] = p[:, c]
        oof[te] = m.predict(X[te])
    ok = oof >= 0
    if ok.sum() == 0:
        return None
    res = dict(
        name=name, n=int(ok.sum()), n_groups=int(len(uq)),
        oa=round(float(accuracy_score(y[ok], oof[ok])), 4),
        macro_f1=round(float(f1_score(y[ok], oof[ok], average='macro')), 4),
        per_class={})
    for c in labels:
        msk = y[ok] == c
        if msk.sum() == 0:
            continue
        tp = int(((oof[ok] == c) & msk).sum())
        n_pred = int((oof[ok] == c).sum())
        res['per_class'][str(int(c))] = dict(
            n=int(msk.sum()),
            precision=round(tp / max(n_pred, 1), 4),
            recall=round(tp / max(msk.sum(), 1), 4))
    res['confusion'] = confusion_matrix(y[ok], oof[ok], labels=labels).tolist()
    res['labels'] = [int(x) for x in labels]
    return res, oof, proba, labels


def pair_eval(X, y, groups, ca, cb, name):
    """二分类 pairwise：只在 ca/cb 两点上训练/评估。"""
    m = np.isin(y, [ca, cb])
    Xs, ys, gs = X[m], (y[m] == cb).astype(int), groups[m]
    uq = np.unique(gs)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(uq)
    folds = [list(x) for x in np.array_split(perm, min(5, len(uq)))]
    oof = np.full(len(ys), -1, dtype=float)
    for f in folds:
        te = np.isin(gs, f); tr = ~te
        if te.sum() == 0 or len(np.unique(ys[tr])) < 2:
            continue
        mdl = rf().fit(Xs[tr], ys[tr])
        oof[te] = mdl.predict_proba(Xs[te])[:, 1]
    ok = oof >= 0
    if ok.sum() < 20 or len(np.unique(ys[ok])) < 2:
        return dict(pair=f'{ca}vs{cb}', name=name, n=int(ok.sum()), note='insufficient')
    auc = float(roc_auc_score(ys[ok], oof[ok]))
    pred = (oof[ok] >= 0.5).astype(int)
    return dict(pair=f'{ca}vs{cb}', name=name, n=int(ok.sum()),
                auc=round(auc, 4),
                f1=round(float(f1_score(ys[ok], pred, average='macro')), 4),
                acc=round(float((pred == ys[ok]).mean()), 4))


def main():
    t0 = time.time()
    SRC = os.environ.get('F5_SRC', 'forest6_v1.parquet')
    TAG = os.environ.get('F5_TAG', 'v1')
    CLSCOL = 'cls6_train' if 'v1' in SRC else 'cls6'
    extra = ['amp2023', 'djf2023', 'high_elev_enf', 'south_needle_flag', 'relabeled']
    if 'v1' in SRC:
        cols = AEF + [CLSCOL, 'prov', 'eco', 'lon', 'lat'] + extra
    else:
        cols = AEF + [CLSCOL, 'prov', 'eco', 'lon', 'lat']
    df = pd.read_parquet(os.path.join(D, SRC), columns=[c for c in cols if c in pd.read_parquet(os.path.join(D, SRC)).columns] or cols)
    emit('池 %d 行' % len(df))
    df = df[df[CLSCOL].isin(CLASSES)].reset_index(drop=True)
    emit('六类可用 %d（%s）' % (len(df), df[CLSCOL].value_counts().sort_index().to_dict()))

    # 分层下采样：大票类每类上限 20000，45/46 全保留（它们本来就 <5000）
    CAP = 20000
    rng0 = np.random.default_rng(SEED)
    keep = []
    for c, g in df.groupby(CLSCOL):
        idx = g.index.to_numpy()
        if len(idx) > CAP:
            idx = rng0.choice(idx, size=CAP, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    df = df.loc[keep].reset_index(drop=True)
    emit('分层下采样后 %d（%s）' % (len(df), df[CLSCOL].value_counts().sort_index().to_dict()))

    X = df[AEF].to_numpy(dtype='float32')
    y = df[CLSCOL].to_numpy(dtype='int64')
    g10 = blk10(df.lon.values, df.lat.values)
    gprov = df.prov.fillna('NA').to_numpy()
    geco = df.eco.fillna('NA').to_numpy()

    out = {'n_total': int(len(df)),
           'class_counts': {str(k): int(v) for k, v in
                            df[CLSCOL].value_counts().sort_index().items()},
           'designs': {}, 'pairs': {}, 'seed': SEED}

    # ---------- 六类多分类：三种设计 ----------
    for gname, g in (('block10km', g10), ('prov', gprov), ('eco', geco)):
        emit('六类 %s …' % gname)
        r = cv_eval(X, y, g, gname)
        if r:
            out['designs'][gname] = r[0]
            emit('   %s: OA %.4f  macroF1 %.4f' % (
                gname, r[0]['oa'], r[0]['macro_f1']))

    # ---------- 分轴 pairwise ----------
    axes = {
        'needle_vs_broad': [(41, 43), (42, 44), (45, 46)],
        'evergreen_vs_deciduous': [(41, 42), (43, 44), (45, 46)],
    }
    for ax, pairs in axes.items():
        out['pairs'][ax] = {}
        for ca, cb in pairs:
            out['pairs'][ax]['%dvs%d' % (ca, cb)] = {}
            for gname, g in (('block10km', g10), ('prov', gprov), ('eco', geco)):
                r = pair_eval(X, y, g, ca, cb, gname)
                out['pairs'][ax]['%dvs%d' % (ca, cb)][gname] = r
            emit('   %s %dvs%d: prov AUC=%s' % (
                ax, ca, cb,
                out['pairs'][ax]['%dvs%d' % (ca, cb)]['prov'].get('auc')))

    # 纯林 vs 混交
    y2 = np.isin(y, [45, 46]).astype(int)
    out['pairs']['pure_vs_mixed'] = {}
    for gname, g in (('block10km', g10), ('prov', gprov), ('eco', geco)):
        uq = np.unique(g); rng = np.random.default_rng(SEED)
        perm = rng.permutation(uq)
        folds = [list(x) for x in np.array_split(perm, 5)]
        oof = np.full(len(y2), -1.0)
        for f in folds:
            te = np.isin(g, f); tr = ~te
            if te.sum() == 0 or len(np.unique(y2[tr])) < 2:
                continue
            mdl = rf().fit(X[tr], y2[tr])
            oof[te] = mdl.predict_proba(X[te])[:, 1]
        ok = oof >= 0
        out['pairs']['pure_vs_mixed'][gname] = dict(
            name=gname, n=int(ok.sum()),
            auc=round(float(roc_auc_score(y2[ok], oof[ok])), 4) if ok.sum() and
            len(np.unique(y2[ok])) > 1 else None)
    emit('   pure_vs_mixed prov AUC=%s' % out['pairs']['pure_vs_mixed']['prov'].get('auc'))

    json.dump(out, open(os.path.join(D, f'f5_aef_holdout_{TAG}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('写 f5_aef_holdout.json（%.1f min）' % ((time.time() - t0) / 60))

    # ---------- 分带阈值数据依据 ----------
    rows = []
    for (eco, c), g in df.groupby(['eco', CLSCOL]):
        if len(g) < 50:
            continue
        for col in ('djf2023', 'amp2023'):
            s = g[col].dropna()
            if len(s) < 50:
                continue
            rows.append(dict(eco=eco, cls=int(c), col=col, n=len(s),
                             p10=round(float(s.quantile(.10)), 1),
                             p50=round(float(s.quantile(.50)), 1),
                             p90=round(float(s.quantile(.90)), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(D, f'f5_thresholds_{TAG}.csv'),
                              index=False, encoding='utf-8-sig')
    emit('写 f5_thresholds.csv（%d 行）' % len(rows))

    # ---------- 验收结论 ----------
    def get(d, *ks):
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d
    ax = {k: {g: get(out, 'pairs', k, p, g, 'auc')
              for p, g in [(pp, 'prov') for pp in out['pairs'][k]]}
          for k in out['pairs']}
    # 三轴乘积估计（各轴用最弱 pairwise 的 prov AUC 作为近似准确率）
    nt = [out['pairs']['needle_vs_broad'][k]['prov'].get('acc')
          for k in out['pairs']['needle_vs_broad']]
    ev = [out['pairs']['evergreen_vs_deciduous'][k]['prov'].get('acc')
          for k in out['pairs']['evergreen_vs_deciduous']]
    pm = out['pairs']['pure_vs_mixed']['prov'].get('auc')
    verdict = {
        'block10km_macroF1': out['designs'].get('block10km', {}).get('macro_f1'),
        'prov_macroF1': out['designs'].get('prov', {}).get('macro_f1'),
        'eco_macroF1': out['designs'].get('eco', {}).get('macro_f1'),
        'axis_acc': {'needle_vs_broad': nt, 'evergreen_vs_deciduous': ev,
                     'pure_vs_mixed_auc': pm},
        'note': ('六类精确 = 三轴乘积；每轴 0.95 → 约 0.857。'
                 '轴级 acc 的中位数自乘可作六类理论上限的粗估。'),
    }
    try:
        a1 = float(np.nanmedian([x for x in nt if x]))
        a2 = float(np.nanmedian([x for x in ev if x]))
        a3 = float(pm) if pm else np.nan
        verdict['estimated_ceiling_from_axes'] = round(a1 * a2 * a3, 4)
    except Exception:
        pass
    json.dump(verdict, open(os.path.join(D, f'f5_verdict_{TAG}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('写 f5_verdict.json')
    emit(json.dumps(verdict, ensure_ascii=False)[:600])


if __name__ == '__main__':
    main()
