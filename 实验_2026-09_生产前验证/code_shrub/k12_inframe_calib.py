# -*- coding: utf-8 -*-
"""k12_inframe_calib.py — 在 FCS10 框内部量 AEF 的可分性（该层随机抽样 87 点上）

* 为什么单独测：混合判读池里 87% 是森林/湿地等"容易的负样本"，其分数校准只反映那个池子的难度；
        而 FCS10 灌丛层是真灌丛率 0.46 的"困难框"，要放量就必须知道**框内**可分性。
        doc57 的 FCS10 分层点（87 判读点）就是该层的随机抽样 → 其框内性能可直接迁移到全国该层。
* 输入：judged_feats.parquet（带 stratum57 标注）
* 规则源（预注册）：
        训练/评估 = stratum57=='FCS10' 的可判读点（n≈87，灌丛≈40）
        交叉验证 = StratifiedKFold(5) × 3 种子（7/11/23），报 mean±sd；另做 1,000 次 bootstrap 给 CI
        特征 = AEF64；模型 = RandomForest（balanced_subsample，n=400，min_samples_leaf=2）
        校准 = P(灌丛 | score≥θ)，逐 θ 报 precision / 召回 / Wilson CI
* 门槛：若 precision≥0.65 且召回≥0.15 的档存在，则记 chosen_theta，作为全国放量阈值
* 输出：data/shrub_nat/inframe_calib.json + data/shrub_nat/inframe_calib.md
* 用法：python k12_inframe_calib.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
SEEDS = (7, 11, 23)


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
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    st57 = d57['stratum'].to_numpy()
    kj = list(zip(np.round(j.lon.to_numpy(float), 6), np.round(j.lat.to_numpy(float), 6)))
    k57 = list(zip(np.round(d57.lon.to_numpy(float), 6), np.round(d57.lat.to_numpy(float), 6)))
    map57 = dict(zip(k57, st57))
    j['stratum57'] = [map57.get(k, '') for k in kj]
    e = j[(j['stratum57'] == 'FCS10') & j['is_shrub'].isin(['是', '否'])].copy()
    y = (e['is_shrub'] == '是').to_numpy(int)
    X = e[['A%02d' % i for i in range(64)]].astype(float).fillna(-999.0).to_numpy()
    emit('FCS10 框判读点 n=%d（灌丛 %d，非灌丛 %d）' % (len(e), y.sum(), (1 - y).sum()))
    ps, aucs = [], []
    for sd in SEEDS:
        p = np.zeros(len(e))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
        for tr, te in skf.split(X, y):
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_leaf_nodes=5000,
                                         max_features='sqrt', n_jobs=15, random_state=sd,
                                         class_weight='balanced_subsample').fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        ps.append(p)
        aucs.append(roc_auc_score(y, p))
    P = np.vstack(ps)                      # [n_seed, n]
    pm = P.mean(axis=0)
    emit('框内 CV AUC = %s（mean %.4f）' % ([round(a, 4) for a in aucs], float(np.mean(aucs))))
    lines = ['# FCS10 灌丛层框内可分性（87 点随机抽样）%s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 样本：doc57 的 FCS10 分层判读点 n=%d（灌丛 %d，非灌丛 %d），**是该层的随机抽样**' % (
                 len(e), y.sum(), (1 - y).sum()),
             '- 模型：RF(AEF64, balanced)，StratifiedKFold(5) × 3 种子；AUC = %.4f±%.4f' % (
                 float(np.mean(aucs)), float(np.std(aucs))),
             '- 别忘基线：**不筛直接用的 precision 就是 0.460**（该层实测真灌丛率）',
             '', '| 阈值θ | 通过 | 灌丛 | precision | 95%CI | 召回 | 通过率(占层) |', '|---|---|---|---|---|---|---|']
    res = dict(n=int(len(e)), n_pos=int(y.sum()), auc_mean=round(float(np.mean(aucs)), 4),
               auc_sd=round(float(np.std(aucs)), 4), base=round(float(y.mean()), 4), theta={})
    best = None
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = pm >= th
        n, k = int(m.sum()), int(y[m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        rec = k / max(1, int(y.sum()))
        passrate = n / len(e)
        res['theta'][str(th)] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                     wilson=[lo, hi], recall=round(rec, 4), pass_rate=round(passrate, 4))
        lines.append('| %.1f | %d | %d | %s | [%.2f,%.2f] | %.3f | %.1f%% |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi, rec, 100 * passrate))
        emit('θ=%.1f：n=%d 灌=%d precision=%s 召回=%.3f 通过率%.1f%%' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', rec, 100 * passrate))
        if prec is not None and prec >= 0.65 and rec >= 0.15 and n >= 8:
            best = th
    res['chosen_theta'] = best
    # bootstrap 估计 chosen 档的 precision 分布
    if best is not None:
        m = pm >= best
        idx = np.where(m)[0]
        boots = []
        rng = np.random.RandomState(0)
        for _ in range(2000):
            s = rng.choice(idx, len(idx), replace=True)
            boots.append(y[s].mean())
        res['boot_ci'] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        lines += ['', '## 预注册门槛内选定 θ = **%s**' % best,
                  '该档 precision 的 bootstrap 95%% 区间 = [%.3f, %.3f]（n=%d，样本小，区间宽是诚实的）' % (
                      res['boot_ci'][0], res['boot_ci'][1], len(idx)),
                  '',
                  '若按该档对全国 FCS10 灌丛层 537,361 点放量，预计保留 %.0f%% 的点' % (100 * len(idx) / len(e))]
    e[['point_id', 'lon', 'lat', 'is_shrub']].assign(p_cv=pm).to_parquet(
        os.path.join(OUTD, 'inframe_scored.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'inframe_calib.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'inframe_calib.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ inframe_calib.json / inframe_calib.md（chosen θ=%s）' % best)


if __name__ == '__main__':
    main()
