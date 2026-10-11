# -*- coding: utf-8 -*-
"""k13_inframe_lopo.py — FCS10 框内的留一省验证 + 灵敏/特异度 + 按省预测放量精度

* 输入：judged_feats.parquet（带 stratum57）+ judged_all.csv（province）
* 规则源（预注册）：
        LOPO = 留一省（4 省轮流做测试集），测 AUC 与 θ=0.6/0.7 的 precision/召回
        全量 fit 的 sens/spec 在 θ 档上计算，用于**按省预测**：
              precision(省) = sens·r / (sens·r + (1-spec)·(1-r))
              其中 r＝该省 FCS10 层实测真灌丛率（宁 0.25/川 0.50/黑 0.54/闽 0.57，
              全国缺测则用 0.460 作保守值，并在报告里注明这是**外推**）
* 输出：data/shrub_nat/inframe_lopo.json + data/shrub_nat/inframe_lopo.md
* 用法：python k13_inframe_lopo.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PROV_BASE = {'宁夏': 0.25, '四川': 0.50, '黑龙江': 0.54, '福建': 0.57}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def fit(X, y, sd=7):
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_leaf_nodes=5000,
                                  max_features='sqrt', n_jobs=15, random_state=sd,
                                  class_weight='balanced_subsample').fit(X, y)


def main():
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    kj = list(zip(np.round(j.lon.to_numpy(float), 6), np.round(j.lat.to_numpy(float), 6)))
    k57 = list(zip(np.round(d57.lon.to_numpy(float), 6), np.round(d57.lat.to_numpy(float), 6)))
    m57 = dict(zip(k57, zip(d57['stratum'], d57['Q1_shrub'], d57['prov'])))
    j['st'] = [m57.get(k, ('', '', ''))[0] for k in kj]
    j['prov57'] = [m57.get(k, ('', '', ''))[2] for k in kj]
    e = j[(j['st'] == 'FCS10') & j['is_shrub'].isin(['是', '否'])].copy()
    y = e['is_shrub'].to_numpy() == '是'
    X = e[['A%02d' % i for i in range(64)]].astype(float).fillna(-999.0).to_numpy()
    prov = e['prov57'].to_numpy()
    emit('FCS10 框 n=%d（灌丛 %d）；省份分布 %s' % (len(e), y.sum(), dict(pd.Series(prov).value_counts())))

    res = dict(lopo={}, sens_spec={}, prov_pred={})
    lines = ['# FCS10 框内留一省验证 + 按省放量精度预测 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 判读点 %d（灌丛 %d）；模型 RF(AEF64)；阈值 θ=0.6/0.7', '',
             '## 留一省（LOPO）', '', '| 测试省 | n | 灌丛 | AUC | θ=0.6 精度 | θ=0.7 精度 |', '|---|---|---|---|---|---|']
    for pv in sorted(set(prov)):
        te = prov == pv
        tr = ~te
        if y[te].sum() < 3 or te.sum() < 8:
            emit('  %s 测试点太少（%d），跳过' % (pv, te.sum()))
            continue
        clf = fit(X[tr], y[tr].astype(int))
        p = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te].astype(int), p) if len(set(y[te])) > 1 else float('nan')
        row = []
        for th in (0.6, 0.7):
            m = p >= th
            row.append('%d/%d=%.2f' % (y[te][m].sum(), m.sum(), y[te][m].mean() if m.sum() else float('nan')))
        res['lopo'][pv] = dict(n=int(te.sum()), shrub=int(y[te].sum()), auc=round(float(auc), 4),
                               th06=row[0], th07=row[1])
        lines.append('| %s | %d | %d | %.3f | %s | %s |' % (pv, te.sum(), y[te].sum(), auc, row[0], row[1]))
        emit('  LOPO %s：AUC=%.3f θ0.6=%s θ0.7=%s' % (pv, auc, row[0], row[1]))

    # 全量 fit 的 sens/spec（用 CV 概率更稳：这里用 k12 的 cv 概率）
    cv = pd.read_parquet(os.path.join(OUTD, 'inframe_scored.parquet'))
    key = {(round(a, 6), round(b, 6)): c for a, b, c in
           zip(cv.lon.to_numpy(float), cv.lat.to_numpy(float), cv.p_cv.to_numpy(float))}
    pcv = np.array([key.get((round(a, 6), round(b, 6)), np.nan) for a, b in
                    zip(e.lon.to_numpy(float), e.lat.to_numpy(float))])
    ok = np.isfinite(pcv)
    lines += ['', '## 用 CV 概率算的灵敏/特异度与按省预测', '',
              '| θ | 敏感度 | 特异度 | 预测精度(基线 r=0.46) | 宁 0.25 | 川 0.50 | 黑 0.54 | 闽 0.57 |',
              '|---|---|---|---|---|---|---|---|']
    for th in (0.5, 0.6, 0.7):
        m = ok & (pcv >= th)
        yy = y[ok]
        # sens = P(通过|真灌丛)；spec = P(不通过|非灌丛)（此前误写成精度/阴性预测值，2026-10-11 修正）
        pass_m = ok & (pcv >= th)
        sens = float(y[pass_m].sum()) / float(y.sum())          # 召回
        neg = ok & (~y)
        spec = float((pcv[neg] < th).sum()) / float(neg.sum())   # 特异度
        emp = float(y[pass_m].sum()) / float(pass_m.sum())       # 87 点上的实测精度（核对用）
        preds = {}
        for r_name, r in [('r46', 0.46)] + list(PROV_BASE.items()):
            pr = sens * r / (sens * r + (1 - spec) * (1 - r)) if (sens * r + (1 - spec) * (1 - r)) else 0
            preds[r_name] = round(pr, 4)
        res['sens_spec'][str(th)] = dict(sens=round(sens, 4), spec=round(spec, 4),
                                          emp_precision=round(emp, 4) if pass_m.sum() else None, pred=preds)
        lines.append('| %.1f | %.3f | %.3f | %.3f | %s | %s | %s | %s |' % (
            th, sens, spec, preds['r46'],
            *[('%.3f' % preds[k]) for k in PROV_BASE]))
        emit('θ=%.1f sens=%.3f spec=%.3f → 预测全国(r=.46)=%.3f，宁=%.3f，闽=%.3f' % (
            th, sens, spec, preds['r46'], preds['宁夏'], preds['福建']))
    lines += ['', '> 注：按省预测是把四省的实测真灌丛率当作该省的"该层先验"，再用框内测得的 sens/spec 推算；',
              '> 全国没有实测的层先验，故用四省合并 0.460 作保守值——**全国真实精度可能低于该数**，',
              '> 尤其是西北（宁夏只 0.25）。各省实测要靠判读审计包补齐。']
    json.dump(res, open(os.path.join(OUTD, 'inframe_lopo.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'inframe_lopo.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ inframe_lopo.json / inframe_lopo.md')


if __name__ == '__main__':
    main()
