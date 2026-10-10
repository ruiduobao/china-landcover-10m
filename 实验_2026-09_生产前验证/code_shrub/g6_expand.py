# -*- coding: utf-8 -*-
"""g6_expand.py — 留一省交叉验证（跨区泛化）+ 用判别器扩增 T2 候选 + 常绿/落叶赋值

* 输入：data/shrub/{judged_all.csv, cand_all.csv}
* 规则源（预注册）：
  E1 留一省验证（LOPO）：以 3 省训练、第 4 省测试 → 报告每省 AUC（测"跨区泛化"）
  E2 扩增：若 LOPO 平均 AUC ≥0.80 → 对候选池按 灌丛概率 ≥0.85 取 T2；否则只用判读确认点
  E3 常绿/落叶：ndvi_amp <0.15 → 常绿；>0.30 → 落叶；中间 → 待判
* 门槛：T2 每省上限 150；T2 点须与已确认点/彼此保持 ≥150 m 间距
* 输出：data/shrub/sample_v1.csv（三档：T1判读确认 / T2判别器扩增 / 待判保留）
        results/d2/shrub_lopo.json
* 用法：python g6_expand.py
"""
import collections, json, os, sys, time
sys.path.insert(0, os.path.join(WORK0 := r'F:/lc_work/v31_exp', 'code'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import v31_common as VC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

W = WORK0
OUTD = os.path.join(W, 'data', 'shrub')
SEED = 7
PROVS = ['宁夏', '四川', '黑龙江', '福建']
T2_PROB = 0.85
T2_MAX = 150
T2_MIN_M = 150.0

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    df = pd.read_csv(os.path.join(OUTD, 'judged_all.csv'), encoding='utf-8-sig')
    cand = pd.read_csv(os.path.join(OUTD, 'cand_all.csv'), encoding='utf-8-sig')
    miss = [c for c in VC.FEATS if c not in df.columns]
    if miss:
        df = df.merge(cand[['point_id'] + miss], on='point_id', how='left')
    j = df[df.Q1_shrub.isin(['是', '否'])].copy()
    j['y'] = (j.Q1_shrub == '是').astype(int)
    # --- E1 留一省 ---
    lopo = {}
    for p in PROVS:
        te = j[j.prov == p]; tr = j[j.prov != p]
        if len(te) == 0 or tr.y.nunique() < 2 or te.y.nunique() < 2:
            lopo[p] = dict(n_test=int(len(te)), note='样本不足/单类')
            continue
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                     n_jobs=15, random_state=SEED).fit(
            np.nan_to_num(tr[VC.FEATS].to_numpy('float32'), nan=-999), tr.y)
        pr = clf.predict_proba(np.nan_to_num(te[VC.FEATS].to_numpy('float32'), nan=-999))[:, 1]
        lopo[p] = dict(n_test=int(len(te)), n_pos=int(te.y.sum()), AUC=round(float(roc_auc_score(te.y, pr)), 4))
        emit('LOPO %s：n=%d 正=%d AUC=%.3f' % (p, len(te), int(te.y.sum()), lopo[p]['AUC']))
    aucs = [v['AUC'] for v in lopo.values() if 'AUC' in v]
    mean_auc = round(float(np.mean(aucs)), 4) if aucs else None
    ok = mean_auc is not None and mean_auc >= 0.80
    emit('LOPO 平均 AUC=%s → 扩增%s' % (mean_auc, '启用' if ok else '停用'))
    # --- E2 全量模型 + 扩增 ---
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                n_jobs=15, random_state=SEED).fit(
        np.nan_to_num(j[VC.FEATS].to_numpy('float32'), nan=-999), j.y)
    Xc = np.nan_to_num(cand[VC.FEATS].to_numpy('float32'), nan=-999)
    cand['p_shrub'] = clf.predict_proba(Xc)[:, 1]
    cand['ndvi_amp_f'] = pd.to_numeric(cand.get('ndvi_amp'), errors='coerce')
    def cls_veg(a):
        if pd.isna(a): return '待判'
        return '常绿' if a < 0.15 else ('落叶' if a > 0.30 else '待判')
    cand['shrub_type'] = cand['ndvi_amp_f'].map(cls_veg)
    # T1 = 判读确认
    t1 = df[df.Q1_shrub == '是'].copy()
    t1['tier'] = 'T1_判读确认'
    t1['shrub_type'] = t1['Q2_evergreen'].map(lambda s: s if s in ('常绿', '落叶') else '待判')
    t1['p_shrub'] = np.nan
    # T2 = 判别器扩增（排除已判读点；每省按概率降序取前 N，间距 ≥150 m）
    if ok:
        picked = []
        judged_ids = set(df.point_id)
        for p in PROVS:
            sub = cand[(cand.prov == p) & (~cand.point_id.isin(judged_ids)) & (cand.p_shrub >= T2_PROB)]
            sub = sub.sort_values('p_shrub', ascending=False)
            used, n = [], 0
            for r in sub.itertuples():
                if n >= T2_MAX: break
                if any(((r.lon - u) ** 2 * 100 + (r.lat - v) ** 2 * 120) < (T2_MIN_M / 111000.0) ** 2 for u, v in used):
                    continue
                used.append((r.lon, r.lat))
                picked.append(r)
                n += 1
            emit('T2 %s：候选 %d → 取 %d' % (p, len(sub), n))
        t2 = pd.DataFrame(picked)
        if len(t2):
            t2['tier'] = 'T2_判别器扩增'
            t2['Q2_evergreen'] = t2['shrub_type']
            t2['conf'] = '中'
            t2['note'] = t2['p_shrub'].map(lambda v: '判别器概率%.2f' % v)
    else:
        t2 = pd.DataFrame()
    cols = ['point_id', 'prov', 'stratum', 'lon', 'lat', 'tier', 'shrub_type', 'Q2_evergreen',
            'conf', 'p_shrub', 'dw_prob', 'ndvi_amp', 'dem', 'note']
    out = pd.concat([t1.reindex(columns=cols), t2.reindex(columns=cols)], ignore_index=True)
    out.to_csv(os.path.join(OUTD, 'sample_v1.csv'), index=False, encoding='utf-8-sig')
    VC.jsave(dict(lopo=lopo, mean_auc=mean_auc, expand=ok, n_T1=int(len(t1)), n_T2=int(len(t2)),
                  t2_prob=T2_PROB,
                  by_prov_tier={('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'tier']).size().items()},
                  by_prov_type={('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'shrub_type']).size().items()}),
             os.path.join(W, 'results', 'd2', 'shrub_lopo.json'))
    emit('样本 v1：T1 %d + T2 %d = %d' % (len(t1), len(t2), len(out)))
    emit('  省×档：%s' % {('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'tier']).size().items()})
    emit('  省×型：%s' % {('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'shrub_type']).size().items()})

if __name__ == '__main__': main()
