# -*- coding: utf-8 -*-
"""
v3_final_eval.py — v-final 模型规格：8 年全量复评（含逐类 + FCS10/非FCS10 分层）
v-final 规格（据 2026-09-14 诊断实验）：
  主模型   RF(150, max_depth=None, min_samples_leaf=2, max_features='sqrt',
             class_weight='balanced_subsample') —— 保住总体 OA（全深度树对大类最优）
  弱类模型 RF(120, max_depth=15, ...) —— 全深度树会让稀有类只在小叶里得票，
             150 棵树多数表决时被淹没；浅树把票摊开，实测 30/30 类非零
  合成规则 若弱类模型判为 5 个弱类之一、且主模型判为其**合理宿主类**之一 → 取弱类标签
             （宿主集来自 2022 混淆矩阵实测，见 HOST）
输出: 数据/本地处理/全国清洗训练/评估_vfinal/{per_class_overall.csv, per_class_<y>.csv,
      confusion_<y>.csv, vfinal_report.json, val_predictions_<y>.parquet}
用法: python v3_final_eval.py [--years 2022 2024] [--trees 150] [--weak-trees 120]
     长任务用 _tools/e11_years_driver.py 逐年跑（单年直跑会覆盖汇总报告）
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = os.path.join(E6.BASE, '评估_vfinal')
WEAK = [52, 62, 91, 140, 11]
# 宿主集：主模型把某弱类像素误判成的类（2022 混淆矩阵实测，取覆盖≥2 例的）
HOST = {
    52:  [51, 61, 71, 72, 120, 121, 62, 92],
    62:  [61, 51, 71, 82, 121, 52, 120],
    91:  [71, 61, 92, 82, 51, 72, 81],
    140: [150, 201, 121, 130, 120, 220, 201],
    11:  [10, 12, 120, 121, 130, 71, 51],
}
NJ = 13
T0 = 0.0
WEAK_DEPTH = 20          # 弱类模型树深（实测拐点：20 保 OA、15 保召回）


def fit(X, y, w, depth, trees, seed):
    rf = RandomForestClassifier(n_estimators=trees, n_jobs=NJ, random_state=seed,
                                max_depth=depth, min_samples_leaf=2, max_features='sqrt',
                                class_weight='balanced_subsample')
    rf.fit(X, y, sample_weight=w)
    return rf


def metrics_multi(ct, pred, labels_all=None):
    labs = sorted(set(ct) | set(pred)) if labels_all is None else labels_all
    ix = {c: i for i, c in enumerate(labs)}
    cm = np.zeros((len(labs), len(labs)), np.int64)
    for a, b in zip(ct, pred):
        cm[ix[a], ix[b]] += 1
    return cm, labs, E6.metrics_from_cm(cm, labs)


def run_year(y, val, hold, trees, weak_trees):
    fp = os.path.join(E6.SUB_DIR, f'r7_train_{y}.parquet')
    if not os.path.isfile(fp):
        return None, f'缺 {os.path.basename(fp)}'
    df = pd.read_parquet(fp)
    bk = E6.albers_bk(df.lon.to_numpy(), df.lat.to_numpy())
    tr = df[~np.isin(bk, hold)]
    X = tr[E6.FEATS].to_numpy(np.float32)
    ytr = tr.class_new.to_numpy(int)
    fin = np.isfinite(X).all(1)
    if not fin.all():
        X, ytr, tr = X[fin], ytr[fin], tr[fin]
    w = np.nan_to_num(tr.train_weight.to_numpy(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
    print(f'y{y}: 训练 {len(tr):,}', flush=True)

    t0 = time.time()
    main = fit(X, ytr, w, None, trees, 42)
    weak = fit(X, ytr, w, WEAK_DEPTH, weak_trees, 77)
    print(f'  两模型训练完成 {time.time()-t0:.0f}s', flush=True)

    v = val.merge(df[['row_id'] + E6.FEATS], on='row_id', how='inner')
    VX = v[E6.FEATS].to_numpy(np.float32)
    pm = main.predict(VX)
    pw = weak.predict(VX)
    fin_pred = pm.copy()
    n_ovr = {}
    for c in WEAK:
        sel = (pw == c) & np.isin(pm, HOST[c])
        n_ovr[c] = int(sel.sum())
        fin_pred[sel] = c
    print(f'  弱类覆盖: {n_ovr}', flush=True)
    ct = v.class_new.to_numpy(int)
    return (fin_pred, ct, v, n_ovr), None


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=list(range(2017, 2025)))
    ap.add_argument('--trees', type=int, default=150)
    ap.add_argument('--weak-trees', type=int, default=120)
    ap.add_argument('--weak-depth', type=int, default=0)
    a = ap.parse_args()
    if a.weak_depth:
        globals()['WEAK_DEPTH'] = a.weak_depth
    os.makedirs(OUT_DIR, exist_ok=True)
    hold = np.asarray(json.load(open(E6.HOLDOUT))['block_keys'], dtype=np.int64)
    val = E6.load_val_ids()
    print(f'验证池 {len(val):,} | holdout 块 {len(hold):,}', flush=True)
    rep = {'time': time.strftime('%Y-%m-%d %H:%M'), 'spec': {
        'main': f'RF({a.trees}, depth=None, leaf=2, sqrt, balanced_subsample)',
        'weak': f'RF({a.weak_trees}, depth=15, leaf=2, sqrt, balanced_subsample)',
        'weak_classes': WEAK, 'host_sets': HOST}, 'by_year': {}, 'by_source': {}}
    preds = []
    for y in a.years:
        r, err = run_year(y, val, hold, a.trees, a.weak_trees)
        if err:
            print(' ', err, flush=True); rep['by_year'][str(y)] = {'error': err}; continue
        pred, ct, v, n_ovr = r
        cm, labs, (met, per) = metrics_multi(ct, pred)
        met['n_override'] = n_ovr
        rep['by_year'][str(y)] = met
        per.to_csv(os.path.join(OUT_DIR, f'per_class_{y}.csv'), index=False)
        pd.DataFrame(cm, index=[f'true_{c}' for c in labs],
                     columns=[f'pred_{c}' for c in labs]).to_csv(
            os.path.join(OUT_DIR, f'confusion_{y}.csv'))
        preds.append(v.assign(pred=pred, eval_year=y))
        preds[-1][['row_id', 'eval_year', 'class_new', 'pred', 'src', 'zone']].to_parquet(
            os.path.join(OUT_DIR, f'val_predictions_{y}.parquet'), index=False)
        nz = int((per.n_pred > 0).sum())
        print(f'  y{y}: OA={met["OA"]:.4f} macro-F1={met["macro_F1"]:.4f} '
              f'bal-acc={met["balanced_accuracy"]:.4f} 非零类={nz}/{len(per)} n={met["n"]}', flush=True)

    if preds:
        P = pd.concat(preds, ignore_index=True)
        P.to_parquet(os.path.join(OUT_DIR, 'val_predictions.parquet'), index=False)
        cm, labs, (met, per) = metrics_multi(P.class_new.to_numpy(int), P.pred.to_numpy(int))
        rep['overall'] = met
        per.to_csv(os.path.join(OUT_DIR, 'per_class_overall.csv'), index=False)
        print(f"\n汇总 OA={met['OA']:.4f} macro-F1={met['macro_F1']:.4f} "
              f"bal-acc={met['balanced_accuracy']:.4f} 非零类={int((per.n_pred>0).sum())}/{len(per)}")
        is_f = P.src.astype(str).str.lower().str.contains('fcs10').to_numpy()
        for tag, mask in (('fcs10', is_f), ('non_fcs10', ~is_f)):
            if int(mask.sum()) == 0:
                continue
            g = P[mask]
            _, _, (m_, _) = metrics_multi(g.class_new.to_numpy(int), g.pred.to_numpy(int))
            rep['by_source'][tag] = m_
            print(f"  [来源 {tag}] OA={m_['OA']:.4f} macro-F1={m_['macro_F1']:.4f} n={m_['n']}")
    rep['elapsed_min'] = round((time.time() - T0) / 60, 1)
    json.dump(rep, open(os.path.join(OUT_DIR, 'vfinal_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print('\n输出目录:', OUT_DIR)


if __name__ == '__main__':
    main()
