# -*- coding: utf-8 -*-
"""
e3_train_zones.py — 全国 2° 分区模型训练 + 评估（不跑分类）
* 输入: cn_samples_v5.parquet（清洗后，含 64 维嵌入）
* 分区: 2°×2° 网格；训练集=本区+相邻8区样本（每格每类 ≤5000）
* 权重: gold_c=1.0 silver_c=0.7 bronze_c=0.5 external_c=0.6 uncovered_c=0.4 × (0.7+0.3·src_conf)
* 评估: 每 1° 格分组 80/20；输出分区 OA + 全国加权 OA + 30 类 UA/PA
* 产物: models_national/zone_{L}N{U}.joblib + trees JSON + training_report.json
* 用法: python e3_train_zones.py [max_zones]
"""
import sys, os, json, glob, time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V5 = '数据/本地处理/样本底座/cn_samples_v5.parquet'
OUT_DIR = '数据/本地处理/全国清洗训练/models_national'
os.makedirs(OUT_DIR, exist_ok=True)
FEATS = [f'A{i:02d}' for i in range(64)]
W_TIER = {'gold_c': 1.0, 'silver_c': 0.7, 'bronze_c': 0.5,
          'external_c': 0.6, 'uncovered_c': 0.4}
MAX_PER_CLASS_PER_ZONE = 5000

def main():
    max_zones = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    df = pd.read_parquet(V5)
    df = df[~df.clean_tier.isin(['removed', 'flagged'])].reset_index(drop=True)
    df['zone'] = (np.floor(df.lon / 2).astype(int) * 2).astype(str) + '_' + \
                 (np.floor(df.lat / 2).astype(int) * 2).astype(str)
    zones = sorted(df.zone.unique())
    if max_zones:
        zones = zones[:max_zones]
    print(f'清洗后总点: {len(df):,}  分区数: {len(zones)}', flush=True)

    all_rep = {}
    t0 = time.time()
    for zi, zone in enumerate(zones):
        zl, zu = zone.split('_')
        zl, zu = int(zl), int(zu)
        # 本区 + 8 邻区
        m = np.zeros(len(df), dtype=bool)
        for dz in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
            l, u = zl + dz[0], zu + dz[1]
            m |= (np.floor(df.lon / 2).astype(int) * 2 == l) & \
                 (np.floor(df.lat / 2).astype(int) * 2 == u)
        sub = df[m]
        if len(sub) < 1000:
            continue
        # 每类截断
        sub = sub.groupby('class_new', group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_PER_CLASS_PER_ZONE), random_state=42))
        X = sub[FEATS].to_numpy(np.float32)
        y = sub['class_new'].to_numpy(int)
        w = sub['tier'].map(W_TIER).fillna(0.5).to_numpy() * \
            (0.7 + 0.3 * sub['src_conf'].fillna(0.8).to_numpy())
        grp = (np.floor(sub.lon).astype(int) * 1000 + np.floor(sub.lat).astype(int)).to_numpy()
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            itr, iva = next(gss.split(X, y, groups=grp))
        except ValueError:
            itr = np.arange(len(X)); iva = itr[:max(1, len(X)//5)]
        rf = RandomForestClassifier(n_estimators=150, max_depth=18, max_features='sqrt',
                                    min_samples_leaf=3, n_jobs=6, random_state=42,
                                    class_weight='balanced_subsample')
        rf.fit(X[itr], y[itr], sample_weight=w[itr])
        rep = classification_report(y[iva], rf.predict(X[iva]),
                                    output_dict=True, zero_division=0)
        oa = rep['accuracy']
        all_rep[zone] = {'n': int(len(sub)), 'oa': round(oa, 4),
                         'kappa_proxy': oa}
        joblib.dump(rf, os.path.join(OUT_DIR, f'zone_{zone}.joblib'))
        # top-24 特征（供树串转换）
        imp = rf.feature_importances_
        top = sorted(np.argsort(imp)[::-1][:24].tolist())
        json.dump({'features': [FEATS[i] for i in top], 'classes': rf.classes_.tolist()},
                  open(os.path.join(OUT_DIR, f'zone_{zone}_cfg.json'), 'w'))
        if (zi + 1) % 10 == 0:
            print(f'  {zi+1}/{len(zones)} 区  累计 {time.time()-t0:.0f}s  最近OA={oa:.3f}', flush=True)
    # 全国汇总
    oas = [v['oa'] for v in all_rep.values() if v['oa'] > 0]
    ns = [v['n'] for v in all_rep.values()]
    wOA = float(np.average(oas, weights=ns)) if oas else 0
    summary = {'zones': len(all_rep), 'weighted_oa': round(wOA, 4),
               'min_oa': round(min(oas), 4) if oas else 0,
               'max_oa': round(max(oas), 4) if oas else 0,
               'detail': all_rep}
    json.dump(summary, open(os.path.join(OUT_DIR, 'training_report.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'\n分区数: {len(all_rep)}  全国加权 OA: {wOA:.4f} '
          f'(min {min(oas):.3f} / max {max(oas):.3f})', flush=True)
    print('模型目录:', OUT_DIR)

if __name__ == '__main__':
    main()
