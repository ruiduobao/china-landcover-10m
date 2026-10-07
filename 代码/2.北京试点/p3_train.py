# -*- coding: utf-8 -*-
"""
p3_train.py — 北京试点：本地训练 RF + top-M 特征选择 + 转 decisionTreeEnsemble 字符串 + G2 一致性校验
* 输入: 数据/本地处理/北京试点/samples_bj_train.parquet（gold+silver, 2020 嵌入 64 维）
* 输出: 数据/本地处理/北京试点/models_pilot/{rf_bj.joblib, trees_bj.json, p3_report.json}
* G2 门禁: sklearn 与 GEE decisionTreeEnsemble 预测不一致率 <= 0.1%
* 树字符串格式（SERVIR/rpart 风格，已实证）:
    1) root 9999 9999 9999
    2) {feat}<{thr} 9999 9999 {label|9999} [*]   —— 左子(2n) 条件为 <, 右子(2n+1) 为 >=
"""
import sys, os, json, io
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

OUT_DIR = PC.OUT_DIR
MODEL_DIR = os.path.join(OUT_DIR, 'models_pilot')
os.makedirs(MODEL_DIR, exist_ok=True)

tr = pd.read_parquet(os.path.join(OUT_DIR, 'samples_bj_train.parquet'))
FEATS = PC.FEATS_ALL
if 'lon' not in tr.columns:
    # 旧版路径：样本文件无坐标，经 v2 底座桥接
    BASE = pd.read_parquet('数据/本地处理/样本底座/cn_samples_v2.parquet')
    bj = BASE[(BASE.lon >= PC.BBOX[0]) & (BASE.lon <= PC.BBOX[2]) &
              (BASE.lat >= PC.BBOX[1]) & (BASE.lat <= PC.BBOX[3])].reset_index(drop=True)
    tr = tr.merge(bj[['lon', 'lat']].reset_index().rename(columns={'index': 'row_id'}),
                  on='row_id', how='left')
X = tr[FEATS].to_numpy(float)
y = tr['cls'].to_numpy(int)
# 分层权重（与 e3 一致；旧 gold/silver 两层保持原行为）
W_TIER = {'gold': 1.0, 'silver': 0.7, 'bronze': 0.5, 'external': 0.6, 'uncovered': 0.4}
w = tr['tier'].map(W_TIER).fillna(0.5).to_numpy() if 'tier' in tr.columns else \
    np.where(tr['tier'] == 'gold', 1.0, 0.5)
grp = (np.floor(tr['lon']).astype(int) * 1000 + np.floor(tr['lat']).astype(int)).to_numpy()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
itr, iva = next(gss.split(X, y, groups=grp))
Xtr, ytr, wtr = X[itr], y[itr], w[itr]

print('=== 训练（全 64 维）===')
N_TREES = int(PC.CFG.get('n_trees', 150))
rf64 = RandomForestClassifier(n_estimators=N_TREES, max_depth=15, max_features='sqrt',
                              min_samples_leaf=3, n_jobs=8, random_state=42,
                              class_weight='balanced_subsample')
rf64.fit(Xtr, ytr, sample_weight=wtr)
rep64 = classification_report(y[iva], rf64.predict(X[iva]), output_dict=True, zero_division=0)
print('留出 OA(64维):', round(rep64['accuracy'], 4))

imp = rf64.feature_importances_
top = np.argsort(imp)[::-1][:24]
FEATS_M = [FEATS[i] for i in sorted(top)]

print(f'=== 重训（top-24, {N_TREES} 树）===')
rf = RandomForestClassifier(n_estimators=N_TREES, max_depth=15, max_features='sqrt',
                            min_samples_leaf=3, n_jobs=8, random_state=42,
                            class_weight='balanced_subsample')
rf.fit(X[:, sorted(top)][itr], y[itr], sample_weight=w[itr])
rep = classification_report(y[iva], rf.predict(X[:, sorted(top)][iva]),
                            output_dict=True, zero_division=0)
print('留出 OA(top-24):', round(rep['accuracy'], 4))

# ---------- 树字符串转换（SERVIR/rpart 风格） ----------
def tree_to_string(tree, classes, feats_m):
    """sklearn 树 → GEE decisionTree 字符串。左子(sklearn <=t)行条件 '<'，右子 '>='"""
    lines = ['1) root 9999 9999 9999']

    def leaf_label(idx):
        return int(classes[int(np.argmax(tree.value[idx][0]))])

    def rec(skidx, node):
        li, ri = tree.children_left[skidx], tree.children_right[skidx]
        f = feats_m[tree.feature[skidx]]
        t = tree.threshold[skidx]
        # 左子: 条件 f < t
        if tree.children_left[li] == -1 and tree.children_right[li] == -1:
            lines.append(f'{2*node}) {f}<{t} 9999 9999 {leaf_label(li)} *')
        else:
            lines.append(f'{2*node}) {f}<{t} 9999 9999 9999')
            rec(li, 2 * node)
        # 右子: 条件 f >= t
        if tree.children_left[ri] == -1 and tree.children_right[ri] == -1:
            lines.append(f'{2*node+1}) {f}>={t} 9999 9999 {leaf_label(ri)} *')
        else:
            lines.append(f'{2*node+1}) {f}>={t} 9999 9999 9999')
            rec(ri, 2 * node + 1)

    rec(0, 1)
    return '\n'.join(lines)

tree_strings = [tree_to_string(est.tree_, rf.classes_, FEATS_M) for est in rf.estimators_]
json.dump({'features': FEATS_M, 'classes': rf.classes_.tolist(), 'trees': tree_strings},
          open(os.path.join(MODEL_DIR, 'trees_bj.json'), 'w'))
joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_bj.joblib'))
print('模型+树串已保存; 树数:', len(tree_strings))

# ---------- G2: GEE 端一致性校验 ----------
print('=== G2 一致性校验 ===')
import ee
import requests
ee, pid = PC.load_account('zsi8emo')
clf_gee = ee.Classifier.decisionTreeEnsemble(tree_strings)
BJ = ee.Geometry(PC.beijing_geojson())
emb20 = (ee.ImageCollection(PC.EMB_COL).filterDate('2020-01-01', '2021-01-01')
         .filterBounds(BJ).mosaic().select(FEATS_M))

ho = pd.read_parquet(os.path.join(OUT_DIR, 'samples_bj_holdout.parquet'))
known = set(rf.classes_.tolist())
ho_k = ho[ho.cls.isin(known)].head(600).reset_index(drop=True)
if 'lon' not in ho_k.columns:
    # 旧版路径：经 v2 底座桥接坐标
    bj = BASE[(BASE.lon >= PC.BBOX[0]) & (BASE.lon <= PC.BBOX[2]) &
              (BASE.lat >= PC.BBOX[1]) & (BASE.lat <= PC.BBOX[3])].reset_index(drop=True)
    ho_k = ho_k.merge(bj[['lon', 'lat']].reset_index().rename(columns={'index': 'row_id'}),
                      on='row_id', how='left')
fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'i': int(r.row_id)})
    for r in ho_k.itertuples()])
samp = emb20.classify(clf_gee).rename('classification').sampleRegions(
    collection=fc, properties=['i'], scale=10, tileScale=2)
url = samp.getDownloadURL(filetype='csv', selectors=['i', 'classification'])
r = requests.get(url, proxies=PC.PROXY, timeout=600)
gd = pd.read_csv(io.BytesIO(r.content))
m = gd.set_index('i')['classification'].to_dict()

Xho = ho_k[FEATS_M].to_numpy(float)
y_sk = rf.predict(Xho)
mismatch = 0
n = 0
for i, rid in enumerate(ho_k['row_id']):
    if rid in m and pd.notna(m[rid]):
        n += 1
        if int(m[rid]) != int(y_sk[i]):
            mismatch += 1
rate = mismatch / max(n, 1)
print(f'可比点: {n}  不一致: {mismatch}  不一致率: {rate:.4f}')
print('G2 判定:', 'PASS' if rate <= 0.001 else 'FAIL')

# 本地留出精度（bronze，已知类）
yho = ho['cls'].to_numpy()
mask_k = np.isin(yho, list(known))
pred_ho = rf.predict(ho.loc[mask_k, FEATS_M].to_numpy(float))
rep_ho = classification_report(yho[mask_k], pred_ho, output_dict=True, zero_division=0)
print('留出(bronze 已知类) OA:', round(rep_ho['accuracy'], 4))

json.dump({'oa_holdout_bronze': rep_ho['accuracy'], 'g2_rate': rate, 'g2_n': n,
           'feats_m': FEATS_M, 'classes': rf.classes_.tolist(), 'rep': rep_ho,
           'oa_train_holdout_top24': rep['accuracy']},
          open(os.path.join(MODEL_DIR, 'p3_report.json'), 'w'), ensure_ascii=False, indent=2)
print('已写 p3_report.json')
