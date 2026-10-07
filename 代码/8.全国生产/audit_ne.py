# -*- coding: utf-8 -*-
"""深度诊断 B：用人工标注库审计成品图 + 与本地 v-final 全模型同点对照
产出：① 成品图在我们自己标签上的逐类命中率（区域混淆矩阵）
      ② 同一批点上，本地 v-final（全量+混合规格）的命中率
      → 两者之差 = GEE 训练样本量上限导致的"模型降级"代价（逐类）
"""
import sys, os, json, time
sys.path.insert(0, '.'); sys.path.insert(0, '../4.全国清洗训练'); sys.path.insert(0, '../0.本地流水线')
import numpy as np, pandas as pd, rasterio
import prod_conf as C
from lc_conf import CLASSES
import e6_spatial_eval as E6
from sklearn.ensemble import RandomForestClassifier

FP = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/生产输出/成品/东北三省_2023_100m.tif'
BBOX_NE = (118.5, 38.5, 135.0, 54.0)
WEAK = [52, 62, 91, 140, 11]
HOST = {52: [51, 61, 71, 72, 120, 121, 62, 92], 62: [61, 51, 71, 82, 121, 52, 120],
        91: [71, 61, 92, 82, 51, 72, 81], 140: [150, 201, 121, 130, 120, 220],
        11: [10, 12, 120, 121, 130, 71, 51]}

t0 = time.time()
sub = pd.read_parquet(os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类/r7_train_2023.parquet'))
sub = sub[(sub.lon >= BBOX_NE[0]) & (sub.lon < BBOX_NE[2]) &
          (sub.lat >= BBOX_NE[1]) & (sub.lat < BBOX_NE[3])].reset_index(drop=True)
print('东北区域内 2023 训练点 %d（%.0fs）' % (len(sub), time.time() - t0), flush=True)

with rasterio.open(FP) as s:
    tr, H, W = s.transform, s.height, s.width
    r = ((tr.f - sub.lat.to_numpy()) / abs(tr.e)).astype(int)
    c = ((sub.lon.to_numpy() - tr.c) / tr.a).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    rr, cc = r[ok], c[ok]
    win = rasterio.windows.Window(cc.min(), rr.min(), cc.max()-cc.min()+1, rr.max()-rr.min()+1)
    blk = np.asarray(s.read(1, window=win))
    mapcls = blk[rr - rr.min(), cc - cc.min()]
sub = sub[ok].reset_index(drop=True)
sub['map'] = mapcls
print('落在图幅内 %d 点；图上为 0（无数据）%d 点' % (len(sub), int((sub['map'] == 0).sum())))

def report(pred, tag):
    d = pd.DataFrame({'lab': sub.class_new.to_numpy(int), 'pred': pred})
    rows = []
    for cl in sorted(d.lab.unique()):
        g = d[d.lab == cl]
        hit = int((g.pred == cl).sum())
        top = g.pred.value_counts().head(2).to_dict()
        rows.append({'class': int(cl), 'name': CLASSES[int(cl)][1], 'n': len(g),
                     'hit': hit, 'recall': round(hit/len(g), 3),
                     '误分成': ' / '.join('%d:%d' % (k, v) for k, v in top.items() if k != cl)})
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    df.to_csv(r'F:/lc_work/audit_%s.csv' % tag, index=False, encoding='utf-8-sig')
    return df

print('\n=== ① 成品图（GEE 小模型）在人工标签上的逐类命中率 ===', flush=True)
d1 = report(sub['map'].to_numpy(int), 'gee')
print(d1[d1.n >= 500].to_string(index=False))

print('\n=== ② 本地 v-final 全模型（2023，全量+混合规格）同点对照 ===', flush=True)
hold = np.asarray(json.load(open(C.HOLDOUT))['block_keys'], dtype=np.int64)
tr_df = sub[~np.isin(E6.albers_bk(sub.lon.to_numpy(), sub.lat.to_numpy()), hold)]
X = tr_df[C.FEATS].to_numpy(np.float32); y = tr_df.class_new.to_numpy(int)
print('  训练点 %d' % len(tr_df), flush=True)
main = RandomForestClassifier(n_estimators=150, n_jobs=13, random_state=42,
                              min_samples_leaf=2, max_features='sqrt',
                              class_weight='balanced_subsample').fit(X, y)
weak = RandomForestClassifier(n_estimators=120, n_jobs=13, random_state=77, max_depth=20,
                              min_samples_leaf=2, max_features='sqrt',
                              class_weight='balanced_subsample').fit(X, y)
VX = sub[C.FEATS].to_numpy(np.float32)
pm = main.predict(VX); pw = weak.predict(VX)
pred = pm.copy()
for c_ in WEAK:
    sel = (pw == c_) & np.isin(pm, HOST[c_]); pred[sel] = c_
d2 = report(pred, 'vfinal')
print(d2[d2.n >= 500].to_string(index=False))

m = d1.merge(d2, on=['class', 'name'], suffixes=('_gee', '_vf'))
m = m[m.n_gee >= 300]
m['差'] = (m.recall_vf - m.recall_gee).round(3)
m.to_csv(r'F:/lc_work/audit_compare.csv', index=False, encoding='utf-8-sig')
print('\n=== ③ 降级代价（本地全模型 − GEE 小模型，按样本量排序）===')
print(m[['class', 'name', 'n_gee', 'recall_gee', 'recall_vf', '差']].sort_values('n_gee', ascending=False).to_string(index=False))
print('\n加权总体命中率：GEE %.4f ｜ 本地 v-final %.4f'
      % ((d1.hit.sum()/d1.n.sum()), (d2.hit.sum()/d2.n.sum())))
