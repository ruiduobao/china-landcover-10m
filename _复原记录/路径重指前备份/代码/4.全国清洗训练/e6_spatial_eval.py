# -*- coding: utf-8 -*-
"""
e6_spatial_eval.py — P3.4 空间独立评估（评审阶段 H4）
* 验证池: 数据/本地处理/样本底座/validation_pool_v2.parquet
  （25km Albers 块整块留出、块内侵蚀 5km，验证点到训练点 ≥5km；4918 点 / 30 类 / 8 生态区）
* 训练集: 年度子集/r7_train_{y}.parquet（e2b 产出）**剔除 holdout 块内全部点**
  （块键清单 validation_pool_v2_holdout_blocks.json，e0c 产出）
* 报告（逐年份 + 逐生态区 + 逐类）: OA / macro-F1 / balanced accuracy /
  UA-PA-F1 / 混淆矩阵；Olofsson 面积置信区间（需 --area-prior，缺省只报精度）
* 用法:
    python e6_spatial_eval.py                 # 全部 8 年
    python e6_spatial_eval.py --years 2020 2023
    python e6_spatial_eval.py --trees 200 --dry   # 只校验输入与留出，不训练
"""
import os, sys, json, glob, time, argparse
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
BASE = os.path.join(PROJ, '数据/本地处理/全国清洗训练')
SUB_DIR = os.path.join(BASE, '年度子集_含稀有类')   # 最终交付训练集（e7 合并稀有类后）
POOL = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2.parquet')
HOLDOUT = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2_holdout_blocks.json')
VALIDITY = os.path.join(PROJ, '数据/本地处理/样本重建/r7_train_validity.parquet')
OUT_DIR = os.path.join(BASE, '评估_P3.4')
os.makedirs(OUT_DIR, exist_ok=True)
FEATS = [f'A{i:02d}' for i in range(64)]
YEARS = list(range(2017, 2025))
BLOCK = 25000

# 与 e0c_validation_redesign.py 完全一致（生态区 = 省 → 8 区）
ZONE_OF_PROV = {
    '黑龙江省': '东北', '吉林省': '东北', '辽宁省': '东北',
    '北京市': '华北', '天津市': '华北', '河北省': '华北', '山西省': '华北',
    '山东省': '华北', '河南省': '华北',
    '上海市': '长江中下游', '江苏省': '长江中下游', '安徽省': '长江中下游',
    '湖北省': '长江中下游', '湖南省': '长江中下游', '江西省': '长江中下游', '浙江省': '长江中下游',
    '福建省': '华南', '广东省': '华南', '广西壮族自治区': '华南', '海南省': '华南',
    '台湾省': '华南', '香港特别行政区': '华南', '澳门特别行政区': '华南',
    '重庆市': '西南', '四川省': '西南', '贵州省': '西南', '云南省': '西南',
    '西藏自治区': '青藏', '青海省': '青藏',
    '新疆维吾尔自治区': '西北', '甘肃省': '西北', '宁夏回族自治区': '西北',
    '陕西省': '西北', '内蒙古自治区': '西北',
}


def albers_bk(lon, lat):
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326',
                              '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 '
                              '+datum=WGS84 +units=m', always_xy=True)
    X, Y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return (np.floor(X / BLOCK).astype(np.int64) * 100000 +
            np.floor(Y / BLOCK).astype(np.int64))


def load_val_ids():
    """validation_pool_v2 → row_id（r7_train_validity 现带显式 row_id 列）"""
    v = pd.read_parquet(VALIDITY, columns=['row_id', 'lon', 'lat', 'class_new', 'province'])
    vp = pd.read_parquet(POOL)
    m = vp.merge(v, on=['lon', 'lat'], how='left', suffixes=('', '_v'))
    miss = m.row_id.isna().sum()
    if miss:
        print(f'⚠️ 验证池 {miss} 点未匹配到 row_id（跳过）', flush=True)
    m = m[m.row_id.notna()].copy()
    m['row_id'] = m.row_id.astype(np.int64)
    m['zone'] = m.province.map(ZONE_OF_PROV).fillna('其他')
    return m[['row_id', 'lon', 'lat', 'class_new', 'zone', 'province']]


def metrics_from_cm(cm, labels):
    """cm: (n,n) 行=真值 列=预测；返回 OA/macro-F1/balanced-acc/逐类表"""
    tot = cm.sum()
    oa = float(np.trace(cm) / tot) if tot else np.nan
    f1s, recs, uas, pas = [], [], [], []
    rows = []
    for i, c in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        sup = cm[i, :].sum()
        pred_n = cm[:, i].sum()
        ua = float(tp / pred_n) if pred_n else np.nan
        pa = float(tp / sup) if sup else np.nan
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan
        rows.append({'class': int(c), 'n_ref': int(sup), 'n_pred': int(pred_n),
                     'UA': ua, 'PA': pa, 'F1': f1})
        if sup > 0:
            f1s.append(f1); recs.append(pa); uas.append(ua)
    return {
        'n': int(tot), 'OA': oa,
        'macro_F1': float(np.nanmean(f1s)) if f1s else np.nan,
        'balanced_accuracy': float(np.nanmean(recs)) if recs else np.nan,
        'macro_UA': float(np.nanmean(uas)) if uas else np.nan,
        'n_classes_scored': len(f1s),
    }, pd.DataFrame(rows)


def olofsson(cm, labels, area_prior):
    """Olofsson et al. 2014 分层面积估计（层=预测类，权重=该层面积占比）
    area_prior: {class: area_proportion}（总和≈1）；缺省返回 None"""
    if not area_prior:
        return None
    W = np.array([float(area_prior.get(str(int(c)), 0.0)) for c in labels])
    if W.sum() <= 0:
        return None
    W = W / W.sum()
    n_h = cm.sum(axis=0).astype(float)          # 每层（预测类）验证点数
    out = {}
    for j, cj in enumerate(labels):
        est = 0.0; var = 0.0
        for i, ci in enumerate(labels):
            if n_h[i] <= 0:
                continue
            p = cm[j, i] / n_h[i]
            est += W[i] * p
            var += (W[i] ** 2) * p * (1 - p) / max(n_h[i] - 1, 1)
        se = float(np.sqrt(var))
        out[int(cj)] = {'area_frac': float(est), 'SE': se,
                        'CI95': [float(max(0.0, est - 1.96 * se)), float(est + 1.96 * se)]}
    return out


def train_and_eval(y, val, holdout_keys, trees, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    fp = os.path.join(SUB_DIR, f'r7_train_{y}.parquet')
    if not os.path.isfile(fp):
        return None, None, f'缺 {os.path.basename(fp)}'
    df = pd.read_parquet(fp)
    bk = albers_bk(df.lon.to_numpy(), df.lat.to_numpy())
    in_hold = np.isin(bk, holdout_keys)
    tr = df[~in_hold]
    print(f'y{y}: 年度子集 {len(df):,}，留出块内 {int(in_hold.sum()):,}，训练 {len(tr):,}', flush=True)
    X = tr[FEATS].to_numpy(np.float32)
    ytr = tr.class_new.to_numpy(int)
    fin = np.isfinite(X).all(1)
    if not fin.all():
        print(f'  丢弃非有限特征 {int((~fin).sum()):,}', flush=True)
        X, ytr = X[fin], ytr[fin]
        tr = tr[fin]
    w = tr.train_weight.to_numpy(np.float32) if 'train_weight' in tr else None
    if w is not None:
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    rf = RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=seed,
                                min_samples_leaf=2, max_features='sqrt',
                                class_weight='balanced_subsample')
    t0 = time.time()
    rf.fit(X, ytr, sample_weight=w)
    print(f'  训练完成 {time.time()-t0:.0f}s，树 {trees}', flush=True)
    # 验证点：取该年存在的嵌入
    v = val.merge(df[['row_id'] + FEATS], on='row_id', how='inner')
    if len(v) == 0:
        return None, None, f'y{y} 验证点无嵌入'
    VX = v[FEATS].to_numpy(np.float32)
    vfin = np.isfinite(VX).all(1)
    if not vfin.all():
        print(f'  验证点丢弃非有限特征 {int((~vfin).sum()):,}', flush=True)
        v = v[vfin]
        VX = VX[vfin]
    P = rf.predict(VX)
    labels = sorted(set(v.class_new.unique()) | set(np.unique(P)))
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(v.class_new.to_numpy(int), P):
        cm[idx[t], idx[p]] += 1
    vv = v.assign(pred=P)
    return (cm, labels, vv), {'train_n': int(len(tr)), 'val_n': int(len(vv))}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    ap.add_argument('--trees', type=int, default=150)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--area-prior', default=None, help='JSON {class: area_proportion}')
    a = ap.parse_args()

    t0 = time.time()
    holdout_keys = np.asarray(json.load(open(HOLDOUT))['block_keys'], dtype=np.int64)
    val = load_val_ids()
    print(f'验证池 row_id 恢复 {len(val):,}，生态区: {val.zone.value_counts().to_dict()}', flush=True)
    print(f'holdout 块键 {len(holdout_keys):,}', flush=True)
    if a.dry:
        print('DRY：输入校验通过，未训练。')
        return

    area_prior = json.load(open(a.area_prior)) if a.area_prior else None
    report = {'time': time.strftime('%Y-%m-%d %H:%M'), 'trees': a.trees,
              'val_pool_n': int(len(val)), 'by_year': {}, 'by_zone': {}, 'per_class': {}}
    all_cm = None
    labels_all = None
    for y in a.years:
        r, info, err = train_and_eval(y, val, holdout_keys, a.trees)
        if err:
            print('  ', err, flush=True); report['by_year'][str(y)] = {'error': err}; continue
        cm, labels, vv = r
        met, per_cls = metrics_from_cm(cm, labels)
        met['train_n'] = info['train_n']; met['val_n'] = info['val_n']
        report['by_year'][str(y)] = met
        per_cls.to_csv(os.path.join(OUT_DIR, f'per_class_{y}.csv'), index=False)
        pd.DataFrame(cm, index=[f'true_{c}' for c in labels],
                     columns=[f'pred_{c}' for c in labels]).to_csv(
            os.path.join(OUT_DIR, f'confusion_{y}.csv'))
        # 逐生态区
        for z, g in vv.groupby('zone'):
            zl = sorted(set(g.class_new.unique()) | set(g.pred.unique()))
            zi = {c: i for i, c in enumerate(zl)}
            zcm = np.zeros((len(zl), len(zl)), dtype=np.int64)
            for t, p in zip(g.class_new.to_numpy(int), g.pred.to_numpy(int)):
                zcm[zi[t], zi[p]] += 1
            zm, _ = metrics_from_cm(zcm, zl)
            report['by_zone'].setdefault(str(y), {})[z] = zm
        print(f'  y{y}: OA={met["OA"]:.4f} macro-F1={met["macro_F1"]:.4f} '
              f'bal-acc={met["balanced_accuracy"]:.4f} n={met["n"]}', flush=True)
        # 汇总混淆矩阵（按并集标签重排后相加）
        if all_cm is None:
            labels_all, all_cm = list(labels), cm
        else:
            merged = sorted(set(labels_all) | set(labels))

            def grow(M, L, newL):
                n = len(newL)
                pos = {c: i for i, c in enumerate(newL)}
                G = np.zeros((n, n), dtype=np.int64)
                for i, ci in enumerate(L):
                    for j, cj in enumerate(L):
                        G[pos[ci], pos[cj]] += M[i, j]
                return G

            all_cm = grow(all_cm, labels_all, merged) + grow(cm, labels, merged)
            labels_all = merged

    if all_cm is not None:
        # 全国汇总
        met_all, per_cls_all = metrics_from_cm(all_cm, labels_all)
        report['overall'] = met_all
        per_cls_all.to_csv(os.path.join(OUT_DIR, 'per_class_overall.csv'), index=False)
        if area_prior:
            report['olofsson'] = olofsson(all_cm, labels_all, area_prior)
        else:
            report['olofsson'] = {'note': '未提供 --area-prior（需最终分类图各类面积占比），'
                                          '面积置信区间待制图后补算'}
    report['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(report, open(os.path.join(OUT_DIR, '评估报告_P3.4.json'), 'w',
                           encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report.get('overall', {}), ensure_ascii=False))
    print('输出目录:', OUT_DIR)


if __name__ == '__main__':
    main()
