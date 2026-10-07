# -*- coding: utf-8 -*-
"""
p10_probe_train_cap.py — 路线 A：证伪/确认"GEE 内训练样本量上限"（账号恢复后第一个要跑的脚本）
为什么关键（技术文档 29 §二）：GEE 内训练被卡在 ≈2 万点（5 万即 `Computed value is too large`），
  这直接把全国 OA 压在 0.60。若这个上限只是"运行时过滤太重"造成的假象，全国精度可提到 0.71。
三个实验（任一成功即改架构）：
  E1 预置资产：上传**恰好 N 点**的独立资产，直接 train(asset) —— 不做运行时 randomColumn 过滤
  E2 降维：把 64 维 PCA 到 K 维后再上传训练，看同样载荷能否容纳更多点
  E3 参数联合调优：maxNodes × variablesPerSplit × numberOfTrees 的组合，找"同载荷最优"

用法（账号恢复后）：
  python p10_probe_train_cap.py --acct <账号> --proj <项目> --exp E1 --sizes 20000 50000 100000
  python p10_probe_train_cap.py --acct <账号> --proj <项目> --exp E2 --pcadim 16 --n 200000
  python p10_probe_train_cap.py --acct <账号> --proj <项目> --exp E3
输出：控制台表 + plan/traincap_report.json（结论直接决定全国用哪条路线）
"""
import os, sys, io, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')
WEAK = [52, 62, 91, 140, 11]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def boot(acct, proj):
    import ee
    C.apply_account_env(acct)
    for i in range(5):
        try:
            ee.Initialize(project=proj)
            return ee
        except Exception as x:
            if i == 4:
                raise
            emit('init retry %d: %s' % (i + 1, str(x)[:70])); time.sleep(20)


def make_points(n, seed=20260915, year=2023):
    """本地 sqrt 配比抽样（与生产一致的配比），返回 DataFrame + meta"""
    import pandas as pd
    fp = os.path.join(C.WORK, 'traincap_sample_%d.csv' % n)
    if os.path.isfile(fp):
        return pd.read_csv(fp)
    src = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类',
                       'r7_train_%d.parquet' % year)
    df = pd.read_parquet(src, columns=['row_id', 'lon', 'lat', 'class_new'] + C.FEATS)
    cnt = df.class_new.value_counts()
    wt = (cnt / cnt.sum()) ** 0.5
    wt = wt / wt.sum()
    per = (wt * n).round().astype(int).clip(lower=150)
    rng = np.random.default_rng(seed)
    keep = []
    for c, k in per.items():
        g = df[df.class_new == c]
        if len(g):
            keep.append(df.loc[g.index[rng.choice(len(g), min(len(g), int(k)), replace=False)]])
    s = pd.concat(keep, ignore_index=True)
    for f_ in C.FEATS:                      # 截位：控制内联载荷（不截会超 10MB 请求上限）
        s[f_] = s[f_].round(3)
    s.to_csv(fp, index=False)
    emit('样本 %d 点 → %s' % (len(s), fp))
    return s


def upload_asset(ee, pid, folder, pts, name, shard=6000, feats=None):
    """上传"恰好 N 点"的独立资产（不带运行时过滤），分片后服务端合并成单资产"""
    root = 'projects/%s/assets/%s' % (pid, folder)
    try:
        ee.data.getAsset(root)
    except Exception:
        try:
            ee.data.createAsset({'type': 'Folder'}, root)
        except Exception:
            pass
    merged = '%s/%s' % (root, name)
    try:
        info = ee.data.getAsset(merged)
        if int(info.get('sizeBytes', 0) or 0) > 1000:
            emit('资产已存在，复用 %s' % merged)
            return merged
    except Exception:
        pass
    parts = [pts.iloc[i:i + shard] for i in range(0, len(pts), shard)]
    tasks = []
    for k, ch in enumerate(parts):
        aid = '%s_%03d' % (merged, k)
        try:
            ee.data.deleteAsset(aid)
        except Exception:
            pass
        _fs = feats or C.FEATS
        recs = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                           {**{f: float(getattr(r, f)) for f in _fs},
                            'cl': int(r.class_new)}) for r in ch.itertuples()]
        t = ee.batch.Export.table.toAsset(collection=ee.FeatureCollection(recs),
                                          description='cap_%s_%03d' % (name[-8:], k), assetId=aid)
        t.start()
        tasks.append({'id': t.id, 'asset': aid, 'n': len(ch)})
        emit('  片 %03d：%d 点' % (k, len(ch)))
    for _ in range(120):
        time.sleep(20)
        st = [ee.data.getTaskStatus(x['id'])[0]['state'] for x in tasks]
        if all(s in ('COMPLETED', 'FAILED', 'CANCELLED') for s in st):
            break
    ok = [x['asset'] for x, s in zip(tasks, st) if s == 'COMPLETED']
    if len(ok) != len(tasks):
        emit('  部分分片失败：%d/%d' % (len(ok), len(tasks)))
    fc = ee.FeatureCollection(ok[0])
    for a_ in ok[1:]:
        fc = fc.merge(ee.FeatureCollection(a_))
    try:
        ee.data.deleteAsset(merged)
    except Exception:
        pass
    t = ee.batch.Export.table.toAsset(collection=fc,
                                      description='cap_merge_%s' % name[-8:], assetId=merged)
    t.start()
    for _ in range(120):
        time.sleep(20)
        s = ee.data.getTaskStatus(t.id)[0]
        if s.get('state') in ('COMPLETED', 'FAILED', 'CANCELLED'):
            emit('  合并 %s' % s.get('state'))
            break
    return merged


def try_train(ee, asset, n_trees=100, max_nodes=20000, vars_per_split=None, pca_bands=None):
    """在 1° 区上真跑一次 classify，判断能否训练成功"""
    feats = pca_bands or C.FEATS
    kw = dict(numberOfTrees=n_trees, minLeafPopulation=2, maxNodes=max_nodes)
    if vars_per_split:
        kw['variablesPerSplit'] = int(vars_per_split)
    reg = ee.Geometry.Rectangle([120.0, 40.0, 121.0, 41.0])
    img = (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
           .filterBounds(reg).mosaic().select(feats))
    t0 = time.time()
    try:
        clf = ee.Classifier.smileRandomForest(**kw).train(ee.FeatureCollection(asset), 'cl', feats)
        h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=100,
                                           maxPixels=10**9, tileScale=4,
                                           bestEffort=True).getInfo()
        ncl = len((h or {}).get('classification', {}) or {})
        return True, ncl, round(time.time() - t0, 1), ''
    except Exception as e:
        return False, 0, round(time.time() - t0, 1), str(e)[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--proj', required=True)
    ap.add_argument('--exp', default='E1', choices=['E1', 'E2', 'E3', 'E4'])
    ap.add_argument('--sizes', type=int, nargs='*', default=[20000, 50000, 100000])
    ap.add_argument('--pcadim', type=int, default=16)
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--folder', default='lccap')
    a = ap.parse_args()
    C.ensure_dirs()
    ee = boot(a.acct, a.proj)
    rep = {'acct': a.acct, 'proj': a.proj, 'exp': a.exp, 'runs': []}
    emit('== %s ==' % a.exp)

    if a.exp in ('E1', 'E3'):
        for n in a.sizes:
            pts = make_points(n)
            asset = upload_asset(ee, a.proj, a.folder, pts, 'cap_n%d' % n)
            ok, ncl, sec, err = try_train(ee, asset, n_trees=100, max_nodes=20000)
            rep['runs'].append({'n': n, 'asset': asset, 'ok': ok, 'n_class': ncl,
                                'sec': sec, 'err': err})
            emit('  E1 n=%-7d → %s  类数=%d  %.0fs  %s' % (n, '✅可训练' if ok else '❌失败',
                                                       ncl, sec, err))
            if not ok:
                break
    if a.exp == 'E2':
        pts = make_points(a.n)
        X = pts[C.FEATS].to_numpy(np.float32)
        from sklearn.decomposition import PCA
        p = PCA(n_components=a.pcadim, random_state=42).fit(X)
        Z = p.transform(X)
        cols = ['P%02d' % i for i in range(a.pcadim)]
        import pandas as pd
        q = pts[['lon', 'lat', 'class_new']].copy()
        for i, c in enumerate(cols):
            q[c] = Z[:, i]
        q.to_csv(os.path.join(C.WORK, 'traincap_pca%d.csv' % a.pcadim), index=False)
        asset = upload_pca(ee, a.proj, a.folder, q, cols, 'pca%d_n%d' % (a.pcadim, a.n))
        ok, ncl, sec, err = try_train(ee, asset, pca_bands=cols)
        rep['runs'].append({'pca': a.pcadim, 'n': a.n, 'ok': ok, 'n_class': ncl,
                            'sec': sec, 'err': err})
        emit('  E2 PCA=%d n=%d → %s 类数=%d %.0fs %s'
             % (a.pcadim, a.n, '✅' if ok else '❌', ncl, sec, err))
    if a.exp == 'E4':
        # E4：值预算测试——固定"点数×特征数"≈1.28M（GEE 已验证可训练的量级），
        # 看多点少特征能否通过。用**前 K 个原始 AEF 波段**而非 PCA，
        # 这样分类影像无需做线性变换（E2 就是因为去 AEF 里找 P00 波段而误报失败）。
        import pandas as pd
        for N, K in [(a.sizes[0], a.pcadim), (a.sizes[-1], a.pcadim)]:
            pts = make_points(N)
            cols = C.FEATS[:K]
            q = pts[['lon', 'lat', 'class_new'] + cols].copy()
            for f_ in cols:
                q[f_] = q[f_].round(3)
            asset = upload_asset(ee, a.proj, a.folder, q, 'vb_n%d_k%d' % (N, K), feats=cols)
            ok, ncl, sec, err = try_train(ee, asset, n_trees=100, max_nodes=20000,
                                          pca_bands=cols)
            rep['runs'].append({'exp': 'E4', 'n': N, 'k': K, 'values': N * K,
                                'ok': ok, 'n_class': ncl, 'sec': sec, 'err': err})
            emit('  E4 %d点×%d维（%.2fM 值）→ %s 类数=%d %.0fs %s'
                 % (N, K, N * K / 1e6, '✅可训练' if ok else '❌失败', ncl, sec, err))
    if a.exp == 'E3':
        pts = make_points(max(a.sizes))
        asset = upload_asset(ee, a.proj, a.folder, pts, 'cap_n%d' % max(a.sizes))
        for trees, mn, vps in [(100, 20000, None), (100, 5000, None), (150, 20000, 8),
                               (100, 20000, 32), (200, 10000, 64)]:
            ok, ncl, sec, err = try_train(ee, asset, trees, mn, vps)
            rep['runs'].append({'trees': trees, 'max_nodes': mn, 'vars': vps,
                                'ok': ok, 'n_class': ncl, 'sec': sec, 'err': err})
            emit('  E3 树%d maxNodes%s vars%s → %s 类数=%d %.0fs %s'
                 % (trees, mn, vps, '✅' if ok else '❌', ncl, sec, err))
    json.dump(rep, open(os.path.join(C.PLAN, 'traincap_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('结论写入 plan/traincap_report.json')
    ok_any = any(r.get('ok') for r in rep['runs'])
    emit('总判：%s' % ('✅ 上限可突破 → 走 20 万点部署，预期全国 OA ≈0.71'
                       if ok_any else '❌ 上限确认 → 转路线 B（分层）或路线 C（区域精编版）'))


def upload_pca(ee, pid, folder, q, cols, name, shard=6000):
    root = 'projects/%s/assets/%s' % (pid, folder)
    try:
        ee.data.getAsset(root)
    except Exception:
        try:
            ee.data.createAsset({'type': 'Folder'}, root)
        except Exception:
            pass
    merged = '%s/%s' % (root, name)
    try:
        if int(ee.data.getAsset(merged).get('sizeBytes', 0) or 0) > 1000:
            return merged
    except Exception:
        pass
    parts = [q.iloc[i:i + shard] for i in range(0, len(q), shard)]
    tasks = []
    for k, ch in enumerate(parts):
        aid = '%s_%03d' % (merged, k)
        try:
            ee.data.deleteAsset(aid)
        except Exception:
            pass
        recs = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                           {**{c: float(getattr(r, c)) for c in cols},
                            'cl': int(r.class_new)}) for r in ch.itertuples()]
        t = ee.batch.Export.table.toAsset(collection=ee.FeatureCollection(recs),
                                          description='pca_%03d' % k, assetId=aid)
        t.start()
        tasks.append({'id': t.id, 'asset': aid})
    for _ in range(120):
        time.sleep(20)
        st = [ee.data.getTaskStatus(x['id'])[0]['state'] for x in tasks]
        if all(s in ('COMPLETED', 'FAILED', 'CANCELLED') for s in st):
            break
    ok = [x['asset'] for x, s in zip(tasks, st) if s == 'COMPLETED']
    fc = ee.FeatureCollection(ok[0])
    for a_ in ok[1:]:
        fc = fc.merge(ee.FeatureCollection(a_))
    try:
        ee.data.deleteAsset(merged)
    except Exception:
        pass
    t = ee.batch.Export.table.toAsset(collection=fc, description='pca_merge', assetId=merged)
    t.start()
    for _ in range(120):
        time.sleep(20)
        if ee.data.getTaskStatus(t.id)[0].get('state') in ('COMPLETED', 'FAILED', 'CANCELLED'):
            break
    return merged


if __name__ == '__main__':
    main()
