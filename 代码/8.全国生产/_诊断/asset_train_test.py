# -*- coding: utf-8 -*-
"""
asset_train_test.py — 生产架构的关键验证：样本能否上 GEE + 服务端训练是否可行
背景：内联树集成受请求体积限制（150/深14/叶200 ≈22MB 时 OA 只有 0.5425），
      而完整模型 2.9GB 不可能内联 → 必须用"上传样本资产 → GEE 服务端 smileRandomForest 训练"。
      未知项：(a) 内联 FeatureCollection 分片导出资产的上限与耗时
              (b) GEE 端训练 + 分类 的墙钟耗时（决定每任务开销）
              (c) 资产能否跨账号共享（省掉每账号重复上传）
本脚本按 (a)→(b) 顺序实测，规模从小到大，任一步失败即停并报告。
用法：python asset_train_test.py --acct zixen8v8 [--n 5000] [--shards 1] [--share-to y30b63ye]
"""
import os, sys, io, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')
WEAK = [52, 62, 91, 140, 11]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def build_sample(n, year=2022):
    fp = os.path.join(C.WORK, 'prod_sample_%d.csv' % n)
    if os.path.isfile(fp):
        return pd.read_csv(fp)
    src = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类',
                       'r7_train_%d.parquet' % year)
    df = pd.read_parquet(src, columns=['lon', 'lat', 'class_new'] + C.FEATS)
    rng = np.random.default_rng(20260914)
    per = max(n // 30, 60)
    keep = []
    for c, g in df.groupby('class_new'):
        k = min(len(g), 300 if c in WEAK else per)
        keep.append(g.iloc[rng.choice(len(g), k, replace=False)])
    s = pd.concat(keep, ignore_index=True)
    for f in C.FEATS:
        s[f] = s[f].round(3)
    s.to_csv(fp, index=False)
    emit('子样本 %d 点（弱类保底 300）' % len(s))
    return s


def boot(acct):
    import ee
    C.apply_account_env(acct)
    for i in range(5):
        try:
            ee.Initialize(project=C.ACCOUNTS[acct]['proj']); break
        except Exception as x:
            if i == 4: raise
            emit('init retry %d' % (i + 1)); time.sleep(20)
    return ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--share-to', default='')
    a = ap.parse_args()
    C.ensure_dirs()
    acct = a.acct
    pid = C.ACCOUNTS[acct]['proj']
    folder = C.ACCOUNTS[acct]['folder']
    ee = boot(acct)
    root = 'projects/%s/assets/%s' % (pid, folder)
    try:
        ee.data.getAsset(root)
        emit('资产根已存在 %s' % root)
    except Exception:
        try:
            ee.data.createAsset({'type': 'Folder'}, root)
            emit('已建资产根 %s' % root)
        except Exception as e:
            emit('建根: %s' % str(e)[:100])

    s = build_sample(a.n)
    per = int(np.ceil(len(s) / a.shards))
    emit('== (a) 分片导出样本资产：%d 点 / %d 片（每片 %d）==' % (len(s), a.shards, per))
    tasks = []
    for k in range(a.shards):
        chunk = s.iloc[k * per:(k + 1) * per]
        recs = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                           {**{f: float(getattr(r, f)) for f in C.FEATS},
                            'cl': int(r.class_new)}) for r in chunk.itertuples()]
        fc = ee.FeatureCollection(recs)
        aid = '%s/samples_%d_%03d' % (root, a.n, k)
        try:                       # 同名资产不可覆盖 → 先删（幂等重跑）
            ee.data.deleteAsset(aid)
            emit('  片 %d 删除同名旧资产' % k)
        except Exception:
            pass
        t0 = time.time()
        try:
            t = ee.batch.Export.table.toAsset(collection=fc, description='lcsamp_%s_%03d' % (acct, k),
                                              assetId=aid)
            t.start()
            tasks.append({'id': t.id, 'asset': aid, 'n': len(chunk),
                          'payload_mb': round(len(chunk) * 64 * 7 / 1e6, 2)})
            emit('  片 %d：%d 点（载荷 ~%.1f MB）→ 提交 %s' % (k, len(chunk),
                                                           len(chunk) * 64 * 7 / 1e6, t.id))
        except Exception as e:
            emit('  片 %d 提交失败: %s' % (k, str(e)[:140]))
            return 1
        time.sleep(2)
    # 轮询
    st = {}
    for _ in range(120):
        time.sleep(15)
        st = {x['id']: ee.data.getTaskStatus(x['id'])[0]['state'] for x in tasks}
        emit('  状态: %s' % st)
        if all(v in ('COMPLETED', 'FAILED', 'CANCELLED') for v in st.values()):
            break
    ok = sum(1 for v in st.values() if v == 'COMPLETED')
    emit('  导出完成 %d/%d' % (ok, len(tasks)))
    if ok < len(tasks):
        return 1

    emit('== (b) GEE 端训练 + 分类 耗时 ==')
    fc = ee.FeatureCollection([t['asset'] for t in tasks]).flatten()
    n_fc = int(fc.size().getInfo())
    emit('  合并后样本 %d 行' % n_fc)
    reg = ee.Geometry.Rectangle([116.80, 36.50, 117.00, 36.70])
    img = (ee.ImageCollection(C.AEF).filterDate('2022-01-01', '2023-01-01')
           .filterBounds(reg).mosaic().select(C.FEATS))
    for ntrees, maxnodes in [(100, None), (150, None)]:
        t0 = time.time()
        try:
            clf = ee.Classifier.smileRandomForest(numberOfTrees=ntrees, minLeafPopulation=2,
                                                  maxNodes=maxnodes).train(fc, 'cl', C.FEATS)
            h = img.classify(clf).reduceRegion(ee.Reducer.frequencyHistogram(), reg,
                                               scale=60, maxPixels=10 ** 9, tileScale=4,
                                               bestEffort=True).getInfo()
            hh = (h or {}).get('classification', {}) or {}
            emit('  树%d maxNodes=%s → %.0fs，区内 %d 类 %s'
                 % (ntrees, maxnodes, time.time() - t0, len(hh),
                    sorted(int(k) for k in hh)[:10]))
        except Exception as e:
            emit('  树%d 失败: %s' % (ntrees, str(e)[:140]))
    if a.share_to:
        emit('== (c) 资产跨账号共享测试 → %s ==' % a.share_to)
        try:
            ee.data.setAssetAcl(tasks[0]['asset'], {'readers': ['earthengine-acl-test@example.com']})
            emit('  setAssetAcl 接口可用（用示例邮箱试写，仅验证接口存在）')
        except Exception as e:
            emit('  setAssetAcl: %s' % str(e)[:140])
    print('\n样本资产：', [t['asset'] for t in tasks])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
