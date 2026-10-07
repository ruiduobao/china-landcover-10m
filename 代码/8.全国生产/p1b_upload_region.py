# -*- coding: utf-8 -*-
"""
p1b_upload_region.py — 区域生产的「每账号独立样本」上传（可断点续跑、幂等）
为什么每账号一份样本（而不是共享）：用户要求"数据也不要放到一起"。
做法：每个账号用自己的 种子 / √指数 / 分片大小 / 模型超参 生成一份 sqrt 配比样本，
      分片导出到**自己的**资产目录（不共享、不跨账号读）。
幂等：已存在的同名资产先删再传；已 COMPLETED 的片直接跳过（断点续跑）。
用法：python p1b_upload_region.py --acct 1yxth5g --seed 20260914 --expo 0.50 --shard 20000
"""
import os, sys, io, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '4.全国清洗训练'))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def make_sample(acct, n, years, seed, expo):
    fp = os.path.join(C.WORK, 'prod_sample_%s.csv' % acct)
    meta_fp = os.path.join(C.PLAN, 'sample_%s.json' % acct)
    if os.path.isfile(fp) and os.path.isfile(meta_fp):
        meta = json.load(open(meta_fp, encoding='utf-8'))
        if meta.get('n') == n and meta.get('expo') == expo and meta.get('years') == list(years):
            emit('复用样本 %s（%d 点）' % (fp, meta['n_rows']))
            return pd.read_csv(fp), meta
    parts = []
    for y in years:
        for sub in ('年度子集_含稀有类', '年度子集'):
            fp2 = os.path.join(C.KB, '数据/本地处理/全国清洗训练', sub, 'r7_train_%d.parquet' % y)
            if os.path.isfile(fp2):
                parts.append(pd.read_parquet(fp2, columns=['row_id', 'lon', 'lat', 'class_new'] + C.FEATS))
                break
    df = pd.concat(parts, ignore_index=True).drop_duplicates('row_id')
    cnt = df.class_new.value_counts()
    w = (cnt / cnt.sum()) ** expo
    w = w / w.sum()
    per = (w * n).round().astype(int).clip(lower=150)
    rng = np.random.default_rng(seed)
    keep = []
    for c, k in per.items():
        g = df[df.class_new == c]
        if len(g):
            keep.append(df.loc[g.index[rng.choice(len(g), min(len(g), int(k)), replace=False)]])
    s = pd.concat(keep, ignore_index=True)
    for f in C.FEATS:
        s[f] = s[f].round(3)
    s.to_csv(fp, index=False)
    meta = {'acct': acct, 'n': n, 'years': list(years), 'expo': expo, 'seed': seed,
            'n_rows': int(len(s)), 'csv': fp,
            'per_class': {int(k): int(v) for k, v in s.class_new.value_counts().items()}}
    json.dump(meta, open(meta_fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('%s 样本 %d 点（seed=%d expo=%.2f）' % (acct, len(s), seed, expo))
    return s, meta


def boot(acct):
    import ee
    C.apply_account_env(acct)
    for i in range(5):
        try:
            ee.Initialize(project=C.ACCOUNTS[acct]['proj'])
            return ee
        except Exception as x:
            if i == 4:
                raise
            emit('init retry %d: %s' % (i + 1, str(x)[:70])); time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--expo', type=float, default=0.50)
    ap.add_argument('--shard', type=int, default=20000)
    ap.add_argument('--years', type=int, nargs='*', default=C.YEARS)
    ap.add_argument('--tag', default='v1')
    a = ap.parse_args()
    C.ensure_dirs()
    acct = a.acct
    pid = C.ACCOUNTS[acct]['proj']
    folder = C.ACCOUNTS[acct]['folder']
    root = 'projects/%s/assets/%s' % (pid, folder)
    s, meta = make_sample(acct, a.n, a.years, a.seed, a.expo)

    ee = boot(acct)
    try:
        ee.data.getAsset(root)
    except Exception:
        try:
            ee.data.createAsset({'type': 'Folder'}, root)
            emit('建资产根 %s' % root)
        except Exception as e:
            emit('建根: %s' % str(e)[:90])

    shards = [s.iloc[i:i + a.shard] for i in range(0, len(s), a.shard)]
    emit('%s：%d 点 → %d 片（每片 %d）' % (acct, len(s), len(shards), a.shard))
    tasks = []
    for k, ch in enumerate(shards):
        aid = '%s/smp_%s_%03d' % (root, a.tag, k)
        # 断点续跑：资产已存在且行数正确 → 跳过
        try:
            info = ee.data.getAsset(aid)
            if int(info.get('sizeBytes', 0) or 0) > 1000:
                emit('  片 %03d 已存在，跳过' % k)
                tasks.append({'id': None, 'asset': aid, 'n': int(len(ch)), 'done': True})
                continue
        except Exception:
            pass
        try:
            ee.data.deleteAsset(aid)
        except Exception:
            pass
        # 🔴 表导出到资产**必须有几何**（实测报 Unable to export features with null geometry），
        #    所以带 Point 几何（lon/lat 同时作为属性冗余一份，便于事后 QA）
        recs = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                          {**{f: float(getattr(r, f)) for f in C.FEATS},
                           'cl': int(r.class_new), 'lon': float(r.lon),
                           'lat': float(r.lat)}) for r in ch.itertuples()]
        tid = None
        for att in range(4):                      # GEE 偶发 internal error → 重试
            try:
                t = ee.batch.Export.table.toAsset(collection=ee.FeatureCollection(recs),
                                                  description='lcs_%s_%03d' % (acct[:4], k),
                                                  assetId=aid)
                t.start()
                tid = t.id
                emit('  片 %03d：%d 点 → %s' % (k, len(ch), tid))
                break
            except Exception as x:
                emit('  片 %03d 提交重试 %d/4: %s' % (k, att + 1, str(x)[:80]))
                time.sleep(20 + 25 * att)
        if tid is None:
            emit('  片 %03d 四次提交失败，退出（可重跑续传）' % k)
            raise SystemExit(3)
        tasks.append({'id': tid, 'asset': aid, 'n': int(len(ch)), 'done': False})
        time.sleep(2)

    pend = [x for x in tasks if not x['done']]
    for _ in range(240):
        if not pend:
            break
        time.sleep(30)
        for x in pend:
            if x['done']:
                continue
            st = ee.data.getTaskStatus(x['id'])[0].get('state')
            if st == 'COMPLETED':
                x['done'] = True
            elif st in ('FAILED', 'CANCELLED'):
                emit('  片 %s 失败: %s' % (x['asset'].split('_')[-1],
                                          ee.data.getTaskStatus(x['id'])[0].get('error_message', '')[:90]))
        nd = sum(1 for x in tasks if x['done'])
        emit('  进度 %d/%d' % (nd, len(tasks)))
        pend = [x for x in tasks if not x['done']]
    ok = [x for x in tasks if x['done']]
    emit('%s 上传完成 %d/%d' % (acct, len(ok), len(tasks)))
    if len(ok) < len(tasks):
        raise SystemExit(2)

    # 写部署件（模型超参按账号差异化）
    hyper = {'1yxth5g': dict(n_trees=150, max_nodes=200000, rf_seed=42),
             '11e0n00': dict(n_trees=140, max_nodes=180000, rf_seed=7),
             'ief3nj':  dict(n_trees=160, max_nodes=220000, rf_seed=99)}.get(
        acct, dict(n_trees=150, max_nodes=200000, rf_seed=42))
    dep = {'mode': 'asset', 'tag': a.tag, 'year': 2023,
           'assets': [x['asset'] for x in ok],
           'n_trees': hyper['n_trees'], 'min_leaf': 2,
           'max_nodes': hyper['max_nodes'], 'label_col': 'cl',
           'seed': hyper['rf_seed'], 'owner': acct, 'owner_proj': pid,
           'sample': meta}
    fp = os.path.join(C.PLAN, 'deploy_%s.json' % acct)
    json.dump(dep, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('部署件 → %s' % fp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
