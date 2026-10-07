# -*- coding: utf-8 -*-
"""
p1_deploy_model.py — 生产模型部署（v2：共享样本资产 + GEE 端训练）
为什么是这个架构（三条实测结论决定，见 技术文档/28 §二）：
  1) 内联树集成不可行：完整模型 2.9 GB 树字符串；即使裁到 ~22 MB（150/深14/叶200），
     OA 也从 0.7413 掉到 0.5425 —— 低于 GLC_FCS30 自报的 16 类 71.4%。
  2) GEE 端 smileRandomForest 训练**只要 10–20 秒**（实测 5,650 点/100–150 树），
     所以"每个导出任务重算一次训练"的开销可接受。
  3) 样本资产可**跨账号 ACL 共享**（实测被授权账号可读、可直接训练+分类）
     → 样本只需上传一次，不必每个账号重复上传。

生产模型参数（由本地配比实验定案，见 技术文档/28 §三）：
  * 上传配比 = **sqrt 配比**（各类样本数 ∝ √p，弱类温和提升；实测 OA 0.7162、macro-F1 0.6256、30/30 非零）
  * 样本量 ≈ --n（默认 200000），来自全部年份训练子集（合并年度，样本更足）
  * GEE 端 `smileRandomForest(numberOfTrees, minLeafPopulation, maxNodes, seed)`：
    无类别权重，故配比即权重；maxNodes 限深可让稀有类票不被淹没（本地实测深 20 是拐点）

用法：
  python p1_deploy_model.py --make-sample            # 只在本地生成 sqrt 配比样本
  python p1_deploy_model.py --upload                 # 分片导出到主账号资产 + ACL 共享
  python p1_deploy_model.py --deploy-json            # 写各账号 deploy_<acct>.json
  python p1_deploy_model.py --all --n 200000         # 一条龙（上传耗时约 1–2 小时）
"""
import os, sys, io, json, re, time, argparse, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '4.全国清洗训练'))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')
WEAK = [52, 62, 91, 140, 11]
SAMPLE_CSV = os.path.join(C.WORK, 'prod_sample_sqrt.csv')
SAMPLE_META = os.path.join(C.PLAN, 'sample_meta.json')
PRIMARY = 'zixen8v8'          # 主上传账号（样本资产归它所有，再共享出去）
SHARD = 20000                 # 每片点数（20k×64 属性 ≈9 MB 内联载荷，实测可提交）


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def email_of(acct):
    p = os.path.join(C.CRED, acct, '_任务登记.md')
    if os.path.isfile(p):
        m = re.search(r'账号：([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})',
                      io.open(p, encoding='utf-8', errors='replace').read())
        if m:
            return m.group(1)
    return None


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


def make_sample(n, years, seed=20260914):
    """sqrt 配比抽样（合并年度）。同一类别在不同年份可能重复出现，按 row_id 去重。"""
    if os.path.isfile(SAMPLE_CSV) and os.path.isfile(SAMPLE_META):
        meta = json.load(open(SAMPLE_META, encoding='utf-8'))
        if meta.get('n') == n and meta.get('years') == list(years):
            emit('复用已有样本 %s（%d 点）' % (SAMPLE_CSV, meta['n_rows']))
            return pd.read_csv(SAMPLE_CSV), meta
    parts = []
    for y in years:
        fp = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类',
                          'r7_train_%d.parquet' % y)
        if not os.path.isfile(fp):
            fp = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集',
                              'r7_train_%d.parquet' % y)
        if not os.path.isfile(fp):
            emit('缺 %d 年子集，跳过' % y); continue
        parts.append(pd.read_parquet(fp, columns=['row_id', 'class_new'] + C.FEATS))
    df = pd.concat(parts, ignore_index=True).drop_duplicates('row_id')
    emit('合并 %d 年：%d 点（去重后 %d）' % (len(years), sum(len(p) for p in parts), len(df)))
    cnt = df.class_new.value_counts()
    p = cnt / cnt.sum()
    w = (p ** 0.5)
    w = w / w.sum()
    per = (w * n).round().astype(int).clip(lower=150)
    rng = np.random.default_rng(seed)
    keep = []
    for c, k in per.items():
        g = df[df.class_new == c]
        if len(g) == 0:
            continue
        keep.append(df.loc[g.index[rng.choice(len(g), min(len(g), int(k)), replace=False)]])
    s = pd.concat(keep, ignore_index=True)
    for f in C.FEATS:
        s[f] = s[f].round(3)
    s.to_csv(SAMPLE_CSV, index=False)
    meta = {'n': n, 'years': list(years), 'n_rows': int(len(s)), 'shard': SHARD,
            'alloc': 'sqrt', 'per_class': {int(k): int(v) for k, v in
                                           s.class_new.value_counts().items()}}
    json.dump(meta, open(SAMPLE_META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('sqrt 样本 %d 点 → %s' % (len(s), SAMPLE_CSV))
    return s, meta


def upload(s, meta, primary=PRIMARY):
    ee = boot(primary)
    pid = C.ACCOUNTS[primary]['proj']
    root = 'projects/%s/assets/%s' % (pid, C.ACCOUNTS[primary]['folder'])
    try:
        ee.data.getAsset(root)
    except Exception:
        try:
            ee.data.createAsset({'type': 'Folder'}, root)
            emit('建资产根 %s' % root)
        except Exception as e:
            emit('建根: %s' % str(e)[:90])
    parts = [s.iloc[i:i + SHARD] for i in range(0, len(s), SHARD)]
    emit('分片上传 %d 片（每片 ≤%d 点）' % (len(parts), SHARD))
    tasks = []
    for k, ch in enumerate(parts):
        aid = '%s/prodsample_v1_%03d' % (root, k)
        try:
            ee.data.deleteAsset(aid)
        except Exception:
            pass
        recs = [ee.Feature(None, {**{f: float(getattr(r, f)) for f in C.FEATS},
                                  'cl': int(r.class_new), 'lon': float(r.lon),
                                  'lat': float(r.lat)}) for r in ch.itertuples()]
        t = ee.batch.Export.table.toAsset(
            collection=ee.FeatureCollection(recs),
            description='lcsamp_v1_%03d' % k, assetId=aid)
        t.start()
        tasks.append({'id': t.id, 'asset': aid, 'n': int(len(ch))})
        emit('  片 %03d：%d 点 → %s' % (k, len(ch), t.id))
        time.sleep(2)
    done = {}
    for _ in range(240):
        time.sleep(20)
        done = {x['id']: ee.data.getTaskStatus(x['id'])[0]['state'] for x in tasks}
        n_ok = sum(1 for v in done.values() if v == 'COMPLETED')
        n_bad = sum(1 for v in done.values() if v in ('FAILED', 'CANCELLED'))
        emit('  进度 %d/%d（失败 %d）' % (n_ok, len(tasks), n_bad))
        if n_ok + n_bad >= len(tasks):
            break
    ok = [x for x in tasks if done.get(x['id']) == 'COMPLETED']
    emit('上传完成 %d/%d' % (len(ok), len(tasks)))
    return ok, pid, root


def share(assets, primary=PRIMARY, targets=None):
    email = email_of(primary)
    others = [a for a in (targets or C.usable_accounts()) if a != primary]
    em = {}
    for a in others:
        e = email_of(a)
        if e:
            em[a] = e
        else:
            emit('  %s 取不到邮箱，跳过共享' % a)
    if not em:
        return {}
    ee = boot(primary)
    for aid in assets:
        try:
            ee.data.setAssetAcl(aid, {'readers': ['user:' + e for e in em.values()]})
        except Exception as e:
            emit('  ACL 失败 %s: %s' % (aid.split('/')[-1], str(e)[:90]))
    emit('已把 %d 个资产共享给 %d 个账号' % (len(assets), len(em)))
    return em


def write_deploy(assets, em, primary=PRIMARY):
    n_trees, max_nodes = 150, 200000
    common = {'mode': 'asset', 'assets': [a['asset'] for a in assets],
              'n_trees': n_trees, 'min_leaf': 2, 'max_nodes': max_nodes,
              'label_col': 'cl', 'seed': 42, 'owner': primary, 'tag': 'v1'}
    for acct in C.usable_accounts():
        d = dict(common)
        if acct != primary:
            if acct not in em:
                continue
            d['reader_email'] = em[acct]
        else:
            d['reader_email'] = email_of(primary)
        fp = os.path.join(C.PLAN, 'deploy_%s.json' % acct)
        json.dump(d, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        emit('部署件 → %s' % fp)
    return common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--years', type=int, nargs='*', default=C.YEARS)
    ap.add_argument('--make-sample', action='store_true')
    ap.add_argument('--upload', action='store_true')
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--deploy-json', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--primary', default=PRIMARY)
    a = ap.parse_args()
    C.ensure_dirs()
    s, meta = make_sample(a.n, a.years)
    if a.make_sample and not (a.all or a.upload):
        print('样本就绪：', SAMPLE_CSV); return 0
    assets, pid, root = upload(s, meta, a.primary)
    em = {}
    if a.share or a.all:
        em = share([x['asset'] for x in assets], a.primary)
    write_deploy(assets, em, a.primary)
    json.dump({'primary': a.primary, 'pid': pid, 'root': root, 'n': len(s),
               'n_shards': len(assets), 'assets': assets, 'shared_to': em,
               'sample_meta': meta},
              open(os.path.join(C.PLAN, 'deploy_assets.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('部署完成，资产 %d 个，共享 %d 账号' % (len(assets), len(em)))


if __name__ == '__main__':
    raise SystemExit(main())
