# -*- coding: utf-8 -*-
"""
p1c_merge_sample.py — 把多片样本在服务端合并成**单资产**（一次性，之后训练只读一个资产）
为什么：worker 里 29 层链式 merge 会让分类任务的计算图极重，实测首瓦片报
`Image.classify: Computed value is too large.`；且 merge 是 O(n²) 评估，交互式都跑不动。
做法：读 plan/deploy_<acct>.json 的 assets → 服务端 merge → Export.table.toAsset 到
      smp_v1_merged → 等完成 → **改写 deploy_<acct>.json**（assets=[单资产] + 新模型超参）。
幂等：已存在同名资产先删。
用法：python p1c_merge_sample.py --acct ief3nj [--max-nodes 20000]
"""
import os, sys, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


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
    ap.add_argument('--max-nodes', type=int, default=20000)
    ap.add_argument('--n-trees', type=int, default=150)
    ap.add_argument('--min-leaf', type=int, default=2)
    ap.add_argument('--tag', default='merged')
    a = ap.parse_args()
    acct = a.acct
    dep_fp = os.path.join(C.PLAN, 'deploy_%s.json' % acct)
    d = json.load(open(dep_fp, encoding='utf-8'))
    ids = d.get('assets') or []
    if len(ids) <= 1:
        emit('%s 只有 %d 片，无需合并' % (acct, len(ids)))
        ids = ids or []
    pid = C.ACCOUNTS[acct]['proj']
    root = 'projects/%s/assets/%s' % (pid, C.ACCOUNTS[acct]['folder'])
    ee = boot(acct)
    merged = '%s/smp_%s' % (root, a.tag)
    try:
        ee.data.deleteAsset(merged)
        emit('删除旧合并资产')
    except Exception:
        pass
    fc = ee.FeatureCollection(ids[0])
    for _a in ids[1:]:
        fc = fc.merge(ee.FeatureCollection(_a))
    t0 = time.time()
    t = ee.batch.Export.table.toAsset(collection=fc, description='lcmerge_%s' % acct[:4],
                                      assetId=merged)
    t.start()
    emit('%s 合并导出已提交 %s（%d 片 → 1）' % (acct, t.id, len(ids)))
    for _ in range(200):
        time.sleep(30)
        st = ee.data.getTaskStatus(t.id)[0]
        s = st.get('state')
        emit('  %s' % s)
        if s in ('COMPLETED', 'FAILED', 'CANCELLED'):
            if s != 'COMPLETED':
                emit('失败: %s' % st.get('error_message', '')[:140])
                raise SystemExit(2)
            break
    try:
        info = ee.data.getAsset(merged)
        emit('合并资产 %s（%s bytes）' % (merged, info.get('sizeBytes')))
    except Exception as e:
        emit('读资产失败: %s' % str(e)[:90])
    d['assets'] = [merged]
    d['n_trees'] = a.n_trees
    d['max_nodes'] = a.max_nodes
    d['min_leaf'] = a.min_leaf
    d['merge_tag'] = a.tag
    json.dump(d, open(dep_fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('部署件已改写 → %s（单资产 + maxNodes=%d + 树%d）' % (dep_fp, a.max_nodes, a.n_trees))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
