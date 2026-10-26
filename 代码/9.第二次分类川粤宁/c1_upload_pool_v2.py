# -*- coding: utf-8 -*-
"""c1_upload_pool_v2.py — 第二次分类·池上传：train2023_v2 分片上传 + 两级链式 merge（幂等/断点续跑）

* 输入：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（1,764,020 点 / 23 类 / AEF64 零缺失）
* 规则源：q2_upload_shards.py 同款机制（3000 点/片；30 片/组；组→账号轮值；组内链式 merge；
  组资产 ACL 共享宿主后宿主链式 merge）；账号与宿主从 upload_cfg.json 读（c0_health 结果落定后填）
* 门槛：每账号同时任务 ≤2（qdepth<2 才提交）；上传阶段 EECU 预算 ≤50 h；组链深 ≤30
* 资产命名（与第一次 prod5p 命名空间隔离）：projects/<pid>/assets/prod2nd/smp2023v2_shard_%04d
  → smp2023v2_grp_%02d → 宿主 smp2023v2_merged
* 属性 schema（新池无 row_id/qc_status）：pid(str)/cl(int)/class(int)/src(str) + A00..A63（round 6）
* 状态：F:/lc_work/prod2nd_2023/data/upload_state_v2.json；EECU 台账 ledger/eecu_ledger_2nd.csv
* 用法：python 代码/9.第二次分类川粤宁/c1_upload_pool_v2.py auto
       或单步：submit|poll|grp|final|status|verify
幂等：资产存在即跳过；FAILED 自动重投；断点续跑=重跑 auto。
"""
import os, sys, time, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
import pandas as pd

SRC = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
ROOT = r'F:/lc_work/prod2nd_2023'
CFG_FP = os.path.join(ROOT, 'upload_cfg.json')
STATE_FP = os.path.join(ROOT, 'data', 'upload_state_v2.json')
LEDGER = os.path.join(ROOT, 'ledger', 'eecu_ledger_2nd.csv')
SHARD = 3000
GROUP = 30
BUDGET_PASS = 40
QCAP = 3          # 上传阶段并发放宽（实测 2 时 22 min/片；3k 点轻任务，q≤3 安全）
EECU_CAP = 50.0
DESC_TAG = 'p2sh'
GRP_TAG = 'p2grp'
FOLDER = 'prod2nd'

_CFG = None


def cfg():
    global _CFG
    if _CFG is None:
        _CFG = VC.jload(CFG_FP)
        if not _CFG or not _CFG.get('host') or not _CFG.get('accounts'):
            raise SystemExit('NEED_MANUAL: 先写 %s（host + accounts 列表，来自 c0_health）' % CFG_FP)
    return _CFG


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def n_shards():
    import pyarrow.parquet as pq
    return math.ceil(pq.ParquetFile(SRC).metadata.num_rows / SHARD)


def groups():
    ACCTS = cfg()['accounts']
    ns = n_shards()
    ng = math.ceil(ns / GROUP)
    return [(g, ACCTS[g % len(ACCTS)], list(range(g * GROUP, min((g + 1) * GROUP, ns))))
            for g in range(ng)]


def root_of(acct):
    return 'projects/%s/assets/%s' % (VC.pid_of(acct), FOLDER)


def shard_asset(j):
    g = j // GROUP
    acct = cfg()['accounts'][g % len(cfg()['accounts'])]
    return acct, '%s/smp2023v2_shard_%04d' % (root_of(acct), j)


def grp_asset(g):
    acct = cfg()['accounts'][g % len(cfg()['accounts'])]
    return acct, '%s/smp2023v2_grp_%02d' % (root_of(acct), g)


def state():
    return VC.jload(STATE_FP, {'tasks': {}, 'merged': None})


def save(st):
    st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    VC.jsave(st, STATE_FP)


def ledger(acct, task_id, desc, eecu_h):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, 'a', encoding='utf-8') as f:
        if new:
            f.write('time,acct,task_id,desc,eecu_h\n')
        f.write('%s,%s,%s,%s,%.4f\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), acct, task_id, desc, eecu_h))


def eecu_used(st):
    return sum(t.get('eecu', 0.0) for t in st['tasks'].values())


def ensure_folder(acct):
    import ee
    VC.ensure_ctx(acct)
    try:
        ee.data.createFolder(root_of(acct))
    except Exception:
        pass


def have_of(acct):
    have = VC.list_assets(acct, root_of(acct))
    if have is None:
        ensure_folder(acct)
        have = VC.list_assets(acct, root_of(acct))
    return have


_DF = None


def load_df():
    global _DF
    if _DF is None:
        emit('加载 %s …' % SRC)
        _DF = pd.read_parquet(SRC)
        emit('loaded %d 点' % len(_DF))
    return _DF


def feats_of(d):
    """3000 行 → ee.Feature 列表。schema：geometry=经纬度；属性 pid/cl/class/src + A00–A63（标量）。
    载荷纪律（沿用第一次实测）：3000 行×68 标量 + AEF round(.,6) ≈ 8MB < 10MB 单请求上限。"""
    import ee
    recs = []
    for r in d.to_dict('records'):
        props = {'pid': str(r['point_id']), 'cl': int(r['cl']), 'class': int(r['cl']),
                 'src': str(r['src'])}
        props.update({f: round(float(r[f]), 6) for f in VC.FEATS})
        recs.append(ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]), props))
    return recs


def submit():
    ACCTS = cfg()['accounts']
    st = state()
    have = {a: have_of(a) for a in ACCTS}
    skip = set()
    df = load_df()
    ns = n_shards()
    submitted = 0
    import ee
    for j in range(ns):
        if submitted >= BUDGET_PASS:
            break
        acct, aid = shard_asset(j)
        if acct in skip or have.get(acct) is None:
            continue
        name = aid.split('/')[-1]
        t_rec = st['tasks'].get(aid)
        if name in have[acct]:
            continue
        if t_rec and t_rec.get('state') not in ('FAILED', 'CANCELLED', None):
            continue
        n_q, err = VC.qdepth(acct)
        if n_q is None or n_q >= QCAP:
            skip.add(acct)
            continue
        if eecu_used(st) > EECU_CAP:
            save(st)
            raise SystemExit('NEED_MANUAL: EECU %.1f 超预算 %.0f' % (eecu_used(st), EECU_CAP))
        VC.ensure_ctx(acct)
        d = df.iloc[j * SHARD:(j + 1) * SHARD]
        desc = '%s_%04d_%s' % (DESC_TAG, j, time.strftime('%m%d%H%M%S'))
        t = ee.batch.Export.table.toAsset(
            collection=ee.FeatureCollection(feats_of(d)), description=desc, assetId=aid)
        t.start()
        st['tasks'][aid] = dict(kind='shard', shard=j, group=j // GROUP, acct=acct,
                                task=t.id, desc=desc, state='SUBMITTED', eecu=0.0)
        submitted += 1
        emit('提交 shard %04d @%s（q=%d）task=%s' % (j, acct, n_q, t.id))
        time.sleep(1.5)
    save(st)
    emit('本轮提交 %d 片；状态表 %d 项；EECU %.2f h' % (submitted, len(st['tasks']), eecu_used(st)))
    return submitted


def poll():
    st = state()
    TERMINAL = ('COMPLETED', 'SUCCEEDED', 'FAILED', 'CANCELLED')
    active = {aid: t for aid, t in st['tasks'].items()
              if t.get('state') not in TERMINAL}
    if not active:
        return
    by_acct = {}
    for aid, t in active.items():
        by_acct.setdefault(t['acct'], []).append((aid, t))
    changed = 0
    for acct, items in by_acct.items():
        ops = _ops_map(acct)
        for aid, t in items:
            info = ops.get(str(t['task']))
            sec = None
            if info:
                md = info.get('metadata', {}) or {}
                ns = md.get('state')
                sec = md.get('batchEecuUsageSeconds')
            else:
                ns, sec = _task_status_eecu(acct, t['task'])
            if ns and ns != t.get('state'):
                t['state'] = ns
                changed += 1
            if sec and not t.get('done_eecu'):
                t['eecu'] = round(float(sec) / 3600.0, 4)
                t['done_eecu'] = True
                ledger(acct, str(t['task']), t['desc'], t['eecu'])
        time.sleep(0.5)
    if changed:
        save(st)
        done = sum(1 for t in st['tasks'].values() if t['state'] == 'COMPLETED')
        emit('poll：状态变化 %d；完成 %d/%d；EECU %.2f h' % (changed, done, len(st['tasks']), eecu_used(st)))


def _ops_map(acct):
    try:
        s, pid = VC.sess(acct)
        r = s.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                  params={'pageSize': 200}, headers={'X-Goog-User-Project': pid}, timeout=90)
        r.raise_for_status()
        return {o['name'].rsplit('/', 1)[-1]: o for o in r.json().get('operations', [])}
    except Exception as e:
        emit('ops %s 失败: %s' % (acct, str(e)[:80]))
        return {}


def _task_status_eecu(acct, tid):
    try:
        import ee
        VC.ensure_ctx(acct)
        ts = ee.data.getTaskStatus(tid)
        if ts:
            d = ts[0]
            return d.get('state'), d.get('batch_eecu_usage_seconds')
    except Exception:
        pass
    return None, None


def grp():
    st = state()
    submitted = 0
    import ee
    for g, acct, shards in groups():
        _, gaid = grp_asset(g)
        gname = gaid.split('/')[-1]
        rec = st['tasks'].get(gaid + '#grp')
        if rec and rec.get('state') not in ('FAILED', 'CANCELLED', None):
            continue
        names = have_of(acct)
        if names is None:
            continue
        if gname in names:
            continue
        shard_names = ['smp2023v2_shard_%04d' % j for j in shards]
        if not all(s in names for s in shard_names):
            continue
        n_q, _ = VC.qdepth(acct)
        if n_q is None or n_q >= QCAP:
            continue
        VC.ensure_ctx(acct)
        fc = ee.FeatureCollection('%s/%s' % (root_of(acct), shard_names[0]))
        for s in shard_names[1:]:
            fc = fc.merge(ee.FeatureCollection('%s/%s' % (root_of(acct), s)))
        desc = '%s_%02d_%s' % (GRP_TAG, g, time.strftime('%m%d%H%M%S'))
        t = ee.batch.Export.table.toAsset(collection=fc, description=desc, assetId=gaid)
        t.start()
        st['tasks'][gaid + '#grp'] = dict(kind='grp', group=g, acct=acct, task=t.id,
                                          desc=desc, state='SUBMITTED', eecu=0.0)
        save(st)
        submitted += 1
        emit('组%02d 链式merge 提交 @%s（%d 片）task=%s' % (g, acct, len(shards), t.id))
        time.sleep(2)
    return submitted


def final():
    HOST = cfg()['host']
    st = state()
    merged_id = '%s/smp2023v2_merged' % root_of(HOST)
    have_h = have_of(HOST)
    if have_h and 'smp2023v2_merged' in have_h:
        st['merged'] = dict(asset=merged_id, state='COMPLETED')
        save(st)
        return 'merged 已存在'
    grp_ids = []
    for g, acct, _s in groups():
        _, gaid = grp_asset(g)
        names = have_of(acct)
        if names is None:
            return 'listAssets 失败 %s' % acct
        if gaid.split('/')[-1] not in names:
            return '组 %02d 尚未生成' % g
        grp_ids.append((acct, gaid))
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    host_email = 'user:' + pool['emails'][HOST]
    done_acl = VC.jload(os.path.join(ROOT, 'data', 'acl_done_v2.json'), {})
    for acct, gaid in grp_ids:
        if done_acl.get(gaid):
            continue
        e2, _ = VC.ctx(acct)
        try:
            e2.data.setAssetAcl(gaid, {'readers': [host_email]})
            done_acl[gaid] = time.strftime('%Y-%m-%d %H:%M')
            VC.jsave(done_acl, os.path.join(ROOT, 'data', 'acl_done_v2.json'))
            emit('ACL %s → %s ✓' % (gaid.split('/')[-1], HOST))
        except Exception as e:
            emit('ACL 失败 %s: %s' % (gaid, str(e)[:100]))
            return 'ACL 失败'
    rec = st['tasks'].get(merged_id)
    if rec and rec.get('state') not in ('FAILED', 'CANCELLED', None):
        return 'final merge 已提交'
    import ee
    VC.ensure_ctx(HOST)
    fc = ee.FeatureCollection(grp_ids[0][1])
    for _a, gaid in grp_ids[1:]:
        fc = fc.merge(ee.FeatureCollection(gaid))
    desc = 'p2mrg_%s' % time.strftime('%m%d%H%M%S')
    t = ee.batch.Export.table.toAsset(collection=fc, description=desc, assetId=merged_id)
    t.start()
    st['tasks'][merged_id] = dict(kind='final', acct=HOST, task=t.id, desc=desc,
                                  state='SUBMITTED', eecu=0.0)
    st['merged'] = dict(asset=merged_id, state='SUBMITTED', task=t.id)
    save(st)
    emit('最终合并提交 %s → %s' % (t.id, merged_id))
    return 'final 提交'


def status():
    st = state()
    kinds = {}
    for t in st['tasks'].values():
        k = (t['kind'], t['state'])
        kinds[k] = kinds.get(k, 0) + 1
    emit('分片 %d（%d 组）/ %s / EECU %.2f h / merged=%s' % (
        n_shards(), len(groups()),
        sorted('%s:%s:%d' % (k[0], k[1], v) for k, v in kinds.items()),
        eecu_used(st), (st.get('merged') or {}).get('state')))


def verify():
    import ee
    HOST = cfg()['host']
    st = state()
    mid = (st.get('merged') or {}).get('asset')
    if not mid:
        raise SystemExit('merged 未生成')
    VC.ensure_ctx(HOST)
    fc = ee.FeatureCollection(mid)
    n = fc.size().getInfo()
    one = fc.limit(1).first().getInfo()
    props = one.get('properties', {})
    need = ['pid', 'cl', 'class', 'src'] + VC.FEATS[:3]
    miss = [k for k in need if k not in props]
    emit('verify：%s 行数=%d（期望 1764020）缺属性=%s（经纬度在 geometry）' % (mid, n, miss or '无'))
    assert n == 1764020, '点数不符：%d' % n
    return n


def auto():
    t0 = time.time()
    while True:
        try:
            submit()
            poll()
            grp()
            r = final()
            poll()
            status()
            st = state()
            if (st.get('merged') or {}).get('state') == 'COMPLETED' or r == 'merged 已存在':
                emit('✅ merged 就绪，驱动退出')
                return
        except SystemExit as e:
            emit(str(e))
            return
        except Exception as e:
            emit('auto 异常（继续）: %s' % str(e)[:160])
            time.sleep(30)
        if time.time() - t0 > 8 * 3600:
            emit('8h failsafe 退出')
            return
        time.sleep(45)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    dict(submit=submit, poll=poll, grp=grp, final=final,
         status=status, verify=verify, auto=auto)[cmd]()
