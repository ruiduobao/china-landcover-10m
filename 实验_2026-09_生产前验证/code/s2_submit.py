# -*- coding: utf-8 -*-
"""s2_submit.py — 构建 A/B/D 任务清单并提交（GEE 服务端训练 + clf.classify(留出点FC) 表格导出）
臂（doc37 §6.2，参数= frozen_params.json）：
  A_s{7,11,23}   局部窗×原30类（完整基线，容量对齐 maxNodes=5000）
  B_s{7,11,23}   局部窗×24类（训练端 v31 重标）
  D_g{1..3}_m{0..4}  24类 K=5 集成成员（g 组与 B_s{base} 同族配对）
冒烟门（--smoke-only 或首次自动）：小训练+200点内联分类交互下载，验证
  跨项目资产读/服务端训练/clf.classify(FC)/CSV下载 全链路，失败→NEED_MANUAL。
纪律：每账号 PENDING≤5、每轮提交≤24、desc 带时间戳防幂等锁。
"""
import os, sys, time, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd

P = VC.cfg('frozen_params.json')
WIN = VC.cfg('windows.json')['windows']
NAT = VC.jload(r'F:/lc_work/样本修正/geetest/window_plan.json', {})
W5_PLAN = lambda: VC.jload(os.path.join(VC.DATA, 'w5_plan.json'), {})


def n_train_of(w):
    if w in NAT.get('shards', {}):
        return int(NAT['shards'][w]['n_train'])
    wp = W5_PLAN()
    return int(wp.get('n_train', 0)) if w == 'w5' else 0


def train_asset_of(w, pool):
    if w == 'w5':
        return VC.reg_get('train_w5') or (
            'projects/%s/assets/v31s_w5_merged' % VC.pid_of(pool['uploader']))
    return 'projects/%s/assets/t4ca_exp/smp_%s_merged' % (VC.pid_of(pool['owner']), w)


def eval_asset_of(w, pool):
    return VC.reg_get('eval_%s' % w) or (
        'projects/%s/assets/v31e_%s_merged' % (VC.pid_of(pool['uploader']), w))


def build_jobs(windows):
    jobs = VC.jload(VC.JOBS_FP, [])
    have = {j['job_id'] for j in jobs}
    seeds = P['seeds']
    for w in windows:
        if n_train_of(w) <= 0:
            continue
        for s in seeds['A']:
            jid = '%s_A_s%02d' % (w, s)
            if jid not in have:
                jobs.append(dict(job_id=jid, window=w, arm='A', seed=s, label_mode='raw30'))
        for s in seeds['B']:
            jid = '%s_B_s%02d' % (w, s)
            if jid not in have:
                jobs.append(dict(job_id=jid, window=w, arm='B', seed=s, label_mode='v31'))
        for g, base in enumerate(seeds['D_groups_base'], 1):
            for m in seeds['D_member_offsets']:
                jid = '%s_D_g%d_m%02d' % (w, g, m)
                if jid not in have:
                    jobs.append(dict(job_id=jid, window=w, arm='D', group=g, member=m,
                                     base=base, label_mode='v31'))
    VC.jsave(jobs, VC.JOBS_FP)
    return jobs


def submit_one(pool, w, job, used=None):
    """返回 attempt dict；失败抛错。C 臂经 job 内 train_asset/eval_asset/feats 覆盖。
    used: {acct: 本轮已用次数} → 优先挑"本轮用得少 + 队列浅"的账号（并行分摊）。"""
    used = used if used is not None else {}
    accts = sorted(pool['workers'], key=lambda a: (used.get(a, 0),))
    last = ''
    chosen = None
    for acct in accts:
        n_q, err = VC.qdepth(acct)
        if n_q is not None and n_q <= 4:
            chosen = acct
            break
        last = 'q=%s %s' % (n_q, err)
    if chosen is None:
        raise RuntimeError('无空闲 worker（%s）' % last)
    acct = chosen
    used[acct] = used.get(acct, 0) + 1
    ee, pid = VC.ctx(acct)
    tr = ee.FeatureCollection(job.get('train_asset') or train_asset_of(w, pool))
    feats = job.get('feats') or VC.FEATS
    roi = ee.Geometry.Rectangle(WIN[w]['bbox'])
    tr = tr.filterBounds(roi.buffer(P['local_margin_deg']))
    if job['label_mode'] == 'v31':
        tr = VC.remap_fc_v31(tr)
    frac = round(min(0.999, P['train_per_model'] / max(n_train_of(w), 1)), 4)
    if job['arm'] == 'D':
        rc_seed = job['base'] + job['member'] * 131
        rf_seed = job['base'] + job['member'] * 17
    else:
        rc_seed = rf_seed = job['seed']
    sub = tr.randomColumn('rr', rc_seed).filter('rr < %s' % frac)
    clf = (ee.Classifier.smileRandomForest(
        numberOfTrees=P['n_trees'], minLeafPopulation=P['min_leaf'],
        maxNodes=P['max_nodes'], seed=rf_seed).train(sub, 'cl', feats))
    out = ee.FeatureCollection(
        job.get('eval_asset') or eval_asset_of(w, pool)).classify(clf)
    ts = time.strftime('%m%d%H%M%S')
    desc = 'v31_%s_%s' % (job['job_id'], ts)
    aid = 'projects/%s/assets/v31o_%s_%s' % (pid, job['job_id'], ts)
    t = ee.batch.Export.table.toAsset(collection=out, description=desc, assetId=aid)
    t.start()
    return dict(desc=desc, acct=acct, task=t.id, asset=aid,
                submitted=time.strftime('%Y-%m-%d %H:%M:%S'), state='SUBMITTED')


def smoke(pool):
    """小模型+200点内联分类交互验证全链路；通过返回 True。"""
    acct = pool['workers'][0]
    w = 'w1'
    VC.emit('冒烟：%s 在 %s 上（小模型 20 树）' % (acct, train_asset_of(w, pool)))
    ee, pid = VC.ctx(acct)
    tr = ee.FeatureCollection(train_asset_of(w, pool)).randomColumn('rr', 7).filter('rr < 0.02')
    clf = ee.Classifier.smileRandomForest(numberOfTrees=20, minLeafPopulation=2,
                                          maxNodes=500, seed=7).train(tr, 'cl', VC.FEATS)
    ev = pd.read_parquet(eval_fp_local(w)).head(200)
    recs = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                       {'rid': int(r['row_id']), **{f: float(r[f]) for f in VC.FEATS}})
            for r in ev.to_dict('records')]
    out = ee.FeatureCollection(recs).classify(clf)
    url = out.getDownloadURL(filetype='csv', selectors=['rid', 'classification'])
    import requests
    import prod_conf as PC
    r = requests.get(url, proxies=PC.PROXY, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    ok = (len(df) >= 180) and df['classification'].notna().all()
    VC.emit('冒烟%s：%d/200 点返回，预测样例 %s' % (
        '通过✅' if ok else '异常❌', len(df), df['classification'].head(5).tolist()))
    if not ok:
        VC.emit('预测分布: %s' % df['classification'].value_counts().head(8).to_dict())
    return bool(ok)


def eval_fp_local(w):
    fp = os.path.join(VC.DATA, 'eval_%s.parquet' % w)
    return fp if os.path.exists(fp) else os.path.join(VC.EVAL_DIR_GEE, 'eval_%s.parquet' % w)


def main(smoke_only=False):
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    if not pool:
        raise SystemExit('NEED_MANUAL: accounts_pool.json 缺失')
    st = VC.state()
    if not st.get('smoke_done'):
        if not smoke(pool):
            raise SystemExit('NEED_MANUAL: 冒烟未通过，勿批量提交')
        st['smoke_done'] = time.strftime('%Y-%m-%d %H:%M')
        VC.save_state(st)
        if smoke_only:
            return
    # 可提交窗口 = 评估合并资产已就绪 的窗（注册表优先，其次当前 uploader 名下）
    ur = 'projects/%s/assets' % VC.pid_of(pool['uploader'])
    ready = [w for w in ['w1', 'w2', 'w3', 'w4', 'w5']
             if n_train_of(w) > 0 and
             (VC.reg_get('eval_%s' % w) or VC.asset_exists(pool['uploader'], '%s/v31e_%s_merged' % (ur, w)))]
    VC.emit('评估资产就绪窗口: %s' % ready)
    jobs = build_jobs(ready)
    n_sub = 0
    used = {}   # 本轮各账号已用次数（并行分摊）
    for j in jobs:
        if n_sub >= 24:
            VC.emit('本轮提交达上限 24，余下轮继续')
            break
        atts = j.get('attempts', [])
        if atts and atts[-1].get('state') in ('SUBMITTED', 'RUNNING', 'PENDING'):
            continue
        if atts and sum(1 for a in atts if a.get('state') == 'SUCCEEDED') > 0:
            continue
        if len([a for a in atts]) >= 2:  # 含重试，2 次止步报停
            VC.emit('%s 已 2 次尝试未成，报停' % j['job_id'])
            continue
        try:
            att = submit_one(pool, j['window'], j, used)
        except Exception as e:
            VC.emit('%s 提交失败: %s' % (j['job_id'], str(e)[:110]))
            time.sleep(10)
            continue
        att['state'] = 'SUBMITTED'
        j.setdefault('attempts', []).append(att)
        n_sub += 1
        VC.jsave(jobs, VC.JOBS_FP)
        VC.emit('提交 %s → %s [%s]' % (j['job_id'], att['task'], att['acct']))
        time.sleep(2)
    VC.emit('s2 本轮新提交 %d' % n_sub)


if __name__ == '__main__':
    main(smoke_only='--smoke-only' in sys.argv)
