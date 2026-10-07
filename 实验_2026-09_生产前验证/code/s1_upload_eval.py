# -*- coding: utf-8 -*-
"""s1_upload_eval.py — 留出评估点上传为 GEE 表资产（w1-w4+w5）+ 合并 + ACL 共享给 workers
设计：
  · 评估点本地 parquet（防泄漏从未进训练）→ 3000点/片 上传到 uploader 名下 v31e_<w>_<iii>
  · 全部片 SUCCEEDED（=资产存在）后链式 merge 成 v31e_<w>_merged（幂等，存在即跳过）
  · setAssetAcl 把 评估合并资产 + owner 的 smp_w*_merged 窗训练资产 共享只读给全部 workers
  · 非阻塞：本次提交不完等，资产出现与否由 listAssets 判断，advance 轮询推进
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd

SHARD = 3000
WIN = VC.cfg('windows.json')['windows']


SHARD = 3000
WIN = VC.cfg('windows.json')['windows']
STALL_MIN = 40  # v31_ 任务 PENDING 超过此值 → uploader 项目疑似配额卡死，轮换


def _age_min(md):
    from datetime import datetime, timezone
    ct = md.get('createTime')
    if not ct:
        return 0.0
    try:
        dt = datetime.fromisoformat(ct.replace('Z', '+00:00'))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return 0.0


def uploader_stuck(acct):
    """该账号名下 v31_ 前缀任务是否有超时未调度的（PENDING/RUNNING 且 createTime 陈旧）。"""
    try:
        s, pid = VC.sess(acct)
        r = s.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                  params={'pageSize': 50}, headers={'X-Goog-User-Project': pid}, timeout=60)
        r.raise_for_status()
    except Exception as e:
        VC.emit('uploader 健康检查失败 %s: %s' % (acct, str(e)[:80]))
        return False
    stuck = []
    for o in r.json().get('operations', []):
        md = o.get('metadata', {})
        if not md.get('description', '').startswith('v31'):
            continue
        if md.get('state') in ('PENDING', 'RUNNING') and _age_min(md) > STALL_MIN:
            stuck.append(md['description'][:32])
    if stuck:
        VC.emit('⚠️ uploader=%s 有 %d 个 v31 任务 >%d 分钟未调度（%s…）→ 轮换' % (
            acct, len(stuck), STALL_MIN, stuck[0]))
    return bool(stuck)


def ensure_uploader(pool):
    """uploader 卡死则换下一个活的可解析账号；老 uploader 从 workers 里补回。"""
    up = pool['uploader']
    if not uploader_stuck(up):
        return up
    cands = [a for a in sorted(VC.C.ACCOUNTS)
             + list(VC.jload(os.path.join(VC.EVAL_DIR_GEE, 'extra_accounts.json'), {}))
             if a in pool.get('emails', {}) and a in pool.get('resolvable', {})
             and a not in (pool.get('owner'), up)]
    for c in cands:
        pool['workers'] = [w for w in pool['workers'] if w != c] + (
            [up] if up not in pool['workers'] else [])
        pool['uploader'] = c
        VC.jsave(pool, os.path.join(VC.CFG, 'accounts_pool.json'))
        VC.emit('uploader %s → %s（已回写 accounts_pool.json）' % (up, c))
        return c
    raise SystemExit('NEED_MANUAL: 无候选 uploader 可轮换')


def eval_fp(w):
    if os.path.exists(os.path.join(VC.DATA, 'eval_%s.parquet' % w)):
        return os.path.join(VC.DATA, 'eval_%s.parquet' % w)
    return os.path.join(VC.EVAL_DIR_GEE, 'eval_%s.parquet' % w)


def main():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    if not pool:
        raise SystemExit('NEED_MANUAL: accounts_pool.json 缺失（先跑 s0）')
    up = ensure_uploader(pool)
    emails = ['user:' + pool['emails'][a] for a in pool['workers'] if a in pool['emails']]
    VC.emit('uploader=%s 共享读者=%d 人' % (up, len(emails)))
    ee, upid = VC.ctx(up)
    up_root = 'projects/%s/assets' % upid
    log = VC.jload(os.path.join(VC.DATA, 'upload_log.json'), {})
    todo_windows = [w for w in ['w1', 'w2', 'w3', 'w4', 'w5']
                    if os.path.exists(eval_fp(w))]
    BUDGET = 24  # 每轮提交上限（片+合并）
    for w in todo_windows:
        have = VC.list_assets(up, up_root)
        if have is None:
            raise SystemExit('NEED_MANUAL: listAssets 失败')
        merged_id = '%s/v31e_%s_merged' % (up_root, w)
        if ('v31e_%s_merged' % w) not in have:
            ev = pd.read_parquet(eval_fp(w))
            n = len(ev)
            n_ch = (n + SHARD - 1) // SHARD
            missing = [i for i in range(n_ch) if ('v31e_%s_%03d' % (w, i)) not in have]
            VC.emit('[%s] 留出 %d 点 → %d 片（待传 %d）' % (w, n, n_ch, len(missing)))
            for i in missing:
                if BUDGET <= 0:
                    break
                n_q, err = VC.qdepth(up)
                if n_q is None or n_q > 6:
                    VC.emit('  队列=%s（%s），换下一窗' % (n_q, err))
                    break
                ch = ev.iloc[i * SHARD:(i + 1) * SHARD]
                recs = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                                   {'rid': int(r['row_id']),
                                    **{f: float(r[f]) for f in VC.FEATS}}) for r in ch.to_dict('records')]
                aid = '%s/v31e_%s_%03d' % (up_root, w, i)
                desc = 'v31e_%s_%s_%03d' % (w, time.strftime('%m%d%H%M%S'), i)
                t = ee.batch.Export.table.toAsset(
                    collection=ee.FeatureCollection(recs), description=desc, assetId=aid)
                t.start()
                log[desc] = dict(kind='eval_shard', w=w, i=i, task=t.id, asset=aid)
                BUDGET -= 1
                VC.emit('  提交 %s 片%03d task=%s' % (w, i, t.id))
                time.sleep(2)
            # 片齐 → 提交合并
            have = VC.list_assets(up, up_root) or set()
            if BUDGET > 0 and ('v31e_%s_merged' % w) not in have and \
                    all(('v31e_%s_%03d' % (w, i)) in have for i in range(n_ch)):
                fc = ee.FeatureCollection('%s/v31e_%s_%03d' % (up_root, w, 0))
                for i in range(1, n_ch):
                    fc = fc.merge(ee.FeatureCollection('%s/v31e_%s_%03d' % (up_root, w, i)))
                desc = 'v31mrg_%s_%s' % (w, time.strftime('%m%d%H%M%S'))
                t = ee.batch.Export.table.toAsset(collection=fc, description=desc, assetId=merged_id)
                t.start()
                log[desc] = dict(kind='eval_merge', w=w, task=t.id, asset=merged_id)
                BUDGET -= 1
                VC.emit('[%s] 合并任务提交 %s' % (w, t.id))
            VC.jsave(log, os.path.join(VC.DATA, 'upload_log.json'))
        else:
            VC.reg_set('eval_%s' % w, merged_id)
            VC.emit('[%s] 评估合并资产已存在，跳过' % w)
    # 合并全部就绪 → 共享（评估资产以注册表为准，可能分散在历任 uploader 名下）
    VC.jsave(log, os.path.join(VC.DATA, 'upload_log.json'))
    share_list = []
    for w in todo_windows:
        full = VC.reg_get('eval_%s' % w) or (
            '%s/v31e_%s_merged' % (up_root, w) if VC.asset_exists(up, '%s/v31e_%s_merged' % (up_root, w)) else None)
        if full:
            share_list.append(full)
    owner = pool['owner']
    if owner:
        opid = VC.pid_of(owner)
        share_list += ['projects/%s/assets/t4ca_exp/smp_%s_merged' % (opid, w)
                       for w in ['w1', 'w2', 'w3', 'w4']]
        w5_train = VC.reg_get('train_w5')
        if w5_train:
            share_list.append(w5_train)
    already = VC.jload(os.path.join(VC.DATA, 'shared_done.json'), {})
    for full in share_list:
        if already.get(full):
            continue
        hold_acct = up if full.startswith(up_root) else owner
        try:
            e2, p2 = VC.ctx(hold_acct)
            e2.data.setAssetAcl(full, {'readers': emails})
            already[full] = time.strftime('%Y-%m-%d %H:%M')
            VC.emit('共享 %s ✓' % full.split('/')[-1])
        except Exception as e:
            VC.emit('共享失败 %s: %s' % (full.split('/')[-1], str(e)[:110]))
    VC.jsave(already, os.path.join(VC.DATA, 'shared_done.json'))
    VC.emit('s1 本轮完成')


if __name__ == '__main__':
    main()
