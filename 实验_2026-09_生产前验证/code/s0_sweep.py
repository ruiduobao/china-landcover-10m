# -*- coding: utf-8 -*-
"""s0_sweep.py — 全量凭证探活 → config/accounts_pool.json（零EECU，轻量OAuth刷新）
产物：活账号清单 + email（Drive about 反查，ACL共享要用）+ 可解析项目ID的角色分配。
角色：owner=zitwwufh(窗样本资产主，已超限→只做资产读/ACL)；uploader=首个活的extra新号(传评估资产)；
      workers=活∩可解析项目，扣除 owner/uploader。
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

CRED_BASE = r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
EXCLUDE = {'znojnja', '272ogw8s', '3qb2r4', 'xqszikht', '8pesstfu', 'ay3pbvj',
           'cpwllrv', 'kqxk70qe', '5rqs3xw2', '679i9zo111222'}  # 已知授权失效（用户侧待重授权）
OUT_FP = os.path.join(VC.CFG, 'accounts_pool.json')


def one(acct):
    t0 = time.time()
    try:
        BASE = CRED_BASE
        os.environ['GEE_PROXY'] = VC.C.PROXY_URL      # 代理端口自动探测（7890/7897 会漂移）
        sys.path.insert(0, BASE)
        import _verify_cred as V
        good, info = V.verify(os.path.join(BASE, acct, '.config', 'earthengine', 'credentials'),
                              proxy=VC.C.PROXY_URL)
    except Exception as e:
        good, info = False, '异常:' + str(e)[:80]
    return dict(acct=acct, ok=bool(good), email=info if good else '',
                info='' if good else str(info)[:100], secs=round(time.time() - t0, 1))


def main(force=False):
    old = VC.jload(OUT_FP, {})
    if old and not force and (time.time() - os.path.getmtime(OUT_FP) < 20 * 3600):
        VC.emit('accounts_pool.json 20小时内已探活（%s），跳过' % old.get('time'))
        return old
    accts = sorted(d for d in os.listdir(CRED_BASE)
                   if os.path.isfile(os.path.join(CRED_BASE, d, '.config', 'earthengine', 'credentials')))
    VC.emit('凭证池 %d 个，开始探活（2线程）' % len(accts))
    rows = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(one, a): a for a in accts}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            if i % 20 == 0:
                VC.emit('  %d/%d' % (i, len(accts)))
    ok = {r['acct']: r['email'] for r in rows if r['ok'] and r['acct'] not in EXCLUDE}
    # 可解析项目ID的账号（prod_conf 20 + extra 6）
    resolvable = {}
    for a in VC.C.ACCOUNTS:
        resolvable[a] = VC.C.ACCOUNTS[a]['proj']
    extra = VC.jload(os.path.join(VC.EVAL_DIR_GEE, 'extra_accounts.json'), {})
    resolvable.update(extra)
    workers = sorted([a for a in resolvable if a in ok])
    owner = 'zitwwufh' if 'zitwwufh' in ok else (workers[0] if workers else None)
    uploader = None
    for a in ['ybzljyma', '5631jn8', 'nnjxsn', 'hj74bml', '744s27z', 'xysongf']:
        if a in ok and a != owner:
            uploader = a
            break
    if uploader is None:
        uploader = next((a for a in workers if a != owner), None)
    wlist = [a for a in workers if a not in (owner, uploader)]
    pool = dict(time=time.strftime('%Y-%m-%d %H:%M'), pool_total=len(accts),
                ok_total=len(ok), bad=sorted(set(a for a in accts if a not in ok) | EXCLUDE),
                emails={r['acct']: r['email'] for r in rows if r['ok']},
                resolvable=resolvable,
                owner=owner, uploader=uploader, workers=wlist, rows=rows)
    VC.jsave(pool, OUT_FP)
    VC.emit('可用 %d ｜ workers=%d（%s…）｜ owner=%s uploader=%s' % (
        len(ok), len(wlist), ' '.join(wlist[:5]), owner, uploader))
    return pool


if __name__ == '__main__':
    main(force='--force' in sys.argv)
