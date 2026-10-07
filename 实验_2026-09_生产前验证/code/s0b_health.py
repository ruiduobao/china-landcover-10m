# -*- coding: utf-8 -*-
"""s0b_health.py — 项目级健康检查（零 EECU）：凭证有效之后，再确认"这个账号的 GEE 仓库真的能用"
三项：① ee.Initialize 成功（含 restricted-mode 告警捕获）② listAssets 能列自己的仓 ③ 队列可查询
产物：config/accounts_pool.json 的 healthy 列表（workers 只保留 healthy）
用法：python s0b_health.py [--keep als74akz,...]
"""
import os, sys, time, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

POOL = os.path.join(VC.CFG, 'accounts_pool.json')


def check(acct):
    """返回 dict(acct, ok, reason, qdepth, n_assets)。整个流程都在 warnings 捕获内——
    受限模式告警可能在 Initialize 之后的任意网络调用里抛出。"""
    import ee
    rec = dict(acct=acct, ok=False, reason='', qdepth=None, n_assets=None)
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            VC.C.apply_account_env(acct)
            pid = VC.pid_of(acct)
            ee.Initialize(project=pid)
            VC._CUR_ACCT['acct'] = acct
            r = ee.data.listAssets({'parent': 'projects/%s/assets' % pid})
            n_assets = len(r.get('assets', []))
            n_q, err = VC.qdepth(acct)
            msgs = ' '.join(str(w.message) for w in ws)
        rec['n_assets'] = n_assets
        if 'restricted mode' in msgs or 'exceeded its noncommercial compute quota' in msgs:
            rec['reason'] = 'restricted-mode（非商业配额受限）'
            return rec
        if n_q is None:
            rec['reason'] = '队列查询失败: %s' % err
            return rec
        rec['qdepth'] = n_q
        rec['ok'] = True
        rec['reason'] = 'OK（%d 资产 / 队列 %d）' % (n_assets, n_q)
    except Exception as e:
        rec['reason'] = str(e)[:110]
    return rec


def main():
    pool = VC.jload(POOL, {})
    if not pool:
        raise SystemExit('NEED_MANUAL: accounts_pool.json 缺失（先跑 s0_sweep）')
    keep = []
    if '--keep' in sys.argv:
        keep = sys.argv[sys.argv.index('--keep') + 1].split(',')
    cands = [pool['uploader']] + [w for w in pool['workers']] + ([pool['owner']] if pool.get('owner') else [])
    cands = [a for a in dict.fromkeys(cands) if a]          # 去重保序
    VC.emit('健康检查 %d 个账号（零 EECU）' % len(cands))
    rows = []
    for i, a in enumerate(cands, 1):
        r = check(a)
        rows.append(r)
        VC.emit('[%2d/%d] %-18s %s %s' % (i, len(cands), a, '✅' if r['ok'] else '❌', r['reason']))
    healthy = [r['acct'] for r in rows if r['ok']]
    bad = {r['acct']: r['reason'] for r in rows if not r['ok']}
    pool['health'] = dict(time=time.strftime('%Y-%m-%d %H:%M'), rows=rows, bad=bad)
    pool['healthy'] = healthy
    up = pool['uploader'] if pool['uploader'] in healthy else next(
        (a for a in healthy if a != pool.get('owner')), None)
    if up is None:
        raise SystemExit('NEED_MANUAL: 无任何健康账号可用')
    pool['uploader'] = up
    pool['workers'] = [a for a in healthy if a != pool.get('owner') and a != up] + \
                      [a for a in keep if a not in healthy and a != pool.get('owner')]
    VC.jsave(pool, POOL)
    VC.emit('健康 %d/%d ｜ uploader=%s ｜ workers=%d ｜ 剔除: %s' % (
        len(healthy), len(cands), up, len(pool['workers']),
        ', '.join('%s(%s)' % (k, v[:28]) for k, v in bad.items())))
    return pool


if __name__ == '__main__':
    main()
