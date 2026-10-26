# -*- coding: utf-8 -*-
"""h0_health_all.py — 全账号项目级健康检查（串行；结果写独立文件，不动 accounts_pool.json）

* 账号来源：v31_exp/config/accounts_pool.json 的 emails（97 个凭证有效账号）
* 检查：ee.Initialize(project) + listAssets + 队列（同 s0b_health.check，串行避免串号）
* 输出：F:/lc_work/bare_sparse_v3/logs/health_all.json  {healthy:[...], rows:[...]}
* 用法：python h0_health_all.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import s0b_health as H  # noqa
import v31_common as VC  # noqa

POOL = os.path.join(VC.CFG, 'accounts_pool.json')
OUT = r'F:/lc_work/bare_sparse_v3/logs/health_all.json'


def main():
    pool = json.load(open(POOL, encoding='utf-8'))
    accts = sorted(set(pool.get('emails', {}).keys()) | set(pool.get('resolvable', {}).keys()))
    print('候选账号 %d' % len(accts), flush=True)
    rows = []
    t0 = time.time()
    for i, a in enumerate(accts):
        try:
            r = H.check(a)
        except Exception as e:
            r = dict(acct=a, ok=False, reason='异常:' + str(e)[:80], qdepth=None, n_assets=None)
        rows.append(r)
        print('[%3d/%3d] %-14s %s %s' % (i + 1, len(accts), a, 'OK' if r['ok'] else 'X',
                                         (r.get('reason') or '')[:60]), flush=True)
    healthy = [r['acct'] for r in rows if r['ok']]
    json.dump({'time': time.strftime('%Y-%m-%d %H:%M'), 'rows': rows, 'healthy': healthy},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('健康 %d/%d  (%.0fs) → %s' % (len(healthy), len(rows), time.time() - t0, OUT), flush=True)


if __name__ == '__main__':
    main()
