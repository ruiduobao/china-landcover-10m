# -*- coding: utf-8 -*-
"""h0c_probe_accounts.py — 账号可用性实测（ee.Initialize+getInfo），产出可用编队

* 账号来源：config/pids_mined.json；排除 accounts_pool.json 的 bad 名单
* 每个账号：apply_account_env + ee.Initialize(project=第一个项目) + ee.Number(1).getInfo()
* 输出：config/accounts_good.json（[{acct,pid,secs}]，按可用顺序）
* 用法：python h0c_probe_accounts.py [--need 30] [--skip a,b,...]
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')
BASE = r'F:/lc_work/bare_sparse_v3'
PIDS = json.load(open(os.path.join(BASE, 'config', 'pids_mined.json'), encoding='utf-8'))
POOL = json.load(open(os.path.join(r'F:/lc_work/v31_exp/config', 'accounts_pool.json'), encoding='utf-8'))
BAD = set(POOL.get('bad', [])) | {'znojnja', '272ogw8s', '3qb2r4', 'xqszikht', '8pesstfu',
                                   'ay3pbvj', 'cpwllrv', 'kqxk70qe', '5rqs3xw2', '679i9zo111222'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--need', type=int, default=30)
    ap.add_argument('--skip', default='')
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(',') if x)
    from prod_conf import apply_account_env
    import ee
    good = []
    for acct in sorted(PIDS):
        if not PIDS[acct] or acct in BAD or acct in skip:
            continue
        pid = PIDS[acct][0]
        t0 = time.time()
        try:
            apply_account_env(acct)
            ee.Initialize(project=pid)
            v = ee.Number(1).add(1).getInfo()
            assert v == 2
            good.append({'acct': acct, 'pid': pid, 'secs': round(time.time() - t0, 1)})
            print('[%2d] %-16s %-32s OK %.1fs' % (len(good), acct, pid, time.time() - t0), flush=True)
        except Exception as e:
            print('     %-16s %-32s FAIL %s' % (acct, pid, str(e)[:70]), flush=True)
        if len(good) >= a.need:
            break
    json.dump(good, open(os.path.join(BASE, 'config', 'accounts_good.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('可用 %d 个 → config/accounts_good.json' % len(good), flush=True)


if __name__ == '__main__':
    main()
