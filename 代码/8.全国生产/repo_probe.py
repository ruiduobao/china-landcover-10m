# -*- coding: utf-8 -*-
"""repo_probe.py — 为三个账号各选一个「EE 已注册、可用」的仓库（按登记表顺序试到通为止）"""
import os, sys, io, re, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C
CAND = {
    '1yxth5g': ['vast-logic-508309-g7', 'round-concept-508309-h8', 'elated-chassis-508309-u6'],
    '11e0n00': ['lustrous-vial-508409-q1', 'rock-sentinel-508409-s9', 'resolute-land-508409-h6'],
    'ief3nj':  ['temporal-web-508410-h0', 'cellular-client-508410-t6', 'stellar-stream-508410-a4',
                'speedy-unison-508410-k9', 'inlaid-marker-508410-a9', 'mythic-veld-508410-r9',
                'esoteric-grove-508410-s8', 'magnetic-runway-508410-i9'],
}
out = {}
for acct, repos in CAND.items():
    C.apply_account_env(acct)
    import ee
    got = None
    for pid in repos:
        try:
            ee.Initialize(project=pid)
            n = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').size().getInfo()
            print('%-10s %-28s ✅ 可用（AEF %d 景）' % (acct, pid, n))
            got = pid
            break
        except Exception as e:
            print('%-10s %-28s ❌ %s' % (acct, pid, str(e)[:70]))
            time.sleep(3)
    out[acct] = got
json.dump(out, open(os.path.join(C.PLAN, 'repos_ne2023.json'), 'w', encoding='utf-8'), indent=1)
print('\n选定:', out)
