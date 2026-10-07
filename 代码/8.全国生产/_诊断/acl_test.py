# -*- coding: utf-8 -*-
"""acl_test.py — 测"样本资产上传一次 + 跨账号共享"是否可行（省掉每账号重复上传）"""
import os, sys, io, re, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

def email_of(acct):
    p = os.path.join(C.CRED, acct, '_任务登记.md')
    if os.path.isfile(p):
        m = re.search(r'账号：([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})',
                      io.open(p, encoding='utf-8', errors='replace').read())
        if m:
            return m.group(1)
    return None

for a in ['zixen8v8', 'y30b63ye', 'seqsiu', 'nhqz5uj', 'r45smj3u', 'zitwwufh', 'iu5f4z', '1yxth5g', 'hte4021n']:
    print(f'  {a:<14} {email_of(a)}')

ASSET = 'projects/braided-horizon-508210-a5/assets/lcf_a01/samples_5000_000'
reader = email_of('y30b63ye')
C.apply_account_env('zixen8v8')
import ee
ee.Initialize(project=C.ACCOUNTS['zixen8v8']['proj'])
print('\n=== 1) owner 设置 ACL ===')
for acl in [{'readers': ['user:' + reader]}, {'readers': [reader]}]:
    try:
        ee.data.setAssetAcl(ASSET, acl)
        print('  成功:', json.dumps(acl)); used = acl; break
    except Exception as e:
        print('  失败 %s → %s' % (json.dumps(acl), str(e)[:110]))
print('=== 2) 读回 ACL ===')
try:
    print('  ', json.dumps(ee.data.getAssetAcl(ASSET), ensure_ascii=False)[:200])
except Exception as e:
    print('  读 ACL 失败:', str(e)[:110])
