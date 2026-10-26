# -*- coding: utf-8 -*-
"""c2_prep_prod.py — 第二次分类·生产前置：① 各生产账号建 prod2nd 资产目录 ② 合并池 ACL 共享 + 跨账号可读校验

* 输入：F:/lc_work/prod2nd_2023/config.json（asset/host/fleet）
* 规则源：第一次（q2d_share_merged）ACL 机制；AGENTS.md 纪律 8（用前健康检查）与 9（账号不得干扰）
* 门槛：① 建目录幂等（已存在即跳过）；② merged 就绪时把全部 fleet email 加进 readers，
  再从一个非宿主账号 FeatureCollection(asset).size() 实测可读，点数必须 == 1764020
* 输出：F:/lc_work/prod2nd_2023/reports/prep_prod.json（目录状态 + ACL + 点数）
* 用法：python 代码/9.第二次分类川粤宁/c2_prep_prod.py [folders|acl|all]
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
import ee

ROOT = r'F:/lc_work/prod2nd_2023'
CFG = VC.jload(os.path.join(ROOT, 'config.json'))
POOL = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
OUT = os.path.join(ROOT, 'reports', 'prep_prod.json')
N_EXPECT = 1764020


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def folders():
    res = {}
    for a in CFG['fleet']:
        pid = VC.pid_of(a)
        VC.ensure_ctx(a)
        try:
            ee.data.createFolder('projects/%s/assets/prod2nd' % pid)
            res[a] = 'created'
            emit('%s 建 prod2nd ✓' % a)
        except Exception as e:
            msg = str(e)[:80]
            dup = ('already exists' in msg or 'Cannot overwrite' in msg
                   or 'not unique' in msg.lower())
            res[a] = 'exists' if dup else msg
        time.sleep(0.4)
    return res


def acl():
    asset = CFG['asset']
    host = CFG['host']
    VC.ensure_ctx(host)
    want = [('user:' + POOL['emails'][a]) for a in CFG['fleet'] if a in POOL.get('emails', {})]
    cur = list(ee.data.getAssetAcl(asset).get('readers', []))
    add = [x for x in want if x not in cur]
    if add:
        ee.data.setAssetAcl(asset, {'readers': cur + add})
        emit('ACL 增 %d readers（现共 %d）' % (len(add), len(cur) + len(add)))
    else:
        emit('ACL 已齐（%d readers）' % len(cur))
    # 跨账号可读校验：挑一个非宿主账号
    probe = next(a for a in CFG['fleet'] if a != host)
    VC.ensure_ctx(probe)
    n = ee.FeatureCollection(asset).size().getInfo()
    emit('跨账号校验：%s 读 %s 行数=%d（期望 %d）' % (probe, asset.split('/')[-1], n, N_EXPECT))
    assert n == N_EXPECT, '点数不符 %d' % n
    return dict(readers=len(cur) + len(add), probe=probe, n=n)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    doc = VC.jload(OUT, {})
    doc['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if cmd in ('folders', 'all'):
        doc['folders'] = folders()
    if cmd in ('acl', 'all'):
        doc['acl'] = acl()
    VC.jsave(doc, OUT)
    emit('→ %s' % OUT)


if __name__ == '__main__':
    main()
