# -*- coding: utf-8 -*-
"""h0_health.py — G-30N 几何实验专用账号：串行项目级健康检查（零 EECU）

* 依据：doc70 §3.1「GEE 仅由几何实验子 agent 独占一个经串行项目健康检查通过的账号」；
        c0_health.py 文件头纪律「必须串行——apply_account_env 改进程级 HOME/USERPROFILE，多线程串号」。
* 规则源：s0b_health.check（restricted-mode 捕获 + listAssets + 队列查询）
* 门槛：ok=true 才入选；restricted-mode / 项目未注册 / 队列查询失败一律淘汰。
        只挑 3 个已知健康候选（不扫全部 104 账号），避免无谓 GEE 调用。
* 附加只读核验（通过后）：目标瓦片 T1206 训练池资产 ACL 可读性（getAsset，零 EECU）。
* 输出：F:/lc_work/improve_2023_pilot_20261015/geometry/reg/health_geometry.json（全表 + 选定账号）
* 用法：python h0_health.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
from s0b_health import check

OUTD = r'F:/lc_work/improve_2023_pilot_20261015/geometry/reg'
OUT = os.path.join(OUTD, 'health_geometry.json')
# 3 个已知健康候选（health_p2.json 2026-10-14 健康；均属第二次分类编队 → 已具池资产读 ACL）
# 回避：同域账号对 s8xpcl1w+hte4021n、ppzynq+r45smj3u；并行会话三账 zitwwufh/iu5f4z/seqsiu
CAND = ['eib2z5bs', 'k3r6ncvd', 'p2vlkaia']
POOL_ASSET = 'projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged'


def acl_probe(acct):
    """零 EECU 只读：能否看到训练池资产 + 自己的仓（getAsset 走元数据，不触发计算）。"""
    rec = dict(acct=acct, pool_readable=None, pool_reason='')
    try:
        ee, pid = VC.ctx(acct)
        a = ee.data.getAsset(POOL_ASSET)
        rec['pool_readable'] = True
        rec['pool_reason'] = 'type=%s' % a.get('type')
    except Exception as e:
        rec['pool_readable'] = False
        rec['pool_reason'] = str(e)[:120]
    return rec


def main():
    os.makedirs(OUTD, exist_ok=True)
    print('串行健康复检 %d 个候选（零 EECU，逐个独立初始化）…' % len(CAND), flush=True)
    rows = []
    for i, a in enumerate(CAND, 1):
        t0 = time.time()
        r = check(a)
        r['dt_s'] = round(time.time() - t0, 1)
        if r['ok']:
            r.update(acl_probe(a))
        rows.append(r)
        print('[%d/%d] %-10s %s %s (%.0fs)' % (
            i, len(CAND), a, 'OK ' if r['ok'] else 'NG ', r['reason'][:70], r['dt_s']), flush=True)
    healthy = [r['acct'] for r in rows if r['ok'] and r.get('pool_readable')]
    chosen = healthy[0] if healthy else None
    out = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), n=len(CAND), rows=rows,
               healthy=healthy, chosen=chosen, pool_asset=POOL_ASSET,
               note='chosen 需 ok=true 且池资产可读；ID 记录不含任何密钥')
    VC.jsave(out, OUT)
    print('\n健康且池可读: %s ｜ chosen=%s\n→ %s' % (healthy, chosen, OUT))


if __name__ == '__main__':
    main()
