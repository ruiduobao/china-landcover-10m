# -*- coding: utf-8 -*-
"""h0_health.py — 三省几何复验专用账号：串行项目级健康检查（零 EECU）

* 依据：doc72 §8「原 GEE 健康账号串行检查、不同材料账号独立进程」；
        AGENTS 硬约束 8「每次用 GEE 账号前跑项目级健康检查」；
        c0_health.py 纪律「必须串行——apply_account_env 改进程级 HOME/USERPROFILE，多线程串号」。
* 规则源：s0b_health.check（restricted-mode 捕获 + listAssets + 队列查询）
* 门槛：ok=true 且训练池资产 ACL 可读（getAsset，零 EECU）才入选；restricted-mode 一律淘汰。
        只挑少量已知健康候选（不扫全部账号），避免无谓 GEE 调用与代理压力。
* 输出：F:/lc_work/label_repair_2023_20261015/geometry/reg/health_geometry.json（全表 + 选定账号）
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

OUTD = r'F:/lc_work/label_repair_2023_20261015/geometry/reg'
OUT = os.path.join(OUTD, 'health_geometry.json')
# 候选（第二次分类编队，均具池资产读 ACL 潜力；串行逐个独立初始化）
# 回避：同域账号对 s8xpcl1w+hte4021n、ppzynq+r45smj3u；禁动/排除账号不列入。
CAND = ['k3r6ncvd', 'p2vlkaia', 'es82gxq']
POOL_ASSET = 'projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged'


def acl_probe(acct):
    """零 EECU 只读：能否看到训练池资产（getAsset 走元数据，不触发计算）。"""
    rec = dict(acct=acct, pool_readable=None, pool_reason='')
    try:
        ee, pid = VC.ctx(acct)
        a = ee.data.getAsset(POOL_ASSET)
        rec['pool_readable'] = True
        rec['pool_reason'] = 'type=%s' % a.get('type')
        rec['proj'] = pid
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
    proj = next((r.get('proj') for r in rows if r['acct'] == chosen), None)
    out = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), n=len(CAND), rows=rows,
               healthy=healthy, chosen=chosen, chosen_proj=proj, pool_asset=POOL_ASSET,
               note='chosen 需 ok=true 且池资产可读；ID 记录不含任何密钥；同账号不同进程不并发切号')
    VC.jsave(out, OUT)
    print('\n健康且池可读: %s ｜ chosen=%s (proj=%s)\n→ %s' % (healthy, chosen, proj, OUT))


if __name__ == '__main__':
    main()
