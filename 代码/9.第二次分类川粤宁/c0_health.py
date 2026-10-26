# -*- coding: utf-8 -*-
"""c0_health.py — 第二次分类（川/粤/宁）账号健康复检：候选名单 → 项目级健康（零 EECU）

* 输入：候选账号 = 今日（2026-10-13 17:49 fleet_health_subset）healthy 21 个
        ∪ 第一次分类（五省批）12 账号 ∪ fleet_100 备用若干
* 规则源：s0b_health.check（restricted-mode 捕获 + listAssets + 队列查询）
* 纪律：**必须串行**——apply_account_env 改的是进程级 HOME/USERPROFILE，多线程会串号
  （2026-10-14 03:2x 实测：8 线程并发 → 34/36 假"N does not have permission"；串行全部正常）
* 门槛：只把 ok=true 写进 used 名单；restricted-mode / 队列查询失败 / 项目未注册一律淘汰
* 输出：F:/lc_work/prod2nd_2023/reports/health_p2.json（全表 + healthy 列表 + 角色建议）
* 用法：python 代码/9.第二次分类川粤宁/c0_health.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
from s0b_health import check

OUT = r'F:/lc_work/prod2nd_2023/reports/health_p2.json'
CAND = [
    # 今日 17:49 healthy
    'e0p36771', 'eib2z5bs', 'es82gxq', 'fcz0iuwf', 'iu5f4z', 'j8tv79n', 'k3r6ncvd',
    'nnjxsn', 'oh6oiel', 'p2vlkaia', 'pjsxq83j', 'q6hbzjk', 'qy7gff', 'save456jr',
    's8xpcl1w', 'seqsiu', 'sfvmtj0u', 'ybzljyma', 'zhnagningdan1', 'zitwwufh', 'zzuoavsm',
    # 第一次分类（五省批）12 账号
    '11e0n00', 'e5h08k', 'hj74bml', 'hqzub6', 'hte4021n', 'r45smj3u',
    'y30b63ye', 'zixen8v8', 'xysongf',
    # fleet_100 备用
    'bx15mw', 'bv9yc0', 'ccstqqb', 'ch2fg2t9', 'clvqm4', 'eib2z5bs', 'r0dt6w',
]


def one(a):
    try:
        return check(a)
    except Exception as e:
        return dict(acct=a, ok=False, reason='check 异常:' + str(e)[:80], qdepth=None, n_assets=None)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cands = list(dict.fromkeys(CAND))
    print('复检 %d 个候选账号（零 EECU，串行）…' % len(cands), flush=True)
    rows = []
    for i, a in enumerate(cands, 1):
        r = one(a)
        rows.append(r)
        print('[%2d/%d] %-16s %s %s' % (i, len(cands), r['acct'],
                                        'OK ' if r['ok'] else 'NG ', r['reason'][:70]), flush=True)
    rows.sort(key=lambda r: r['acct'])
    healthy = [r['acct'] for r in rows if r['ok']]
    out = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), n=len(cands),
               healthy=healthy, rows=rows)
    VC.jsave(out, OUT)
    print('\n健康 %d/%d → %s' % (len(healthy), len(cands), OUT))
    print('healthy:', healthy)


if __name__ == '__main__':
    main()
