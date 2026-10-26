# -*- coding: utf-8 -*-
"""r0_health.py — materials 专用 GEE 账号：串行项目级健康检查（零 EECU，只读不提交）

* 依据：doc70 §3.1 + 主 agent 2026-10-15 指令：geometry 组独占 k3r6ncvd（备 p2vlkaia），
        materials 组用**其它**健康账号；本脚本按 GEE_CANDIDATES 顺序串行探测，取第一个 ok。
* 纪律：apply_account_env 改进程级 HOME/USERPROFILE → **必须串行**，不得多线程/多进程并行切号。
* 门槛：ok=true（Initialize + listAssets + 队列查询）；restricted-mode 一律淘汰。
* 输出：qc/health_materials.json（含选定账号；其它组按此避开）
* 用法：python r0_health.py [--cands p2vlkaia,j8tv79n] [--keep <acct>]
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

sys.path.insert(0, os.path.join(WORK, 'v31_exp', 'code'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', default=','.join(GEE_CANDIDATES))
    ap.add_argument('--keep', default=None)
    a = ap.parse_args()
    cands = [a.keep] if a.keep else [c.strip() for c in a.cands.split(',') if c.strip()]
    cands = [c for c in cands if c not in GEE_AVOID]
    import v31_common as VC
    from s0b_health import check
    out = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), avoid=GEE_AVOID, tried=[], selected=None)
    for acct in cands:
        rec = check(acct)
        out['tried'].append(rec)
        print('%-12s ok=%s  %s' % (acct, rec['ok'], rec['reason']), flush=True)
        if rec['ok'] and out['selected'] is None:
            out['selected'] = acct
    ensure_dirs()
    jdump(os.path.join(QC, 'health_materials.json'), out)
    print('选定 materials 专用账号：%s → %s' % (out['selected'], os.path.join(QC, 'health_materials.json')))


if __name__ == '__main__':
    main()
