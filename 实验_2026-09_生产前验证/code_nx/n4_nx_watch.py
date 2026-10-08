# -*- coding: utf-8 -*-
"""n4_nx_watch.py — 宁夏 R4 批次监视器：轮询 6 瓦任务，全部终态后退出并打印汇总

* 输入：F:/lc_work/prod5p_2023/nx_r4_state.json（n3_nx_r4.py 维护）
* 规则源：每 5 分钟轮询一次；SUBMITTED/READY/RUNNING 视为在跑；连续 3 次查询异常不视为失败
* 门槛：全部终态（COMPLETED/FAILED/CANCELLED）或超过 8 小时退出
* 输出：stdout 日志（含每瓦 EECU）
* 用法：python n4_nx_watch.py
"""
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
import ee

STATE = r'F:/lc_work/prod5p_2023/nx_r4_state.json'
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED')
ACTIVE = ('SUBMITTED', 'READY', 'RUNNING')
DEADLINE = time.time() + 8 * 3600


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def main():
    while time.time() < DEADLINE:
        st = VC.jload(STATE, {'tiles': {}})
        pending = []
        lines = []
        for t, r in sorted(st['tiles'].items()):
            if r.get('state') in TERMINAL:
                lines.append('%s=%s(%.1fh)' % (t, r['state'], r.get('eecu', 0)))
                continue
            try:
                VC.ensure_ctx(r['acct'])
                s = ee.data.getTaskStatus(r['task'])[0]
                r['state'] = s['state']
                r['eecu'] = float(s.get('batch_eecu_usage_seconds', 0) or 0) / 3600.0
                lines.append('%s=%s(%.1fh)' % (t, r['state'], r['eecu']))
            except Exception as e:
                lines.append('%s=?%s' % (t, str(e)[:40]))
            if r.get('state') not in TERMINAL:
                pending.append(t)
        VC.jsave(st, STATE)
        log(' | '.join(lines))
        if not pending:
            log('全部终态，退出')
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
