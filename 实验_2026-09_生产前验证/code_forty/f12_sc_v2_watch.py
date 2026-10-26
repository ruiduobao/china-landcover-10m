# -*- coding: utf-8 -*-
"""f12_sc_v2_watch.py — 四川 v2 批次监视器（轮询至全部终态，打印每瓦状态/EECU）

* 输入：F:/lc_work/prod5p_2023/sc_v2_state.json
* 输出：stdout 日志 + 更新 state
* 用法：python f12_sc_v2_watch.py
"""
import sys, time, collections
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
import ee
STATE = r'F:/lc_work/prod5p_2023/sc_v2_state.json'
TERM = ('COMPLETED', 'FAILED', 'CANCELLED')
DEADLINE = time.time() + 10 * 3600

def log(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def main():
    while time.time() < DEADLINE:
        st = VC.jload(STATE, {'tiles': {}})
        pend = []
        for t, r in sorted(st['tiles'].items()):
            if r.get('state') in TERM: continue
            try:
                VC.ensure_ctx(r['acct']); s = ee.data.getTaskStatus(r['task'])[0]
                r['state'] = s['state']
                r['eecu'] = float(s.get('batch_eecu_usage_seconds', 0) or 0)/3600
            except Exception as e:
                r['state'] = r.get('state', '?')
            if r['state'] not in TERM: pend.append(t)
        VC.jsave(st, STATE)
        c = collections.Counter(v.get('state') for v in st['tiles'].values())
        tot = sum(v.get('eecu', 0) for v in st['tiles'].values())
        log('状态 %s ｜ 累计 EECU %.1f h ｜ 未终态 %d' % (dict(c), tot, len(pend)))
        if not pend:
            log('全部终态'); return
        time.sleep(300)

if __name__ == '__main__': main()
