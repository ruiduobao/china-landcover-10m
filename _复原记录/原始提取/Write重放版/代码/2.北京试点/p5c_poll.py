# -*- coding: utf-8 -*-
"""
p5c_poll.py — 轮询 {out_dir}/tasks.jsonl 中全部任务直至完成（监控胶水脚本）
* 全部 SUCCEEDED → 退出码 0；有 FAILED/CANCELLED → 退出码 2；超时 → 退出码 3
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

MAX_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 120

def main():
    tasks = json.load(open(os.path.join(PC.OUT_DIR, 'tasks.jsonl'), encoding='utf-8'))
    by_acct = {}
    for t in tasks:
        by_acct.setdefault(t['account'], []).append(t)
    sess = {}
    t0 = time.time()
    while True:
        states = {}
        for acct, ts in by_acct.items():
            if acct not in sess:
                e, pid = PC.load_account(acct)
                sess[acct] = e
            e = sess[acct]
            for t in ts:
                try:
                    st = e.data.getTaskStatus(t['taskId'])
                    states[t['id']] = st[0]['state'] if isinstance(st, list) else st.get('state')
                except Exception as ex:
                    states[t['id']] = f'ERR:{str(ex)[:40]}'
        n_ok = sum(1 for s in states.values() if s == 'SUCCEEDED')
        n_bad = sum(1 for s in states.values() if s in ('FAILED', 'CANCELLED'))
        print(f"[{time.strftime('%H:%M:%S')}] SUCCEEDED {n_ok}/{len(tasks)}  "
              f"明细: {states}", flush=True)
        if n_ok == len(tasks):
            sys.exit(0)
        if n_bad:
            print('存在失败/取消任务', flush=True)
            sys.exit(2)
        if (time.time() - t0) > MAX_MIN * 60:
            print('轮询超时', flush=True)
            sys.exit(3)
        time.sleep(120)

if __name__ == '__main__':
    main()
