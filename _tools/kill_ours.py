import psutil, time, sys
killed, left = [], []
for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = ' '.join(p.info.get('cmdline') or [])
        if p.pid == 0 or 'kill_ours' in cl:
            continue
        if 'run_e1b_resume' in cl or ('supervise_resume' in cl and 'kill_ours' not in cl):
            p.kill(); killed.append(p.pid)
    except Exception:
        pass
time.sleep(3)
for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = ' '.join(p.info.get('cmdline') or [])
        if 'run_e1b_resume' in cl or 'supervise_resume' in cl:
            left.append((p.pid, cl[:80]))
    except Exception:
        pass
print('killed:', killed)
print('left:', left)
