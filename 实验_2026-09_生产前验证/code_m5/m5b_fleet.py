# -*- coding: utf-8 -*-
"""m5b_fleet.py — D0 材料生成的账号舰队驱动（分片 + 并行 + 断点续跑；doc45 §一）
* 输入：--task {ind,chip} --points csv --out 目录 --nshards N --concurrency K
* 账号源：../config/accounts_pool.json 的 ok 名单（或 --accts 覆盖）；失败分片自动换账号重投
* 门槛：并发 ≤ K（默认 8，网络型任务）；分片幂等（m5a_materials 端点级跳过已完成点）
* 输出：logs/m5_{task}_{shard}.log；本脚本打印进度与末尾汇总
* 用法：python m5b_fleet.py --task chip --points ../data/m3/m5_all_points.csv --out ../data/m3/m5_geemat/chip --nshards 12
"""
import os, sys, json, time, argparse, subprocess
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, '..', 'config', 'accounts_pool.json')
LOGD = os.path.join(HERE, '..', 'logs')
FLEET = os.path.join(HERE, '..', 'config', 'fleet_100.json')


def _pids_file():
    return os.path.join(HERE, '..', 'config', 'm5_pids.json')


def _resolvable(accts):
    """保留有项目 ID 的账号（静态映射 m5_pids.json 缓存；零 GEE 调用）。"""
    import json as _json
    cache = {}
    fp = _pids_file()
    if os.path.exists(fp):
        cache = _json.load(open(fp, encoding='utf-8'))
    import v31_common as VC
    out = []
    for a in accts:
        try:
            if VC.pid_of(a):
                out.append(a); continue
        except Exception:
            pass
        if cache.get(a):
            out.append(a)
    return out


def pid_of_acct(a):
    import json as _json
    try:
        import v31_common as VC
        p = VC.pid_of(a)
        if p:
            return p
    except Exception:
        pass
    fp = _pids_file()
    if os.path.exists(fp):
        return _json.load(open(fp, encoding='utf-8')).get(a, '')
    return ''


def load_accounts(override=None):
    if override:
        return _resolvable(override.split(','))
    if os.path.exists(FLEET):
        d = json.load(open(FLEET, encoding='utf-8'))
        if d.get('accounts'):
            return _resolvable(list(d['accounts']))
    d = json.load(open(POOL, encoding='utf-8'))
    return _resolvable([r['acct'] for r in d.get('rows', []) if r.get('ok')])


def write_fleet(accts):
    json.dump({'time': time.strftime('%Y-%m-%d %H:%M'), 'n_ok': len(accts), 'accounts': accts,
               'note': 's0_sweep 探活 ok ∩ 项目 ID 可解析名单；材料任务并发 8-16（失败自动换号）'},
              open(FLEET, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('fleet_100.json 已写：%d 个可用账号' % len(accts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['ind', 'chip', 'ch'], required=True)
    ap.add_argument('--points', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nshards', type=int, default=12)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--accts', default=None)
    ap.add_argument('--max-retry', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(LOGD, exist_ok=True)
    accts = load_accounts(a.accts)
    write_fleet(accts)
    print('账号 %d 个（用 %d 个并行）：%s' % (len(accts), min(a.concurrency, len(accts)), accts[:10]), flush=True)
    # 队列：shard -> (retry, acct_idx)；启动时跳过"正在被用"的账号（并发内账号互异）
    queue = [(i, 0, i % len(accts)) for i in range(a.nshards)]
    running = {}   # popen -> (shard, retry, lf, acct)
    done, failed = [], []
    while queue or running:
        while queue and len(running) < a.concurrency:
            shard, retry, ai = queue.pop(0)
            used = {v[3] for v in running.values()}
            n = len(accts)
            acct = None
            for off in range(n):
                cand = accts[(ai + off) % n]
                if cand not in used:
                    acct = cand
                    break
            if acct is None:
                queue.insert(0, (shard, retry, ai))
                break
            log = os.path.join(LOGD, 'm5_%s_%d_r%d.log' % (a.task, shard, retry))
            cmd = [sys.executable, '-u', os.path.join(HERE, 'm5a_materials.py'),
                   '--task', a.task, '--points', a.points, '--acct', acct,
                   '--shard', str(shard), '--nshards', str(a.nshards), '--out', a.out]
            pid = pid_of_acct(acct)
            if pid:
                cmd += ['--pid', pid]
            lf = open(log, 'w', encoding='utf-8')
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=HERE, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            running[p] = (shard, retry, lf, acct)
            print('启动 shard %d（%s，r%d）' % (shard, acct, retry), flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is None:
                continue
            shard, retry, lf, acct = running.pop(p)
            lf.close()
            if p.returncode == 0:
                done.append(shard)
                print('✓ shard %d（%s）' % (shard, acct), flush=True)
            elif retry < a.max_retry:
                queue.append((shard, retry + 1, (shard + 17 * (retry + 1)) % len(accts)))
                print('✗ shard %d（%s）rc=%d → 换号重投' % (shard, acct, p.returncode), flush=True)
            else:
                failed.append(shard)
                print('✗✗ shard %d 放弃（rc=%d）' % (shard, p.returncode), flush=True)
    print('\n== 完成 %d/%d ｜ 失败 %s ==' % (len(done), a.nshards, failed or '无'), flush=True)


if __name__ == '__main__':
    main()
