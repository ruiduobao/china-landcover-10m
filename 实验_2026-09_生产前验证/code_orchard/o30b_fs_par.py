# -*- coding: utf-8 -*-
"""o30b_fs_par.py — figshare 并行分块下载器（worker 预签名 + Range 并发，~6× 提速）

* 输入：_meta/files_<aid>.json（id/name/size/supplied_md5）
* 原理：每块（默认 8MB）独立走 test.cors.workers.dev 取全新预签名（10s 窗口），线程池 6 并发
        立即发起 Range GET（经代理），块落盘 <name>.p0000 ...；全部完成后拼装 + MD5 校验。
* 门槛：MD5 与清单一致才保留；断点：已存在且全的块跳过
* 输出：<out>/<name> + dl_par.log
* 用法：python o30b_fs_par.py --aid 28113302 --out .../AOMC_apple [--workers 6] [--chunkmb 8]
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/外部样本/rare_class_dl'
META = os.path.join(BASE, '_meta')
PROXY = 'socks5h://127.0.0.1:7890'
WORKER = 'https://test.cors.workers.dev/?'
_log_lock = threading.Lock()
_logf = None


def log(m):
    line = time.strftime('%H:%M:%S ') + str(m)
    with _log_lock:
        print(line, flush=True)
        if _logf:
            open(_logf, 'a', encoding='utf-8').write(line + '\n')


def presign(file_id):
    r = random.randint(1000, 9999999)
    target = 'https://ndownloader.figshare.com/files/%d?x=%d' % (file_id, r)
    for attempt in range(4):
        try:
            out = subprocess.run(
                ['curl', '-s', '--max-time', '30', '-x', PROXY, '-H', 'Origin: https://example.com',
                 '-D', '-', '-o', os.devnull, WORKER + target],
                capture_output=True, text=True, timeout=40)
            for line in out.stdout.splitlines():
                if line.lower().startswith('location:'):
                    url = line.split(':', 1)[1].strip()
                    if 'X-Amz-Signature' in url:
                        return url
        except Exception:
            pass
        time.sleep(2)
    return None


def fetch_chunk(file_id, fp, start, end, want_md5_part=None):
    """下载 [start, end] 到 fp（.part 文件）。返回 True/False。"""
    if os.path.exists(fp) and os.path.getsize(fp) == end - start + 1:
        return True
    for attempt in range(4):
        url = presign(file_id)
        if not url:
            continue
        tmp = fp + '.tmp'
        try:
            subprocess.run(['curl', '-s', '--max-time', '300', '-x', PROXY,
                            '-r', '%d-%d' % (start, end), '-o', tmp, url],
                           timeout=320)
        except Exception:
            pass
        if os.path.exists(tmp) and os.path.getsize(tmp) == end - start + 1:
            os.replace(tmp, fp)
            return True
        time.sleep(2)
    return False


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    global _logf
    ap = argparse.ArgumentParser()
    ap.add_argument('--aid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--chunkmb', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _logf = os.path.join(a.out, 'dl_par.log')
    files = json.load(open(os.path.join(META, 'files_%s.json' % a.aid), encoding='utf-8'))
    CH = a.chunkmb * (1 << 20)
    ok = skip = fail = 0
    for i, f in enumerate(files):
        name, want = f['name'], f.get('size')
        fp = os.path.join(a.out, name)
        if os.path.exists(fp) and (not want or os.path.getsize(fp) == want):
            skip += 1
            continue
        log('[%d/%d] %s (%.1f MB)' % (i + 1, len(files), name, (want or 0) / 1e6))
        if not want:
            # 无尺寸：单块全文
            if fetch_chunk(f['id'], fp + '.pall', 0, 1 << 62):
                os.replace(fp + '.pall', fp)
                ok += 1
            continue
        nch = (want + CH - 1) // CH
        tasks = [(c, c * CH, min(want - 1, (c + 1) * CH - 1)) for c in range(nch)]
        good = True
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(fetch_chunk, f['id'], fp + '.p%04d' % c, s, e) for c, s, e in tasks]
            good = all(fu.result() for fu in futs)
        if good:
            # 拼装
            with open(fp, 'wb') as w:
                for c, _, _ in tasks:
                    with open(fp + '.p%04d' % c, 'rb') as r:
                        w.write(r.read())
            for c, _, _ in tasks:
                os.remove(fp + '.p%04d' % c)
            m = md5_of(fp)
            mw = (f.get('supplied_md5') or '').lower()
            if mw and m != mw:
                log('  MD5 不符 → 删除')
                os.remove(fp)
                fail += 1
            else:
                ok += 1
                log('  OK %.1f MB md5=%s' % (os.path.getsize(fp) / 1e6, m[:8]))
        else:
            fail += 1
            log('  FAIL %s' % name)
    log('完成 ok=%d skip=%d fail=%d' % (ok, skip, fail))


if __name__ == '__main__':
    main()
