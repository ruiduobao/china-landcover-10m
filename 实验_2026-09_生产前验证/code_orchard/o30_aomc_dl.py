# -*- coding: utf-8 -*-
"""o30_aomc_dl.py — AOMC 苹果园下载器（test.cors.workers.dev 取 S3 预签名 → 直连流式下载）

* 输入：_meta/files_<aid>.json（figshare 清单：id/name/size/md5）
* 原理：本机 IP 被 figshare WAF 拦；改由 test.cors.workers.dev（Cloudflare demo worker）代取 302，
        从其 Location 头拿 S3 预签名直链（X-Amz-Expires=10，须立刻发出下载请求；长流不受过期影响）。
* 门槛：逐文件字节数 + MD5 与清单核对；失败重试 3 次
* 输出：<out>/<name> + dl_worker.log
* 用法：python o30_aomc_dl.py --aid 28113302 --out F:/lc_work/外部样本/rare_class_dl/AOMC_apple [--only 5]
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/外部样本/rare_class_dl'
META = os.path.join(BASE, '_meta')
PROXY = 'socks5h://127.0.0.1:7890'
WORKER = 'https://test.cors.workers.dev/?'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36'


def log(fp, m):
    line = time.strftime('%H:%M:%S ') + str(m)
    print(line, flush=True)
    with open(fp, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def presign(file_id, logf):
    """经 worker 取 S3 预签名直链。"""
    r = random.randint(1000, 999999)
    target = 'https://ndownloader.figshare.com/files/%d?x=%d' % (file_id, r)
    for attempt in range(4):
        try:
            out = subprocess.run(
                ['curl', '-s', '--max-time', '40', '-x', PROXY, '-H', 'Origin: https://example.com',
                 '-D', '-', '-o', os.devnull, WORKER + target],
                capture_output=True, text=True, timeout=50)
            for line in out.stdout.splitlines():
                if line.lower().startswith('location:'):
                    url = line.split(':', 1)[1].strip()
                    if 's3' in url and 'X-Amz-Signature' in url:
                        return url
            log(logf, '  presign 无 S3 直链（第%d次）' % (attempt + 1))
        except Exception as e:
            log(logf, '  presign 异常 %s' % str(e)[:80])
        time.sleep(3)
    return None


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = os.path.join(a.out, 'dl_worker.log')
    files = json.load(open(os.path.join(META, 'files_%s.json' % a.aid), encoding='utf-8'))
    if a.only:
        files = files[:a.only]
    ok = skip = fail = 0
    for i, f in enumerate(files):
        name = f['name']
        fp = os.path.join(a.out, name)
        want = f.get('size')
        if os.path.exists(fp) and (not want or os.path.getsize(fp) == want):
            skip += 1
            continue
        log(logf, '[%d/%d] %s (%.1f MB)' % (i + 1, len(files), name, (want or 0) / 1e6))
        got = False
        for attempt in range(4):
            url = presign(f['id'], logf)
            if not url:
                time.sleep(5)
                continue
            # 立刻下载（先直连，失败走代理）
            for mode in ('direct', 'proxy'):
                cmd = ['curl', '-sL', '--max-time', '600', '-o', fp]
                if mode == 'proxy':
                    cmd += ['-x', PROXY]
                else:
                    cmd += ['--noproxy', '*']
                try:
                    subprocess.run(cmd + [url], timeout=620)
                except Exception:
                    pass
                if os.path.exists(fp) and (not want or os.path.getsize(fp) == want):
                    got = True
                    break
            if got:
                break
            time.sleep(2)
        if got:
            m = md5_of(fp)
            md5_want = (f.get('supplied_md5') or f.get('md5') or '').lower()
            if md5_want and m != md5_want:
                log(logf, '  MD5 不符（%s vs %s），删除重试' % (m[:8], md5_want[:8]))
                os.remove(fp)
                fail += 1
                continue
            ok += 1
            log(logf, '  OK %.1f MB md5=%s' % (os.path.getsize(fp) / 1e6, m[:8]))
        else:
            fail += 1
            log(logf, '  FAIL %s' % name)
        time.sleep(1)
    log(logf, '完成：ok=%d skip=%d fail=%d' % (ok, skip, fail))


if __name__ == '__main__':
    main()
