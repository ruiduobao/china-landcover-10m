# -*- coding: utf-8 -*-
"""a0_hash_manifest.py — 对指定文件/目录生成 md5+sha256+size+mtime 清单（审计归档用）

* 输入：命令行若干路径（文件或目录，目录递归）
* 规则源：无（通用哈希工具）
* 门槛：无
* 输出：CSV（path,size_bytes,mtime,md5,sha256）；默认写到 audit/out/manifest_<ts>.csv
* 用法：python a0_hash_manifest.py <path> [<path> ...] [--out <csv>]
"""
import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
DEFAULT_OUT_DIR = os.path.join(AUDIT, 'out')

SKIP_DIRS = {'__pycache__', '.git'}


def _digests(path, chunk=1 << 20):
    m, s = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            m.update(b)
            s.update(b)
    return m.hexdigest(), s.hexdigest()


def iter_files(paths):
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in sorted(files):
                    yield os.path.join(root, fn)
        else:
            print('[WARN] not found: %s' % p, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = []
    for fp in iter_files(args.paths):
        try:
            md5, sha = _digests(fp)
        except OSError as e:
            print('[WARN] skip %s (%s)' % (fp, e), file=sys.stderr)
            continue
        st = os.stat(fp)
        rows.append({
            'path': fp,
            'size_bytes': st.st_size,
            'mtime': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'md5': md5,
            'sha256': sha,
        })

    out = args.out
    if not out:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out = os.path.join(DEFAULT_OUT_DIR, 'manifest_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'size_bytes', 'mtime', 'md5', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    print('[OK] %d files -> %s' % (len(rows), out))


if __name__ == '__main__':
    main()
