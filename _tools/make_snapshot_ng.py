# -*- coding: utf-8 -*-
"""
make_snapshot_ng.py — 为"下一代补样"这次更新打版本快照（代码/文档/小数据 + MD5 清单）

快照目录：Z:\地理所\论文\中国土地覆盖数据_2017-2024\版本快照\v<日期>_ng1
  · 代码：代码/6.下一代补样/ 全量（含生成的 15 份账号专属工作器）
  · 文档：技术文档/22、23、24
  · 工具：_tools/acct_inventory.py + out/acct_inventory.json/log
  · 小数据：ng_spec.json、ng_fleet/*.json、ng_idx/plan_*.json、ng_qa/*.json
  · 大产物（ng_raw / ng_gate 的 parquet）**不复制**，只在 MANIFEST 里登记路径+大小+行数
    —— 它们已在 K 盘权威副本内，复制只会浪费空间。

用法: python _tools/make_snapshot_ng.py [--tag v20260913_ng1]
"""
import os
import sys
import json
import glob
import shutil
import hashlib
import argparse
import datetime

sys.stdout.reconfigure(encoding='utf-8')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
NG = os.path.join(KB, '数据/本地处理/下一代补样')


def md5(fp, bs=1 << 20):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm(p):
    return p.replace('\\', '/')


def collect():
    items = []          # (src_abs, rel_in_snapshot)
    base = os.path.join(PROJ, '代码/6.下一代补样')
    for fp in glob.glob(os.path.join(base, '**', '*'), recursive=True):
        if os.path.isfile(fp) and '__pycache__' not in fp:
            items.append((fp, os.path.join('代码/6.下一代补样', os.path.relpath(fp, base))))
    for name in ('22_账号全量盘点与15账号选定（2026-09-13）.md',
                 '23_下一代补样方案（路径结构·检验标准·15家族）.md',
                 '24_下一代补样试点报告.md',
                 '25_下一代补样批量展开与并入母库报告.md',
                 '26_下一代补样第二轮与K盘迁移报告.md'):
        fp = os.path.join(PROJ, '技术文档', name)
        if os.path.isfile(fp):
            items.append((fp, os.path.join('技术文档', name)))
    for rel in ('_tools/acct_inventory.py', '_tools/out/acct_inventory.json',
                '_tools/out/acct_inventory.log', '_tools/make_snapshot_ng.py'):
        fp = os.path.join(PROJ, norm(rel))
        if os.path.isfile(fp):
            items.append((fp, rel))
    for rel in ('ng_spec.json',):
        fp = os.path.join(base, rel)
        if os.path.isfile(fp):
            items.append((fp, os.path.join('代码/6.下一代补样', rel)))
    for pat, sub in ((os.path.join(NG, 'ng_fleet', '*.json'), 'ng_fleet'),
                     (os.path.join(NG, 'ng_idx', '*.json'), 'ng_idx'),
                     (os.path.join(NG, 'ng_qa', '*.json'), 'ng_qa')):
        for fp in sorted(glob.glob(pat)):
            items.append((fp, os.path.join('数据/下一代补样_小产物', sub, os.path.basename(fp))))
    return items


def heavy_manifest():
    """大产物只登记不复制"""
    rows = []
    for sub in ('ng_raw', 'ng_gate'):
        for fp in sorted(glob.glob(os.path.join(NG, sub, '**', '*.parquet'), recursive=True)):
            rows.append({'path': norm(fp), 'bytes': os.path.getsize(fp),
                         'mtime': datetime.datetime.fromtimestamp(
                             os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v20260913_ng1')
    a = ap.parse_args()
    dst = os.path.join(KB, '版本快照', a.tag)
    if os.path.isdir(dst):                      # 同名快照已存在 → 追加时刻，绝不覆盖
        dst += '_' + datetime.datetime.now().strftime('%H%M')
    os.makedirs(dst, exist_ok=True)
    items = collect()
    man = []
    total = 0
    for src, rel in items:
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        sz = os.path.getsize(src)
        total += sz
        man.append({'rel': norm(rel), 'bytes': sz, 'md5': md5(src)})
        print(f'  {sz:>10,}  {norm(rel)}')
    heavy = heavy_manifest()
    payload = {'tag': a.tag,
               'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'files': len(man), 'bytes': total,
               'snapshot_items': man,
               'heavy_artifacts_registered_not_copied': heavy,
               'heavy_bytes': sum(h['bytes'] for h in heavy)}
    json.dump(payload, open(os.path.join(dst, 'MANIFEST.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n快照 {a.tag}: {len(man)} 个小文件 / {total/1e6:.2f} MB → {norm(dst)}')
    print(f'大产物（登记不复制）: {len(heavy)} 个 / {payload["heavy_bytes"]/1e6:.2f} MB')
    print('清单:', norm(os.path.join(dst, 'MANIFEST.json')))


if __name__ == '__main__':
    main()
