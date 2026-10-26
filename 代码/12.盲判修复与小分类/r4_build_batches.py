# -*- coding: utf-8 -*-
"""r4_build_batches.py — 分批材料：复用缓存 chips → 2×2 联系表 → 表单并入 S2 指标提示列

* 输入：materials/batch_XX/blind_batch.csv（r2 产物）、cache/chips_all/chips/（r1 后抓取）、
        blind/s2_metrics.csv（r3 产物，可缺）
* 规则源：skill references/判读协议.md §2 盲判（联系表只写编号）；§3 效率（2×2×512 ≤1568 px）
* 门槛：联系表单边 = 2×(512+6)+6 = 1042 ≤ 1568（不缩水）；缺 chips 时用 skill fetch_chips.py 补齐
* 输出：materials/batch_XX/chips/、sheets/、sheet_map.json、batches.csv、
        form_with_metrics.csv（判读表单 + S2 提示列）、materials/batch_manifest.json（更新）
* 用法：python r4_build_batches.py [--batches 2,3,4,5]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

SKILL = r'C:/Users/Administrator/.zcode/skills/rs-sample-labeling/scripts'
HINT_COL = '指标提示（S2 物候，判读时交叉验证；与影像冲突以影像为准）'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def hint_of(row):
    p = []
    n = row.get('n_months_valid')
    if n == n:
        p.append('S2 有效月 %d' % int(n))
    if row.get('ndvi_amp') == row.get('ndvi_amp'):
        p.append('NDVI振幅%.2f' % row['ndvi_amp'])
    if row.get('ndvi_peak_month') == row.get('ndvi_peak_month'):
        p.append('峰值月%s' % str(int(row['ndvi_peak_month'])).zfill(2))
    if row.get('ndvi_winter2023') == row.get('ndvi_winter2023'):
        p.append('冬2023NDVI%.2f' % row['ndvi_winter2023'])
    if row.get('ndvi_winter2019') == row.get('ndvi_winter2019'):
        p.append('冬2019NDVI%.2f' % row['ndvi_winter2019'])
    if row.get('flood_evidence'):
        p.append({'yes': '有淹水证据', 'weak': '弱淹水信号', 'no': '无淹水'}.get(row['flood_evidence'], ''))
    p.append('（S2 有日期；高分底图无日期）')
    return '；'.join([x for x in p if x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='1,2,3,4,5')
    a = ap.parse_args()
    want = [int(x) for x in a.batches.split(',') if x.strip()]
    s2 = None
    fp = os.path.join(BLIND, 's2_metrics.csv')
    if os.path.exists(fp):
        s2 = pd.read_csv(fp, encoding='utf-8-sig')
        s2['point_id'] = s2['point_id'].astype(str)
        s2 = s2.set_index('point_id')
    cache_chips = os.path.join(CACHE, 'chips_all', 'chips')
    man = jload(os.path.join(MAT, 'batch_manifest.json'), {})
    byk = {b['batch']: b for b in man.get('batches', [])}

    for k in want:
        bdir = os.path.join(MAT, 'batch_%02d' % k)
        bp = os.path.join(bdir, 'blind_batch.csv')
        if not os.path.exists(bp):
            emit('批 %d 缺 blind_batch.csv，跳过' % k)
            continue
        pts = pd.read_csv(bp, encoding='utf-8-sig')
        pts['point_id'] = pts['point_id'].astype(str)
        chipd = os.path.join(bdir, 'chips')
        os.makedirs(chipd, exist_ok=True)
        n_copied = 0
        for pid in pts['point_id']:
            src = os.path.join(cache_chips, pid + '.png')
            dst = os.path.join(chipd, pid + '.png')
            if not os.path.exists(dst) and os.path.exists(src):
                shutil.copy2(src, dst)
                n_copied += 1
        missing = [p for p in pts['point_id'] if not os.path.exists(os.path.join(chipd, p + '.png'))]
        if missing:
            emit('批 %d 缺 %d 张，调用 skill fetch_chips 补齐' % (k, len(missing)))
            subprocess.run([sys.executable, os.path.join(SKILL, 'fetch_chips.py'),
                            '--points', bp, '--outdir', bdir, '--workers', '8'], check=False)
        subprocess.run([sys.executable, os.path.join(SKILL, 'build_sheets.py'),
                        '--chips', chipd, '--outdir', bdir, '--cols', '2', '--rows', '2'], check=False)
        smap = jload(os.path.join(bdir, 'sheet_map.json'), {})
        # 表单并入 S2 提示列
        fcol = list(pd.read_csv(os.path.join(bdir, 'form_skeleton.csv'), encoding='utf-8-sig').columns)
        if s2 is not None:
            with open(os.path.join(bdir, 'form_with_metrics.csv'), 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(fcol + [HINT_COL])
                for pid in pts['point_id']:
                    h = hint_of(s2.loc[pid]) if pid in s2.index else ''
                    w.writerow([pid] + [''] * (len(fcol) - 1) + [h])
        if k in byk:
            byk[k].update(dict(n=len(pts), sheets=os.path.join(bdir, 'sheets'),
                               sheet_map=os.path.join(bdir, 'sheet_map.json'),
                               form_with_metrics=(os.path.join(bdir, 'form_with_metrics.csv') if s2 is not None else None),
                               n_sheets=len(smap), sheets_ids=smap, chips_dir=chipd))
        emit('批 %d：%d 点 / %d 张联系表（复用 chips %d）' % (k, len(pts), len(smap), n_copied))
    man['batches'] = [byk[k] for k in sorted(byk)]
    man['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    man['s2_metrics'] = os.path.join(BLIND, 's2_metrics.csv') if s2 is not None else None
    jdump(os.path.join(MAT, 'batch_manifest.json'), man)
    emit('清单 → materials/batch_manifest.json')


if __name__ == '__main__':
    main()
