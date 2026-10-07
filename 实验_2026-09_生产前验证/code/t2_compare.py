# -*- coding: utf-8 -*-
"""t2_compare.py — R0 试点瓦片 vs R4 复测瓦片：逐像元一致率与主要转换（同网格 10m）"""
import os, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import rasterio
from rasterio.windows import Window

OLD = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片'
NEW = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片_R4复测'


def cmp(tile):
    names = VC.V31_NAMES()
    fp_a = os.path.join(OLD, '%s_10m.tif' % tile)
    fp_b = os.path.join(NEW, '%s_10m.tif' % tile)
    if not (os.path.exists(fp_a) and os.path.exists(fp_b)):
        VC.emit('%s 缺文件' % tile); return None
    c = collections.Counter(); n = 0
    with rasterio.open(fp_a) as a, rasterio.open(fp_b) as b:
        if (a.width, a.height) != (b.width, b.height):
            VC.emit('%s 网格不一致 %s vs %s' % (tile, (a.width, a.height), (b.width, b.height))); return None
        for r0 in range(0, a.height, 2048):
            hh = min(2048, a.height - r0)
            w = Window(0, r0, a.width, hh)
            A = a.read(1, window=w).astype('uint16')
            B = b.read(1, window=w).astype('uint16')
            m = (A > 0) & (B > 0) & (A < 25) & (B < 25)
            n += int(m.sum())
            key = A[m] * 32 + B[m]
            u, cnt = np.unique(key, return_counts=True)
            for k, v in zip(u.tolist(), cnt.tolist()):
                c[int(k)] += int(v)
    same = sum(v for k, v in c.items() if k // 32 == k % 32)
    VC.emit('== %s == 有效像元 %d ｜ 逐像元一致率 %.2f%% ｜ 变化 %.2f%%' % (
        tile, n, 100.0 * same / n, 100.0 * (n - same) / n))
    trans = collections.Counter({(k // 32, k % 32): v for k, v in c.items() if k // 32 != k % 32})
    rows = []
    for (x, y), v in trans.most_common(10):
        rows.append('| %s → %s | %d | %.2f%% |' % (names.get(str(x), x), names.get(str(y), y), v, 100.0 * v / n))
        VC.emit('   %-10s → %-10s %10d (%.2f%%)' % (names.get(str(x), x), names.get(str(y), y), v, 100.0 * v / n))
    return dict(tile=tile, n=n, agree=100.0 * same / n, rows=rows)


def main():
    out = [cmp(t) for t in ('songnen', 'qinling')]
    L = ['# R0（试点原版） vs R4（复测）逐像元对照 %s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '> 同网格 10m，仅统计两侧均有效（非 0、<25）像元', '']
    for o in out:
        if not o:
            continue
        L += ['## %s' % o['tile'], '', '- 有效像元 %d ｜ **逐像元一致率 %.2f%%**（变化 %.2f%%）' % (
            o['n'], o['agree'], 100 - o['agree']), '',
            '| 转换 | 像元数 | 占有效像元 |', '|---|---|---|'] + o['rows'] + ['']
    fp = os.path.join(VC.REPT, 'R0vR4逐像元对照_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()
