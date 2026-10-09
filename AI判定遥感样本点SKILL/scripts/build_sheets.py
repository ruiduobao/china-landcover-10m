# -*- coding: utf-8 -*-
"""build_sheets.py — 切片拼"联系表"（判读批阅用）与批次清单

* 输入：--chips <dir>（fetch_chips.py 产物目录）
* 规则源：references/判读协议.md —— 盲判：联系表只写编号，绝不带先验标签/分层
* 门槛：单边 ≤--max-px（默认 1568，超出会被视觉模型缩水丢细节）；默认 2×2×512=1036 原生不缩水
* 输出：<outdir>/sheets/sheet_XX.png；<outdir>/sheet_map.json（表→点位清单）；<outdir>/batches.csv（判读批次）
* 用法：python build_sheets.py --chips ./run1/chips --outdir ./run1
        python build_sheets.py --chips ./run1/chips --outdir ./run1 --cols 3 --rows 3
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description='切片→联系表')
    ap.add_argument('--chips', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--cols', type=int, default=2)
    ap.add_argument('--rows', type=int, default=2)
    ap.add_argument('--gap', type=int, default=6)
    ap.add_argument('--max-px', type=int, default=1568)
    a = ap.parse_args()
    sheetd = os.path.join(a.outdir, 'sheets')
    os.makedirs(sheetd, exist_ok=True)
    pts = sorted(f[:-4] for f in os.listdir(a.chips) if f.lower().endswith('.png'))
    if not pts:
        raise SystemExit('没有切片：%s' % a.chips)
    side = max(Image.open(os.path.join(a.chips, pts[0] + '.png')).size)
    per = a.cols * a.rows
    W = a.cols * (side + a.gap) + a.gap
    H = a.rows * (side + a.gap) + a.gap
    if max(W, H) > a.max_px:
        print('警告：联系表单边 %d > %d，会被视觉模型缩水；建议减小 --cols/--rows 或 --crop' % (max(W, H), a.max_px))
    smap, batches = {}, []
    n = 0
    for k in range(0, len(pts), per):
        grp = pts[k:k + per]
        sh = Image.new('RGB', (W, H), (40, 40, 40))
        for i, pid in enumerate(grp):
            im = Image.open(os.path.join(a.chips, pid + '.png'))
            r, c = divmod(i, a.cols)
            sh.paste(im, (a.gap + c * (side + a.gap), a.gap + r * (side + a.gap)))
        name = 'sheet_%02d' % (k // per)
        sh.save(os.path.join(sheetd, name + '.png'))
        smap[name] = grp
        batches.append({'sheet': name, 'n': len(grp), 'points': ';'.join(grp)})
        n += 1
    json.dump(smap, open(os.path.join(a.outdir, 'sheet_map.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    with open(os.path.join(a.outdir, 'batches.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        f.write('sheet,n,points\n')
        for b in batches:
            f.write('%s,%d,%s\n' % (b['sheet'], b['n'], b['points']))
    print('联系表 %d 张（每张 %d 点，格 %d×%d）→ %s' % (n, per, a.cols, a.rows, sheetd))
    print('判读批次清单：%s（盲判：只含编号）' % os.path.join(a.outdir, 'batches.csv'))


if __name__ == '__main__':
    main()
