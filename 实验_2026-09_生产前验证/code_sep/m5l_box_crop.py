# -*- coding: utf-8 -*-
"""m5l_box_crop.py — 10 m 青框邻域精判图：从原始 512 切片中心裁窗并放大（单点 + 4 点拼图）

* 输入：<outdir>/chips/<point_id>.png（512×512 原生，中心=点位）
        <outdir>/sheet_map.json（若存在，按其顺序分组；否则按 chips 字典序）
* 规则源：以切片中心 (256,256) 为心裁 WIN 原始像素窗口（默认 120 px ≈ 55–65 m），LANCZOS 放大 SCALE 倍（默认 8）；
  重绘 point_id 标签（ASCII）；不加任何遮挡标记——青色 10 m 框在原图中已存在。
* 门槛：WIN ≤ 512；拼图按 2×2 组（与 sheet_map 的 4 点分组一致，便于对照 zoom_sheets）。
* 输出：<outdir>/box8/<point_id>.png（默认 120→960 px）
        <outdir>/box4_sheets/sheet_%02d.png（4 点拼图 ≈1950×1950）
* 用法：python m5l_box_crop.py --outdir data/m3/expA_imagery [--pids a,b,c] [--win 120] [--scale 8] [--force]
幂等：单点图存在即跳过。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont

CROP = 512
CENTER = 256


def font(sz=30):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', sz)
    except Exception:
        return ImageFont.load_default()


def main():
    a = sys.argv[1:]
    outdir = a[a.index('--outdir') + 1]
    win = int(a[a.index('--win') + 1]) if '--win' in a else 120
    scale = int(a[a.index('--scale') + 1]) if '--scale' in a else 8
    force = '--force' in a
    chipd = os.path.join(outdir, 'chips')
    bd = os.path.join(outdir, 'box8')
    os.makedirs(bd, exist_ok=True)
    smap_fp = os.path.join(outdir, 'sheet_map.json')
    groups = []
    if '--pids' in a:
        pids = a[a.index('--pids') + 1].split(',')
        groups = [pids[i:i + 4] for i in range(0, len(pids), 4)]
    elif os.path.exists(smap_fp):
        smap = json.load(open(smap_fp, encoding='utf-8'))
        groups = [smap[k] for k in sorted(smap, key=lambda s: int(s.split('_')[1]))]
    else:
        pids = sorted(f[:-4] for f in os.listdir(chipd) if f.endswith('.png'))
        groups = [pids[i:i + 4] for i in range(0, len(pids), 4)]
    half = win // 2
    for grp in groups:
        for pid in grp:
            fp = os.path.join(bd, pid + '.png')
            if os.path.exists(fp) and not force:
                continue
            im = Image.open(os.path.join(chipd, pid + '.png')).convert('RGB')
            im2 = im.crop((CENTER - half, CENTER - half, CENTER + half, CENTER + half))
            im2 = im2.resize((im2.width * scale, im2.height * scale), Image.LANCZOS)
            d = ImageDraw.Draw(im2)
            f = font(34)
            tb = d.textbbox((12, 10), pid, font=f)
            d.rectangle((tb[0] - 5, tb[1] - 4, tb[2] + 5, tb[3] + 4), fill=(0, 0, 0))
            d.text((12, 10), pid, font=f, fill=(255, 255, 0))
            im2.save(fp)
    print('box8 单点图 %d 点 → %s（%d px 窗 ×%d）' % (sum(len(g) for g in groups), bd, win, scale))
    sd = os.path.join(outdir, 'box4_sheets')
    os.makedirs(sd, exist_ok=True)
    T = 8
    n = 0
    for k, grp in enumerate(groups):
        f0 = os.path.join(bd, grp[0] + '.png')
        if not os.path.exists(f0):
            continue
        im0 = Image.open(f0)
        W = 2 * (im0.width + T) + T
        H = 2 * (im0.height + T) + T
        sh = Image.new('RGB', (W, H), (25, 25, 25))
        for i, pid in enumerate(grp):
            fp = os.path.join(bd, pid + '.png')
            if not os.path.exists(fp):
                continue
            im = Image.open(fp)
            r, c = divmod(i, 2)
            sh.paste(im, (T + c * (im.width + T), T + r * (im.height + T)))
        sh.save(os.path.join(sd, 'sheet_%02d.png' % k))
        n += 1
    print('box4 拼图 %d 张 → %s' % (n, sd))


if __name__ == '__main__':
    main()
