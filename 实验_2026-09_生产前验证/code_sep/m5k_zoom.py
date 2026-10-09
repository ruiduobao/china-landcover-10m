# -*- coding: utf-8 -*-
"""m5k_zoom.py — 判读用放大切片与联系表（在 m5j 原生 512 切片基础上做窗口裁剪+放大）

* 输入：<outdir>/chips/<point_id>.png（512×512，中心=点位，含十字丝与 10 m 青框）
        <outdir>/fetch_log.json（每点 mpp/half_px）
* 规则源：以点位为中心裁 W 米窗口（默认 140 m，纬度换算用 fetch_log.mpp），LANCZOS 放大 SCALE 倍（默认 3），
  重绘 point_id 标签；联系表 2×2（默认 4 点/张，灰底 6px 间距）。
* 门槛：缺 fetch_log 时按纬度估算 mpp；窗口不得超 512 px；联系表按 point_id 字典序（与 m5j 一致）。
* 输出：<outdir>/zoom/<point_id>.png（默认 140m→约 900×900）
        <outdir>/zoom_sheets/sheet_%02d.png（约 1800×1800，含 sheet/象限说明）
        <outdir>/sheet_map.json（sheet → [point_id×4 顺序]，与 m5j 的 sheets 顺序同步写出）
* 用法：python m5k_zoom.py --outdir data/m3/expA_imagery [--win 140] [--scale 3] [--per 4]
幂等：zoom 切片存在即跳过（--force 重写）；联系表每次重写。
"""
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont

CROP = 512


def emit(m):
    import time
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def font(sz=26):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', sz)
    except Exception:
        return ImageFont.load_default()


def make_zoom(outdir, win=140.0, scale=3, force=False):
    chipd = os.path.join(outdir, 'chips')
    zd = os.path.join(outdir, 'zoom')
    os.makedirs(zd, exist_ok=True)
    fl = {}
    fp = os.path.join(outdir, 'fetch_log.json')
    if os.path.exists(fp):
        fl = json.load(open(fp, encoding='utf-8'))
    pids = sorted(f[:-4] for f in os.listdir(chipd) if f.endswith('.png'))
    n = 0
    for pid in pids:
        out = os.path.join(zd, pid + '.png')
        if os.path.exists(out) and not force:
            continue
        im = Image.open(os.path.join(chipd, pid + '.png')).convert('RGB')
        mpp = (fl.get(pid) or {}).get('mpp')
        if not mpp:
            lat = (fl.get(pid) or {}).get('lat')
            mpp = 156543.03392 * math.cos(math.radians(lat or 35.0)) / (2 ** 18)
        w = min(CROP, max(64, int(round(win / mpp))))
        c = CROP // 2
        half = w // 2
        im2 = im.crop((c - half, c - half, c + half, c + half))
        im2 = im2.resize((im2.width * scale, im2.height * scale), Image.LANCZOS)
        d = ImageDraw.Draw(im2)
        f = font(30)
        tb = d.textbbox((10, 8), pid, font=f)
        d.rectangle((tb[0] - 5, tb[1] - 4, tb[2] + 5, tb[3] + 4), fill=(0, 0, 0))
        d.text((10, 8), pid, font=f, fill=(255, 255, 0))
        d.text((10, tb[3] + 12), '%d m 窗 ×%d' % (win, scale), font=font(22), fill=(255, 255, 0))
        im2.save(out)
        n += 1
    emit('zoom 切片 %d 张（窗口 %.0f m ×%d）→ %s' % (len(pids), win, scale, zd))
    return pids


def make_sheets(outdir, pids, per=4, which='zoom'):
    cols = 2
    rows = per // cols
    srcd = os.path.join(outdir, which)
    shd = os.path.join(outdir, which + '_sheets')
    os.makedirs(shd, exist_ok=True)
    T = 6
    sample = Image.open(os.path.join(srcd, pids[0] + '.png'))
    W = cols * (sample.width + T) + T
    H = rows * (sample.height + T) + T
    smap = {}
    for k in range(0, len(pids), per):
        grp = pids[k:k + per]
        sh = Image.new('RGB', (W, H), (30, 30, 30))
        for i, pid in enumerate(grp):
            im = Image.open(os.path.join(srcd, pid + '.png'))
            r, c = divmod(i, cols)
            sh.paste(im, (T + c * (im.width + T), T + r * (im.height + T)))
        name = 'sheet_%02d' % (k // per)
        sh.save(os.path.join(shd, name + '.png'))
        smap[name] = grp
    json.dump(smap, open(os.path.join(outdir, 'sheet_map.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('%s 联系表 %d 张（%d 点/张）→ %s；sheet_map.json 已写出' % (which, len(smap), per, shd))


def main():
    a = sys.argv[1:]
    outdir = a[a.index('--outdir') + 1]
    win = float(a[a.index('--win') + 1]) if '--win' in a else 140.0
    scale = int(a[a.index('--scale') + 1]) if '--scale' in a else 3
    per = int(a[a.index('--per') + 1]) if '--per' in a else 4
    pids = make_zoom(outdir, win, scale, '--force' in a)
    make_sheets(outdir, pids, per, 'zoom')


if __name__ == '__main__':
    main()
