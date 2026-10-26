# -*- coding: utf-8 -*-
"""f9_blind_sheets.py — 目视抽查盲判封装：中性编号重贴标签 + 3×3 联系表

* 输入：F:/lc_work/forty_exp/vis/judge_points.csv（point_id/lon/lat/group）
        F:/lc_work/forty_exp/vis/imagery/chips/<point_id>.png（m5j z18 切片）
* 规则源：
  ① 每张切片左上角的原 point_id（含分组前缀，会泄漏答案）用黑块覆盖，改贴**随机中性编号**；
  ② 3×3 联系表（每张 9 点，1536×1536），命名 sheet_XX.png；
  ③ 映射写入 `vis/blind_map.csv`（**不交给判读方**）。
* 输出：vis/blind/*.png、vis/sheets/sheet_XX.png、vis/blind_map.csv
* 用法：python f9_blind_sheets.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

VIS = r'F:/lc_work/forty_exp/vis'
CHIPS = os.path.join(VIS, 'imagery', 'chips')
BLIND = os.path.join(VIS, 'blind')
SHEETS = os.path.join(VIS, 'sheets')
CROP = 512
COLS, ROWS = 3, 3
GAP = 6


def main():
    os.makedirs(BLIND, exist_ok=True)
    os.makedirs(SHEETS, exist_ok=True)
    d = pd.read_csv(os.path.join(VIS, 'judge_points.csv'), encoding='utf-8-sig')
    rng = np.random.default_rng(20261026)
    perm = rng.permutation(len(d))
    neutral = ['S%03d' % (i + 1) for i in range(len(d))]
    d['neutral'] = ''
    d.loc[d.index[perm], 'neutral'] = neutral
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 26)
    except Exception:
        font = ImageFont.load_default()
    ok = 0
    for r in d.itertuples():
        src = os.path.join(CHIPS, r.point_id + '.png')
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert('RGB')
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, 300, 46), fill=(0, 0, 0))
        dr.text((10, 10), r.neutral, font=font, fill=(255, 255, 255))
        im.save(os.path.join(BLIND, r.neutral + '.png'))
        ok += 1
    d = d[d.neutral != ''].reset_index(drop=True)
    d.to_csv(os.path.join(VIS, 'blind_map.csv'), index=False, encoding='utf-8-sig')
    per = COLS * ROWS
    n = 0
    for k in range(0, len(d), per):
        grp = d.iloc[k:k + per]
        W, H = COLS * (CROP + GAP) + GAP, ROWS * (CROP + GAP) + GAP
        sh = Image.new('RGB', (W, H), (30, 30, 30))
        for i, r in enumerate(grp.itertuples()):
            im = Image.open(os.path.join(BLIND, r.neutral + '.png'))
            rr, cc = divmod(i, COLS)
            sh.paste(im, (GAP + cc * (CROP + GAP), GAP + rr * (CROP + GAP)))
        sh.save(os.path.join(SHEETS, 'sheet_%02d.png' % n))
        n += 1
    print('盲判封装：%d 张切片，%d 张联系表（每张 %d 点）' % (ok, n, per))
    print('映射（勿外传）：%s' % os.path.join(VIS, 'blind_map.csv'))


if __name__ == '__main__':
    main()
