# -*- coding: utf-8 -*-
"""mkzoom4.py — 盲判用 4× 目标框放大拼图（本地计算，无模型调用）

* 输入：materials/batch_XX/sheets/sheet_*.png（2×2 联系表，每格 512px ≈ 258m，青色框≈10m）
* 输出：materials/batch_XX/zoom4/sheet_<nn>4.png（1024×1024）
  每格 = 该格中心 128×128 原像素（≈64m）放大 4 倍 → 512，青色框≈80px，看冠形/纹理；
  左上角与右下角分别标注 blind_id（沿用原格编号位置）。
* 用法：python mkzoom4.py <batch_dir> [...]
"""
import os, sys, glob
from PIL import Image, ImageDraw

def main(dirs):
    for bd in dirs:
        src = sorted(glob.glob(os.path.join(bd, 'sheets', 'sheet_*.png')))
        smap = os.path.join(bd, 'sheet_map.json')
        ids = {}
        if os.path.exists(smap):
            import json
            ids = json.load(open(smap, encoding='utf-8'))
        out = os.path.join(bd, 'zoom4')
        os.makedirs(out, exist_ok=True)
        for f in src:
            im = Image.open(f).convert('RGB')
            W, H = im.size
            cw, ch = W // 2, H // 2
            sheet = os.path.splitext(os.path.basename(f))[0]  # sheet_00
            pts = ids.get(sheet, ['?'] * 4)
            centers = [(cw // 2, ch // 2), (cw + cw // 2, ch // 2),
                       (cw // 2, ch + ch // 2), (cw + cw // 2, ch + ch // 2)]
            comp = Image.new('RGB', (1024, 1024))
            for i, (cx, cy) in enumerate(centers):
                r = 64
                x0, y0 = max(0, cx - r), max(0, cy - r)
                crop = im.crop((x0, y0, x0 + 2 * r, y0 + 2 * r)).resize((512, 512), Image.LANCZOS)
                px, py = (i % 2) * 512, (i // 2) * 512
                comp.paste(crop, (px, py))
                d = ImageDraw.Draw(comp)
                # 取原始格左上角编号字样（原格有 B#### 标签），这里重画在每格左上
                pid = pts[i] if i < len(pts) else '?'
                d.rectangle([px + 4, py + 4, px + 96, py + 30], fill=(0, 0, 0))
                d.text((px + 10, py + 9), str(pid), fill=(255, 255, 0))
            name = sheet + '4.png'
            comp.save(os.path.join(out, name))
            print('->', os.path.join(out, name))

if __name__ == '__main__':
    main(sys.argv[1:] or ['F:/lc_work/label_repair_2023_20261015/materials/batch_01'])
