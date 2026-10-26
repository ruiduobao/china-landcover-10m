# -*- coding: utf-8 -*-
"""mkzoom.py — 为盲判生成逐点放大复合图（本地计算，无模型调用）

* 输入：materials/batch_XX/sheets/sheet_*.png（2×2 联系表，1042×1042，每格512）
* 规则源：rs-sample-labeling 技能——判的是青色框（10m 目标像元）内主导地物；
  联系表每格中心即目标点，青色框画在中心。
* 输出：materials/batch_XX/zooms/zsheet_<nn>.png（1024×1024）
  每格 = 原格中心 256×256 区域 ×2 放大（含青色框与十字丝），四格不变，
  另在四角加 64px 上下文条带（原格 512 缩至 256 的四邻域）以看stand结构。
* 用法：python mkzoom.py [batch_dir ...]
"""
import os, sys, glob
from PIL import Image

def main(dirs):
    for bd in dirs:
        src = sorted(glob.glob(os.path.join(bd, 'sheets', 'sheet_*.png')))
        out = os.path.join(bd, 'zooms')
        os.makedirs(out, exist_ok=True)
        for f in src:
            im = Image.open(f).convert('RGB')
            W, H = im.size          # 1042x1042
            cw, ch = W // 2, H // 2  # 521
            # 四格中心
            centers = [(cw // 2, ch // 2), (cw + cw // 2, ch // 2),
                       (cw // 2, ch + ch // 2), (cw + cw // 2, ch + ch // 2)]
            tiles = []
            for (cx, cy) in centers:
                r = 128  # 目标像素半径128 -> 256×256
                x0, y0 = max(0, cx - r), max(0, cy - r)
                crop = im.crop((x0, y0, x0 + 2 * r, y0 + 2 * r)).resize((512, 512), Image.LANCZOS)
                tiles.append(crop)
            comp = Image.new('RGB', (1024, 1024))
            comp.paste(tiles[0], (0, 0)); comp.paste(tiles[1], (512, 0))
            comp.paste(tiles[2], (0, 512)); comp.paste(tiles[3], (512, 512))
            name = os.path.basename(f).replace('sheet_', 'zsheet_')
            comp.save(os.path.join(out, name))
            print('->', os.path.join(out, name))

if __name__ == '__main__':
    main(sys.argv[1:] or ['F:/lc_work/label_repair_2023_20261015/materials/batch_01'])
