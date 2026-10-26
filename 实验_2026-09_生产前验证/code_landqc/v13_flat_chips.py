# -*- coding: utf-8 -*-
"""v13_flat_chips.py — 湖河滩地海岸修复的独立验证材料：z18 切片 + 联系表（盲判用）

* 输入：海岸污染修复/证据表.parquet（含 verdict/coast_qc）
* 抽样：A(out_tidal) 48 ｜ B(review_tidal) 30 ｜ C(out_beach) 1 ｜ D(review_nearshore) 30 ｜ E(keep) 24
        （种子固定；E 对照优先取"沿海省但远岸"与"内陆"混合）
* 输出：海岸污染修复/验证/chips/<pid>.png（512²，十字丝+10m 框，Esri World Imagery z18）
        验证/sheet_batch1.png（A+B 混排）｜ sheet_batch2.png（C+D+E 混排）——**表头只写 pid 与经纬度（盲判）**
        验证/抽样清单.csv（pid → 组别/证据，判读对照用，勿给判读人）
        验证/判读说明.md（rubric + 判读输出格式）
* 用法：python v13_flat_chips.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
CODE = r'F:/lc_work/v31_exp/code'
sys.path.insert(0, CODE)
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/湖河滩地/海岸污染修复'
VAL = os.path.join(DEST, '验证')


def main():
    os.makedirs(os.path.join(VAL, 'chips'), exist_ok=True)
    ev = pd.read_parquet(os.path.join(DEST, '证据表.parquet'))
    ev['point_id'] = ev.point_id.astype(str)
    rng = np.random.RandomState(20261013)
    def take(m, n):
        g = ev[m]
        return g.iloc[rng.permutation(len(g))[:min(n, len(g))]]
    A = take(ev.verdict == 'out_tidal', 48)
    B = take(ev.coast_qc == 'review_tidal', 30)
    C = ev[ev.verdict == 'out_beach']
    D = take(ev.coast_qc == 'review_nearshore', 30)
    E1 = take((ev.verdict == 'keep') & (ev.coastal_prov == True), 12)
    E2 = take((ev.verdict == 'keep') & (ev.coastal_prov == False), 12)
    E = pd.concat([E1, E2])
    parts = [A.assign(grp='A'), B.assign(grp='B'), C.assign(grp='C'), D.assign(grp='D'), E.assign(grp='E')]
    s = pd.concat(parts, ignore_index=True)
    s['grp'] = s['grp'].astype(str)
    s.to_csv(os.path.join(VAL, '抽样清单.csv'), index=False, encoding='utf-8-sig')
    import z1_south_decid_imagery as Z1
    Z1.OUTD = VAL
    cache = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda r: Z1.one_point(str(r.point_id), float(r.lon), float(r.lat), cache), s.itertuples()))
    errs = [x for x in res if isinstance(x, dict) and 'err' in x]
    print('切片 %d 点，失败 %d，用时 %.0fs' % (len(res), len(errs), time.time() - t0))
    if errs:
        print(errs[:3])
    # 联系表（盲判表头：仅 pid + 经纬度）
    cols = 6
    TH, PAD, HDR = 256, 6, 34
    for tag, sel in [('batch1', s[s.grp.isin(['A', 'B'])]), ('batch2', s[s.grp.isin(['C', 'D', 'E'])])]:
        sel = sel.reset_index(drop=True)
        rows = (len(sel) + cols - 1) // cols
        W = cols * (TH + PAD) + PAD
        H = rows * (TH + PAD + HDR) + PAD
        sheet = Image.new('RGB', (W, H), (18, 18, 18))
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
        except Exception:
            font = ImageFont.load_default()
        for i, r in sel.iterrows():
            fp = os.path.join(VAL, 'chips', r.point_id + '.png')
            x = PAD + (i % cols) * (TH + PAD)
            y = PAD + (i // cols) * (TH + PAD + HDR)
            if os.path.exists(fp):
                im = Image.open(fp).convert('RGB').resize((TH, TH))
                sheet.paste(im, (x, y))
            d.text((x + 2, y + TH + 4), '#%d  %s  (%.4f, %.4f)' % (i + 1, r.point_id, r.lon, r.lat),
                   fill=(255, 255, 0), font=font)
        sheet.save(os.path.join(VAL, 'sheet_%s.png' % tag))
        print('联系表 %s：%d 点 → %dx%d' % (tag, len(sel), sheet.size[0], sheet.size[1]))
    with open(os.path.join(VAL, '判读说明.md'), 'w', encoding='utf-8') as f:
        f.write('''# 湖河滩地·海岸修复独立判读说明（盲判）

你会看到两张联系表（`sheet_batch1.png` / `sheet_batch2.png`），每格是一处样点的 z18 高清影像切片
（512² 中心十字丝＝样点位置，青色方框≈10 m 尺度；切片来自 Esri World Imagery）。
**表头只有点号与经纬度，不含任何分组信息（盲判）。** 每张联系表上方/下方的编号 #N 与点号一一对应。

## 判读类别（四选一，可加"不确定"）
| 代码 | 类别 | 判据（目视） |
|---|---|---|
| T | 潮滩/海岸潮间带 | 大面积泥滩/沙滩与海水直接相连；可见潮沟（树枝状水槽）；无植被的湿滩面；河口外缘感潮滩面 |
| S | 沙滩/沙丘（高潮线以上） | 干燥沙面/沙丘、常见沙脊波纹或沙生植被斑；与海水之间隔着滩脊或高差明显 |
| F | 河湖滩地（内陆） | 内陆河/湖边：沙洲、心滩、消落带岸带；水体为河道/湖泊形态，无潮沟 |
| O | 其它 | 农田、建筑、山体、森林、养殖塘等；或影像看不出滩地特征 |
| ? | 不确定 | 云雪遮挡、影像过暗或尺度不够 |

## 输出格式（CSV，逐行）
编号,点号,判读,置信度(0-1),一句话依据
1,1753184,T,0.95,明显潮沟与海相连

把结果写成 `判读结果_batch1.csv` / `判读结果_batch2.csv` 放在本目录。
''')
    print('→ %s' % VAL)


if __name__ == '__main__':
    main()
