# -*- coding: utf-8 -*-
"""make_figures.py — 生成 README 配图（全部由 _试跑 真实影像拼合，可复现）

* 输入：_试跑/轮1_东北作物/（chips、sheets、s2 时序图）
* 输出：assets/fig1_流程.png、fig2_切片解剖.png、fig3_三作物物候.png、fig4_质控协议.png
* 说明：所有文本走 wrap()/fit() 自适应（按像素宽换行与缩字号），避免溢出——首版曾出现卡片文字被裁。
* 用法：python assets/make_figures.py   （在 skill 根目录执行）
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, '_试跑', '轮1_东北作物')
FONT = None
for c in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
    if os.path.exists(c):
        FONT = c
        break
FONTB = 'C:/Windows/Fonts/msyhbd.ttc' if os.path.exists('C:/Windows/Fonts/msyhbd.ttc') else FONT
BG = (255, 255, 255)
INK = (32, 38, 46)
ACC = (31, 111, 235)
ACC2 = (16, 133, 90)
AMBER = (150, 96, 20)
RED = (178, 34, 34)
GREY = (110, 120, 130)


def f(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def tw(d, s, font):
    return d.textbbox((0, 0), s, font=font)[2]


def fit(d, s, font, maxw):
    """字号自适应：从给定字号开始，逐级缩小直到文本宽度 ≤ maxw。"""
    sz = font.size
    while sz > 10 and tw(d, s, f(sz, True)) > maxw:
        sz -= 1
    return s, f(sz, True)


def wrap(d, s, font, maxw):
    """按像素宽换行（中文逐字，西文按空格）；返回行列表。"""
    lines, cur = [], ''
    for ch in s:
        if ch == '\n':
            lines.append(cur)
            cur = ''
            continue
        if tw(d, cur + ch, font) > maxw and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def wrap_all(d, rows, maxw, fontmaker):
    kept = [r for r in rows if os.path.exists(r[0])]
    wmax = max([Image.open(r[0]).width for r in kept] + [900]) + 40
    H = 96
    plan = []
    for p, title, sub, col in kept:
        t = Image.open(p)
        tl = wrap(d, title, f(20, True), wmax - 60)
        sl = wrap(d, sub, f(17), wmax - 60)
        h = t.height + 16 + 28 * len(tl) + 24 * len(sl) + 18
        plan.append((t, tl, sl, col, h))
        H += h
    H += 60
    im = Image.new('RGB', (wmax, H), BG)
    d = ImageDraw.Draw(im)
    return im, d, plan, wmax, H


def fig1():
    W = 1440
    steps = [
        ('① 输入与质检', 'prep_points.py', ['点位表 CSV/SHP/XLSX', '坐标系检查·去重·分层', '→ 表单骨架 + 盲判清单']),
        ('② 高分切片', 'fetch_chips.py', ['Esri 主源 / Google 备源', 'z18≈0.5 m，3×3 拼图', '十字丝 + 10m 目标框']),
        ('③ 时序物候', 's2_timeseries.py', ['哨兵-2 生长季 N 期', '逐窗云掩膜 → NDVI/NDWI', '振幅·峰值日·淹水证据']),
        ('④ 盲判批阅', 'build_sheets.py', ['2×2 联系表（512 原生）', '批次只给编号（防锚定）', '逐点：类别+置信+证据']),
        ('⑤ 质控协议', 'qc_protocol.py', ['复判 ≥10% + 低置信全查', '随机"否定"审计', '→ 最小复核清单']),
        ('⑥ 统计报告', 'merge_stats.py', ['确认率（总/分组）', '先验×判读混淆·敏感性', '复判一致率']),
        ('⑦ GIS 交付', 'points_to_shp.py', ['SHP：WGS84 + GCJ-02', 'ASCII 字段名 + 对照表', 'UTF-8(.cpg) 可直接打开']),
    ]
    bw, bh, gap = 330, 168, 18
    rows = [steps[:4], steps[4:]]
    notes = ['硬规则：① 盲判（批次不含先验标签） ② 每条结论必带证据+置信 ③ "无法判读"单列，不并入分母 ④ 没有日期就没有作物',
             '实测速度：高分切片 ≈1.6 s/点（并发 8）   ｜   哨兵时序 1.5–3 min/点（免密钥跨境 S3，>30 点建议分批或走 GEE）',
             '真数据试跑：东北 24 点（玉米/水稻/大豆）——切片 39 s 全出；三作物物候可分（fig3）；质控链路全通（fig4）']
    H = 150 + 2 * (bh + 40) + 120
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 34), '遥感样本点验证与地物类型标记 · 全流程', font=f(38, True), fill=INK)
    d.text((48, 88), 'rs-sample-labeling skill：从点位表到可复核判读结论', font=f(21), fill=GREY)
    y = 140
    for ri, row in enumerate(rows):
        for ci, (t, s, bull) in enumerate(row):
            x = 48 + ci * (bw + gap)
            col = ACC if (ri == 0 and ci < 3) else (ACC2 if (ri == 0) else AMBER)
            d.rounded_rectangle((x, y, x + bw, y + bh), 14, fill=(246, 249, 253), outline=col, width=3)
            d.rounded_rectangle((x, y, x + bw, y + 42), 14, fill=col)
            d.rectangle((x, y + 28, x + bw, y + 42), fill=col)
            d.text((x + 14, y + 8), t, font=f(21, True), fill=(255, 255, 255))
            d.text((x + 14, y + 50), s, font=f(15), fill=col)
            for k, line in enumerate(bull):
                d.text((x + 14, y + 78 + k * 26), line, font=f(15), fill=INK)
            if ci < len(row) - 1:
                ax = x + bw + 3
                d.polygon([(ax, y + bh / 2 - 8), (ax + 11, y + bh / 2), (ax, y + bh / 2 + 8)], fill=GREY)
        y += bh + 40
    yy = y + 6
    for i, n in enumerate(notes):
        for k, line in enumerate(wrap(d, n, f(19 if i == 0 else 17, i == 0), W - 96)):
            d.text((48, yy), line, font=f(19 if i == 0 else 17, i == 0), fill=RED if i == 0 else GREY)
            yy += 28
        yy += 6
    im = im.crop((0, 0, W, yy + 20))
    im.save(os.path.join(HERE, 'fig1_流程.png'))
    print('fig1_流程.png', im.size)


def fig2():
    W, H = 1440, 700
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 30), '切片长什么样：十字丝不遮靶心，青框 = 目标像元（10 m）', font=f(32, True), fill=INK)
    chip = Image.open(os.path.join(RUN, 'chips', 'NER004.png')).resize((420, 420))
    im.paste(chip, (48, 100))
    d.rectangle((48, 100, 468, 520), outline=(200, 200, 200), width=2)
    cx, cy = 48 + 210, 100 + 210
    ann = [((cx + 100, cy - 100), (474, 132), ['十字丝：中心留空 48 px，', '不遮挡被判的那个像元']),
           ((cx + 24, cy + 24), (474, 190), ['青色框 = 目标像元尺寸', '（10 m，按纬度换算像素）']),
           ((66, 118), (48, 536), ['左上角编号：', '盲判只给编号']),
           ((300, 470), (300, 536), ['格田 + 红褐田埂 + 防护林', '= 水稻的教科书形态'])]
    for (px, py), (tx, ty), lines in ann:
        d.line((px, py, tx, ty), fill=RED, width=2)
        for k, line in enumerate(lines):
            d.text((tx, ty + k * 22), line, font=f(15), fill=RED)
    caps = [('NEM003.png', ['收获后：割茬 + 车辙', '（无日期底图会误判成裸地）']),
            ('NEM004.png', ['条带田 + 防护林', '（作物种植结构）']),
            ('NER003.png', ['均质绿冠层：只能到"耕地"', '作物种级必须看时序'])]
    x = 700
    for name, cap in caps:
        p = os.path.join(RUN, 'chips', name)
        if os.path.exists(p):
            t = Image.open(p).resize((230, 230))
            im.paste(t, (x, 130))
            d.rectangle((x, 130, x + 230, 360), outline=(200, 200, 200), width=2)
            for k, line in enumerate(cap):
                d.text((x, 374 + k * 24), line, font=f(15), fill=INK)
            x += 246
    d.text((700, 460), '结论：高分图判"结构与边界"，作物种级判"日期与物候"——缺一不可。', font=f(19, True), fill=ACC2)
    d.text((700, 500), 'R1 实测：24 点里水稻格田一眼可判（NER004），而玉米/大豆的', font=f(16), fill=GREY)
    d.text((700, 526), '同季均匀冠层无法区分——这条缺口直接催生了第 ③ 步哨兵时序模块。', font=f(16), fill=GREY)
    im.save(os.path.join(HERE, 'fig2_切片解剖.png'))
    print('fig2_切片解剖.png', im.size)


def fig3():
    rows = [(os.path.join(RUN, 's2', 'NEM003_时序.png'), '玉米（先验 ne_crops_maize_2023_HLJ）',
             'NDVI 峰值 07-31（居中）｜振幅 0.71｜9 月底收割回落，留高茬', AMBER),
            (os.path.join(RUN, 's2', 'NER006_时序.png'), '水稻（先验 ne_crops_rice_2023_HLJ）',
             'NDVI 峰值 07-13（最早）｜振幅 0.67｜NDWI 前期高＝泡田—封行节律', ACC2),
            (os.path.join(RUN, 's2', 'NES001_时序.png'), '大豆（先验 ne_crops_soybean_2023_HLJ）',
             'NDVI 峰值 08-07（最晚）｜振幅 0.71｜9 月中旬叶片先黄（油料作物早衰）', ACC)]
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    W = max(Image.open(r[0]).width for r in rows) + 80
    W = max(W, 1040)
    inner = W - 84                       # 卡片内文字可用宽度
    head = wrap(probe, '三作物物候可分：峰值日期 + 成熟期颜色（东北 2023 实测）', f(28, True), W - 60)
    plan = []
    for p, title, sub, col in rows:
        t = Image.open(p)
        tl = wrap(probe, title, f(20, True), inner)
        sl = wrap(probe, sub, f(17), inner)
        h = 12 + t.height + 8 + 26 * len(tl) + 22 * len(sl) + 12
        plan.append((t, tl, sl, col, h))
    note = wrap(probe, '判读链：峰值最早→水稻；居中＋留高茬→玉米；最晚＋先黄→大豆。与高分图结构（格田/行距/茬高）互证后定案。',
                f(18, True), W - 60)
    H = 24 + 40 * len(head) + 16 + sum(h + 16 for *_, h in plan) + 28 * len(note) + 30
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 24
    for line in head:
        d.text((30, y), line, font=f(28, True), fill=INK)
        y += 40
    y += 16
    for (t, tl, sl, col, h) in plan:
        d.rounded_rectangle((24, y, 24 + t.width + 24, y + h), 10, fill=(248, 250, 252), outline=col, width=2)
        im.paste(t, (36, y + 12))
        yy = y + 12 + t.height + 8
        for line in tl:
            d.text((36, yy), line, font=f(20, True), fill=col)
            yy += 26
        for line in sl:
            d.text((36, yy), line, font=f(17), fill=INK)
            yy += 22
        y += h + 16
    y += 12
    for line in note:
        d.text((30, y), line, font=f(18, True), fill=RED)
        y += 28
    im.save(os.path.join(HERE, 'fig3_三作物物候.png'))
    print('fig3_三作物物候.png', im.size)


def fig4():
    W = 1400
    boxes = [
        (60, '盲判批次', ['联系表只给编号', '不含先验类/分层', '先判后对照'], ACC),
        (420, '复判抽样', ['随机 ≥10%', '+ 全部低置信', '+ 全部无法判读', '门槛：一致率 ≥0.90'], ACC2),
        (780, '否定审计', ['随机"否定"点重判', '算"其实是对的"比例', '门槛：准确率 ≥0.90'], AMBER),
        (1120, '最小复核清单', ['分歧+无法+低置信', '人工只核这些', '其余按判读结论'], (120, 60, 160)),
    ]
    stats = [('确认率 0.909', '分母 = 可判读点；无法判读 8.3% 单列'),
             ('复判一致率 0.889', '9 点复判（含 2 点故意不一致，检出 1 点）'),
             ('最小复核清单 8 点', '分歧 1 + 无法判读 2 + 低置信是 5，去重'),
             ('敏感性', '推翻"≤0.40"判定需 36.3% 的"否"错判（实测 7.1%）')]
    y0, bh = 128, 190
    H = y0 + bh + 84 + 2 * 78 + 60
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 30), '质控协议：让"判读结论"变成"可复核证据"', font=f(34, True), fill=INK)
    for x, t, items, col in boxes:
        d.rounded_rectangle((x, y0, x + 250, y0 + bh), 14, fill=(248, 250, 252), outline=col, width=3)
        d.rounded_rectangle((x, y0, x + 250, y0 + 42), 14, fill=col)
        d.rectangle((x, y0 + 28, x + 250, y0 + 42), fill=col)
        d.text((x + 14, y0 + 8), t, font=f(20, True), fill=(255, 255, 255))
        for k, s in enumerate(items):
            d.text((x + 14, y0 + 58 + k * 28), '· ' + s, font=f(16), fill=INK)
    d.text((60, y0 + bh + 36), '真数据试跑结果（24 点干跑）', font=f(22, True), fill=INK)
    y = y0 + bh + 80
    for i, (k, v) in enumerate(stats):
        x = 60 + (i % 2) * 660
        yy = y + (i // 2) * 78
        d.rounded_rectangle((x, yy, x + 620, yy + 62), 10, fill=(240, 246, 255), outline=(190, 210, 235))
        d.text((x + 16, yy + 8), k, font=f(20, True), fill=ACC)
        d.text((x + 16, yy + 34), v, font=f(15), fill=INK)
    note = '报告口径强制声明：确认率/一致率是判读结果≠精度；AI 判读须标注是否经人工复核；训练池抽点只评标签质量。'
    for line in wrap(d, note, f(18, True), W - 120):
        d.text((60, H - 46), line, font=f(18, True), fill=RED)
    im.save(os.path.join(HERE, 'fig4_质控协议.png'))
    print('fig4_质控协议.png', im.size)


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
