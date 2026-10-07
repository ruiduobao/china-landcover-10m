# -*- coding: utf-8 -*-
"""t_colorchart.py — 生成"看得见颜色"的对照材料
产物（试点成果_2023瓦片/）：
  地类颜色表.png    24 类色卡图（编码+名称+HEX，直接看图即可）
  地类颜色对照.html 可双击打开的网页色卡（含每瓦片逐类面积彩色条形）
  地类颜色对比_大图.png  每瓦片主导类彩色条带（便于横向比较）
"""
import os, sys, csv, html
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from d_tiles import COLORS

DST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/试点成果_2023瓦片'
TILES = {'qinling': '森林（秦岭西）', 'qinling_e': '森林（秦岭东）',
         'songnen': '农林（松嫩）', 'sanjiang': '湿地（三江）'}
NAMES = VC.V31_NAMES()


def hex_of(k):
    r, g, b = [round(x * 255) for x in COLORS[k]]
    return '#%02X%02X%02X' % (r, g, b)


def chart_png():
    ks = sorted(COLORS)
    ncol = 4
    nrow = (len(ks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.35 * nrow))
    plt.subplots_adjust(hspace=0.35, wspace=0.06, top=0.94, bottom=0.03)
    for i, k in enumerate(ks):
        ax = axes[i // ncol][i % ncol]
        ax.add_patch(Rectangle((0.02, 0.08), 0.30, 0.84, facecolor=COLORS[k],
                               edgecolor='#555', linewidth=1.2))
        ax.text(0.36, 0.66, '%d  %s' % (k, NAMES.get(str(k), '?')), fontsize=13, va='center')
        r, g, b = [round(x * 255) for x in COLORS[k]]
        ax.text(0.36, 0.26, '%s   RGB(%d,%d,%d)' % (hex_of(k), r, g, b), fontsize=10,
                va='center', color='#444')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.suptitle('试点成果 · 24 类地表覆盖 颜色对照表（2023，10m 栅格像元值 = 编码）', fontsize=16)
    fp = os.path.join(DST, '地类颜色表.png')
    plt.savefig(fp, dpi=110); plt.close()
    VC.emit('色卡图 → %s' % fp)


def bars_png():
    """四个子图：每瓦片一张逐类面积条形（类色 + 该瓦片专属底纹，图例明确）"""
    data = {}
    for t in TILES:
        fp = os.path.join(DST, '%s_面积.csv' % t)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8-sig') as f:
                data[t] = {int(r['code']): float(r['area_km2']) for r in csv.DictReader(f)}
    if not data:
        return
    ks = sorted(set().union(*[set(d) for d in data.values()]))
    hatches = ['', '///', '...', 'xxx']
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 0.40 * len(ks) + 2.4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, t, hh in zip(axes, list(data.keys()), hatches):
        note = TILES.get(t, t)
        vals = [data[t].get(k, 0) for k in ks]
        ax.barh(range(len(ks)), vals, color=[COLORS[k] for k in ks],
                edgecolor='#333', linewidth=0.6, hatch=hh)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(['%d %s' % (k, NAMES.get(str(k), '?')) for k in ks], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(t + ' ' + note, fontsize=12)
        ax.set_xlabel('面积（km²）', fontsize=10)
        ax.grid(axis='x', alpha=0.25)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v, i, ' %.0f' % v, va='center', fontsize=8, color='#333')
    fig.suptitle('四块试点瓦片 · 逐类面积（条形颜色 = 该地类在 10m 栅格里的显示颜色）', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(DST, '地类面积对比_彩图.png')
    plt.savefig(fp, dpi=105); plt.close()
    VC.emit('面积对比图 → %s' % fp)


def html_doc():
    rows_html = []
    for k in sorted(COLORS):
        r, g, b = [round(x * 255) for x in COLORS[k]]
        rows_html.append(
            '<tr><td class="c">%d</td>'
            '<td><span class="sw" style="background:%s"></span></td>'
            '<td>%s</td><td>%s</td><td class="m">%d, %d, %d</td></tr>' % (
                k, hex_of(k), NAMES.get(str(k), '?'), hex_of(k), r, g, b))
    # 每瓦片逐类面积条
    sections = []
    for t, note in TILES.items():
        fp = os.path.join(DST, '%s_面积.csv' % t)
        if not os.path.exists(fp):
            continue
        with open(fp, encoding='utf-8-sig') as f:
            rows = sorted(csv.DictReader(f), key=lambda r: -float(r['area_km2']))
        tot = sum(float(r['area_km2']) for r in rows) or 1
        items = []
        for r in rows:
            k = int(r['code']); share = 100 * float(r['area_km2']) / tot
            items.append(
                '<div class="row"><span class="sw" style="background:%s"></span>'
                '<span class="lbl">%d %s</span>'
                '<span class="bar"><i style="width:%.1f%%;background:%s"></i></span>'
                '<span class="num">%s km² (%.1f%%)</span></div>' % (
                    hex_of(k), k, NAMES.get(str(k), '?'), min(share * 1.6, 100), hex_of(k),
                    r['area_km2'], share))
        sections.append('<h3>%s <small>%s</small></h3>%s' % (t, note, '\n'.join(items)))
    h = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>试点成果 · 地类颜色与面积对照</title>
<style>
 body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:28px;color:#222;background:#fafafa}
 h1{font-size:22px} h3{margin-top:26px;border-left:5px solid #2659D9;padding-left:9px}
 table{border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}
 td,th{border:1px solid #e2e2e2;padding:7px 12px;font-size:14px}
 th{background:#f0f4fb}
 .sw{display:inline-block;width:34px;height:18px;border:1px solid #666;border-radius:3px;vertical-align:middle}
 .c{font-weight:700;text-align:center;width:52px}
 .m{color:#666;font-family:Consolas,monospace}
 .row{display:flex;align-items:center;gap:9px;margin:3px 0;font-size:13px}
 .lbl{width:150px} .num{color:#555;width:150px;text-align:right}
 .bar{flex:1;background:#eee;height:14px;border-radius:3px;overflow:hidden}
 .bar i{display:block;height:100%}
 small{color:#777;font-weight:400;font-size:13px}
 p.note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:10px 12px;font-size:13px}
</style></head><body>
<h1>试点成果 · 地类颜色与面积对照（2023 年，10m）</h1>
<p class="note">像元值 = 表中「编码」。<b>0 = 无数据</b>（文件已设 nodata，查看器中显示为透明）。<br>
打开 GeoTIFF 即按下列颜色显示（调色板已嵌入文件）；如需手动套色，导入同目录的 <code>地类颜色表.clr</code>。<br>
<b>提醒：请看本网页或 <code>地类颜色表.png</code> 认颜色——Markdown 文件里的色块只是字符，显示不出真实颜色。</b></p>
<h2>一、编码 ↔ 颜色 ↔ 地类</h2>
<table><tr><th>编码</th><th>颜色</th><th>地类名称</th><th>HEX</th><th>RGB(0–255)</th></tr>
__ROWS__</table>
<h2>二、各瓦片逐类面积（条长按占比，颜色即地图上的颜色）</h2>
__SECTIONS__
</body></html>"""
    h = h.replace('__ROWS__', chr(10).join(rows_html)).replace('__SECTIONS__', chr(10).join(sections))
    fp = os.path.join(DST, '地类颜色对照.html')
    open(fp, 'w', encoding='utf-8').write(h)
    VC.emit('网页色卡 → %s' % fp)


def main():
    chart_png(); bars_png(); html_doc()
    # 修正 md：明确说明颜色看哪里
    fp = os.path.join(DST, '地类颜色对照.md')
    if os.path.exists(fp):
        s = open(fp, encoding='utf-8').read()
        warn = ('> ⚠️ **颜色请看 [`地类颜色表.png`](地类颜色表.png) 或双击 [`地类颜色对照.html`](地类颜色对照.html)**——'
                'Markdown 里的色块只是文字符号，显示不出真实颜色。\n\n')
        if '颜色请看' not in s:
            s = s.replace('> 适用文件：', warn + '> 适用文件：', 1)
            open(fp, 'w', encoding='utf-8').write(s)
            VC.emit('已在 md 顶部加"看颜色去哪里"提示')


if __name__ == '__main__':
    main()
