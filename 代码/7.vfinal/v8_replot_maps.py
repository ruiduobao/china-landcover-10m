# -*- coding: utf-8 -*-
"""replot.py — 用中文字体重绘端到端分类图（原图 DejaVu Sans 缺 CJK 字形）"""
import os, sys, glob
import numpy as np, pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/0.本地流水线')
from lc_conf import CLASSES
OUTD = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练/端到端验证'
PALETTE = ['#1b9e77','#66a61e','#e6ab02','#a6761d','#d95f02','#7570b3','#e7298a','#1f78b4',
           '#33a02c','#b15928','#006400','#7fc97f','#41ab5d','#00441b','#78c679','#c7e9c0',
           '#ffffcc','#fdd49e','#d9d9d9','#969696','#525252','#bdbdbd','#fdbf6f','#a63603',
           '#ffeda0','#c6dbef','#3182bd','#08519c','#d73027','#4575b4','#8c6bb1']
for tif in sorted(glob.glob(os.path.join(OUTD, '*_class.tif'))):
    base = os.path.basename(tif).replace('_class.tif', '')
    with rasterio.open(tif) as s:
        lab = s.read(1); tr = s.transform; hh, ww = s.height, s.width
    classes = sorted(int(c) for c in np.unique(lab) if c > 0)
    fwd = np.zeros(max(classes) + 1, np.int16)
    for i, c in enumerate(classes):
        fwd[c] = i
    idx = np.zeros_like(lab, np.int16)
    m = lab > 0
    idx[m] = fwd[lab[m]]
    cmap = ListedColormap(PALETTE[:len(classes)])
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8.6), dpi=170)
    ax.imshow(np.ma.masked_where(lab == 0, idx), cmap=cmap, norm=norm,
              extent=[tr.c, tr.c + ww * tr.a, tr.f + hh * tr.e, tr.f], interpolation='nearest')
    ax.set_title(f'v-final 分类图 · {base} · 2022 · 30 m（窗口 0.3°×0.3°）', fontsize=13)
    ax.set_xlabel('经度'); ax.set_ylabel('纬度')
    ax.grid(alpha=0.15, linestyle=':')
    hi = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(len(classes))]
    ax.legend(hi, [f'{c} {CLASSES[c][1]}' for c in classes], loc='center left',
              bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    png = os.path.join(OUTD, f'{base}_map.png')
    plt.savefig(png, bbox_inches='tight'); plt.close()
    print('重绘', png, len(classes), '类')
