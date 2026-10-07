# -*- coding: utf-8 -*-
"""
e9_class_maps.py — 每个类别样本点全国分布图（30 类）
* 样本来源: r7_train.parquet（母库 2,241,785 点）+ 稀有类补样 r7_rare_samples.parquet（4,677 点）
* 底图: 数据/边界/china_100000_full.json（DataV 省级）
* 产出（数据/样本交付/分布图/）:
    class_{code}_{name}.png        每类单独一张（>5万点抽稀到 5 万）
    全部类别_全国分布_6x5.png      30 类 6×5 面板总图
    全部类别_总览_彩色.png         一张图 30 类着色
* 用法: python e9_class_maps.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import geopandas as gpd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
from lc_conf import CLASSES, LEVEL0, class_name

WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
RARE = os.path.join(PROJ, '数据/本地处理/全国清洗训练/稀有类补样/r7_rare_samples.parquet')
BOUND = os.path.join(PROJ, '数据/边界/china_100000_full.json')
OUT = os.path.join(PROJ, '数据/样本交付/分布图')
os.makedirs(OUT, exist_ok=True)
MAX_PTS = 50000          # 单类绘图上限（抽稀）
XLIM, YLIM = (72, 136), (16, 55)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# level0 组 → 颜色（同组同色系，便于识别）
L0_COLOR = {'crop': '#d9a441', 'forest': '#2e7d32', 'shrub': '#8bc34a',
            'grass': '#9ccc65', 'wetland': '#26a69a', 'imperv': '#c62828',
            'bare': '#a1887f', 'sparse': '#bdbdbd', 'water': '#1e88e5', 'snow': '#90caf9'}


def load_samples():
    t = pd.read_parquet(os.path.join(WORKR, 'r7_train.parquet'),
                        columns=['row_id', 'lon', 'lat', 'class_new'])
    if 'row_id' not in t.columns:            # 兼容旧文件（现母库带显式 row_id）
        t['row_id'] = np.arange(len(t))
    if os.path.isfile(RARE):
        r = pd.read_parquet(RARE, columns=['lon', 'lat', 'class_new', 'row_id'])
        t = pd.concat([t, r], ignore_index=True)
    # 应用错位样本剔除清单（省级白名单违规，r10_purge_misplaced）
    pl = os.path.join(PROJ, '数据/本地处理/全国清洗训练/错位样本剔除清单.parquet')
    if os.path.isfile(pl):
        bad = set(pd.read_parquet(pl, columns=['row_id']).row_id.astype(np.int64))
        t = t[~t.row_id.isin(bad)]
        print(f'应用剔除清单: -{len(bad):,} 点', flush=True)
    t = t.dropna(subset=['lon', 'lat'])
    t = t[t.class_new > 0]
    return t


def load_boundary():
    d = json.load(open(BOUND, encoding='utf-8'))
    g = gpd.GeoDataFrame.from_features(d['features'], crs='EPSG:4326')
    return g


def draw_base(ax, bnd):
    bnd.plot(ax=ax, facecolor='#f7f7f7', edgecolor='#9e9e9e', linewidth=0.35, zorder=0)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#cccccc')


def main():
    t0 = time.time()
    print('载入样本…', flush=True)
    s = load_samples()
    print(f'样本 {len(s):,} 点，{s.class_new.nunique()} 类', flush=True)
    bnd = load_boundary()

    codes = sorted(CLASSES)
    ncol, nrow = 5, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 3.0), dpi=150)
    axes = axes.ravel()

    # 各类点数统计表（供论文/核对）
    cnt = s.class_new.value_counts()
    ttl = s.class_new.value_counts().sum()
    rows = [{'class': int(c), 'name': class_name(c), 'level1': CLASSES[c][2],
             'n': int(cnt.get(c, 0)), 'share': round(float(cnt.get(c, 0)) / ttl, 6),
             'png': f'class_{c}_{class_name(c)}.png'} for c in codes]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, '各类点数统计.csv'),
                              index=False, encoding='utf-8-sig')
    print('统计表 → 各类点数统计.csv', flush=True)

    # 每类单独图 + 面板
    for i, c in enumerate(codes):
        g = s[s.class_new == c]
        n = len(g)
        if n > MAX_PTS:
            g = g.sample(MAX_PTS, random_state=42)
        col = L0_COLOR.get(CLASSES[c][2], '#1976d2')
        ax = axes[i]
        draw_base(ax, bnd)
        if n:
            ax.scatter(g.lon, g.lat, s=0.6, c=col, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(f'{c} {class_name(c)}\nn={n:,}', fontsize=8.5, pad=2)
        # 单独一张
        fp = os.path.join(OUT, f'class_{c}_{class_name(c)}.png')
        fig1, ax1 = plt.subplots(figsize=(7.2, 6.0), dpi=180)
        draw_base(ax1, bnd)
        if n:
            ax1.scatter(g.lon, g.lat, s=1.2, c=col, alpha=0.6, linewidths=0, rasterized=True)
        ax1.set_title(f'{c} {class_name(c)}  全国样本分布  n={n:,}', fontsize=12)
        fig1.tight_layout()
        fig1.savefig(fp, bbox_inches='tight')
        plt.close(fig1)
        print(f'  {c} {class_name(c)} n={n:,} → {os.path.basename(fp)}', flush=True)

    # 未用到的子图隐藏
    for j in range(len(codes), len(axes)):
        axes[j].axis('off')
    fig.suptitle('中国 10m 土地覆盖样本点 30 类全国分布（r7 母库 + 稀有类补样）',
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp_panel = os.path.join(OUT, '全部类别_全国分布_6x5.png')
    fig.savefig(fp_panel, bbox_inches='tight')
    plt.close(fig)
    print('面板图 →', fp_panel, flush=True)

    # 总览彩色图
    fig2, ax2 = plt.subplots(figsize=(14, 11), dpi=180)
    draw_base(ax2, bnd)
    cmap = plt.colormaps['turbo'].resampled(len(codes))
    for i, c in enumerate(codes):
        g = s[s.class_new == c]
        if len(g) > MAX_PTS:
            g = g.sample(MAX_PTS, random_state=42)
        if len(g):
            ax2.scatter(g.lon, g.lat, s=0.35, c=[cmap(i)], alpha=0.45,
                        linewidths=0, rasterized=True,
                        label=f'{c} {class_name(c)}')
    ax2.set_title('中国 10m 土地覆盖样本点 30 类全国总览（r7 母库 + 稀有类补样）', fontsize=15)
    ax2.legend(markerscale=14, fontsize=6.2, ncol=3, loc='lower left',
               framealpha=0.85, columnspacing=0.8, handletextpad=0.3)
    fig2.tight_layout()
    fp_all = os.path.join(OUT, '全部类别_总览_彩色.png')
    fig2.savefig(fp_all, bbox_inches='tight')
    plt.close(fig2)
    print('总览图 →', fp_all, flush=True)
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
