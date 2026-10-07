# -*- coding: utf-8 -*-
"""
n13_by_class.py — 按类别出分布图（省级底图+标题）+ 输出各类 gpkg
* 输入: cn_samples_v4.parquet + CTAMAP 2026 省级 shp
* 输出: 数据/样本交付/by_class/
    ├── class_XX_中文名.png   （点分布图，省界底图，含标题）
    └── class_XX_中文名.gpkg  （该类全部样本点）
* 用法: python n13_by_class.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'代码/0.本地流水线')
from lc_conf import CLASSES

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
PROV_SHP = r'E:\地理所\论文\中国人口密度2000-2026\1.数据\1.1_CTAMAP\2026\省级\T2026年初省级.shp'
OUT = r'数据/样本交付/by_class'
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    print('读取数据…', flush=True)
    df = pd.read_parquet(V4, columns=['lon', 'lat', 'class_new', 'tier', 'year'])
    prov = gpd.read_file(PROV_SHP)
    prov = prov.to_crs(4326) if prov.crs.to_epsg() != 4326 else prov

    classes = sorted(df.class_new.unique())
    print(f'类别数: {len(classes)}', flush=True)
    for i, cls in enumerate(classes):
        sub = df[df.class_new == cls]
        en, cn, l1, l0 = CLASSES.get(int(cls), ('?', f'类{cls}', '?', '?'))
        g = gpd.GeoDataFrame(
            sub[['class_new', 'tier', 'year']].copy(),
            geometry=gpd.points_from_xy(sub.lon, sub.lat), crs=4326)
        # gpkg
        gpf = os.path.join(OUT, f'class_{cls}_{cn}.gpkg')
        g.to_file(gpf, layer='samples', driver='GPKG')
        # 分布图
        fig, ax = plt.subplots(figsize=(11, 9))
        prov.boundary.plot(ax=ax, color='#444444', linewidth=0.6)
        ax.scatter(sub.lon, sub.lat, s=0.4, c='#c0392b', alpha=0.35,
                   rasterized=True, edgecolors='none')
        ax.set_title(f'{cls} {cn}（{en}）样本点分布  N={len(sub):,}',
                     fontsize=14, pad=12)
        ax.set_xlabel('经度 (°E)'); ax.set_ylabel('纬度 (°N)')
        ax.set_xlim(73, 137); ax.set_ylim(16, 55)
        ax.set_aspect(1.25)
        fig.text(0.99, 0.01, f'CNLC10 v4 样本底座 · {time.strftime("%Y-%m-%d")}',
                 ha='right', fontsize=8, color='#666666')
        png = os.path.join(OUT, f'class_{cls}_{cn}.png')
        fig.savefig(png, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f'[{i+1}/{len(classes)}] {cls} {cn}: {len(sub):,} 点 -> png+gpkg', flush=True)
    print(f'完成，用时 {(time.time()-t0)/60:.1f} 分钟 -> {OUT}')

if __name__ == '__main__':
    main()

