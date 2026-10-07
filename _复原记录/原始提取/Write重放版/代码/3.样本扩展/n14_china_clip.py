# -*- coding: utf-8 -*-
"""
n14_china_clip.py — v4 国界裁剪修复（清除 GLC_FCS10 境外采样污染）
* 问题: n8 采样用矩形 bbox（73-136E/17-54N）未做国界裁剪，蒙古/俄罗斯/朝鲜半岛/琉球等
  境外点混入（src='glc_fcs10_2023' 为主，esri_nat 边缘瓦片可能有零星）。
* 修复: 对全表做中国国界（DataV 34省级 union）contains 过滤，境外点移入
  foreign_removed.parquet 存档（不删除，供审计）。
* 输出: cn_samples_v4.parquet（原地修正）+ foreign_removed.parquet
"""
import sys, os, json
import numpy as np
import pandas as pd
import shapely
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
BOUND = r'数据/边界/china_100000_full.json'
FOREIGN = '数据/本地处理/样本底座/foreign_removed.parquet'

def load_polys():
    d = json.load(open(BOUND, encoding='utf-8'))
    ps = []
    for f in d['features']:
        sg = shapely.geometry.shape(f['geometry'])
        ps += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return ps

def main():
    df = pd.read_parquet(V4)
    n0 = len(df)
    polys = load_polys()
    xs = df['lon'].to_numpy(); ys = df['lat'].to_numpy()
    from shapely import contains_xy
    keep = np.zeros(len(df), dtype=bool)
    m = (xs >= 70) & (xs <= 138) & (ys >= 15) & (ys <= 56)
    km = np.zeros(m.sum(), dtype=bool)
    sub_x, sub_y = xs[m], ys[m]
    for p in polys:
        km |= contains_xy(p, sub_x, sub_y)
    keep[m] = km
    keep[~m] = False
    foreign = df[~keep]
    print('原点数:', f'{n0:,}', ' 境外点:', f'{len(foreign):,}',
          f'({100*len(foreign)/n0:.2f}%)')
    print('境外点来源分布:', dict(foreign.src.value_counts().head(8)))
    # 境外点存档
    foreign.to_parquet(FOREIGN, index=False)
    # 原地修正
    df2 = df[keep].reset_index(drop=True)
    df2.to_parquet(V4, index=False)
    print('修正后:', f'{len(df2):,}')
    json.dump({'before': int(n0), 'foreign_removed': int(len(foreign)),
               'after': int(len(df2)),
               'foreign_by_src': {k: int(v) for k, v in foreign.src.value_counts().items()}},
              open(V4.replace('.parquet', '_clip_summary.json'), 'w'),
              ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
