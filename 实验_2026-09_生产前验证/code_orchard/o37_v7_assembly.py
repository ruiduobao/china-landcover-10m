# -*- coding: utf-8 -*-
"""o37_v7_assembly.py — v7 乔灌园地 2017–2024 逐年面板装配（长表 + 分年 + 点表 + 统计）

* 输入：data/orchard/v6_yearly/y*/aef_p*.parquet（o35 产出）
* 规则源：长表 = 点 × 8 年；标签继承 2023 锚（多年生作物）；逐点 aef_ok
* 输出：交付目录 v7_逐年面板/
        - 乔灌园地v7_逐年面板_2017-2024.parquet（长表，含 AEF64）
        - 乔灌园地v7_逐年面板_2024.csv 等 8 个分年精简 CSV（无 AEF 列）
        - 乔灌园地v7_点表.csv（131k 点：subtype/grade/srcset/rec）
        - v7_stats.json
* 用法：python o37_v7_assembly.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v7_逐年面板'


def emit(m):
    print(m, flush=True)


def main():
    os.makedirs(DEST, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(ORD, 'v6_yearly', 'y*', 'aef_p*.parquet')))
    emit('分块 %d 个' % len(fs))
    parts = []
    for f in fs:
        y = int(os.path.basename(os.path.dirname(f))[1:])
        d = pd.read_parquet(f)
        d['year'] = y
        parts.append(d)
    long = pd.concat(parts, ignore_index=True)
    long = long.drop(columns=['lon6', 'lat6'], errors='ignore')
    # 柑橘档位修正（2026-10-13：判读 12/165 确认 + 面积 8× 虚高 → 存疑档；块内旧值就地改写，无需重抓）
    long.loc[long.srcset == 'citrus', 'grade'] = 'C_专题产品(存疑)'
    # 去重与体检
    long = long.drop_duplicates(subset=['uid', 'year'])
    n_pts = long.uid.nunique()
    hit = float(long.aef_ok.mean())
    emit('长表 %d 行（%d 点 × %d 年）；AEF 命中 %.4f' % (len(long), n_pts, long.year.nunique(), hit))
    long.to_parquet(os.path.join(DEST, '乔灌园地v7_逐年面板_2017-2024.parquet'), index=False)
    # 分年精简
    slim_cols = ['uid', 'year', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'rec', 'aef_ok']
    for y, g in long.groupby('year'):
        g[slim_cols].to_csv(os.path.join(DEST, '乔灌园地v7_逐年面板_%d.csv' % y),
                            index=False, encoding='utf-8-sig')
    # 点表
    pts = long.drop_duplicates(subset=['uid'])[['uid', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'rec']]
    pts.to_csv(os.path.join(DEST, '乔灌园地v7_点表.csv'), index=False, encoding='utf-8-sig')
    # 统计
    stats = {'n_points': int(n_pts), 'n_rows': int(len(long)), 'years': sorted(long.year.unique().tolist()),
             'aef_hit': round(hit, 4),
             'by_srcset': pts.srcset.value_counts().to_dict(),
             'by_grade': pts.grade.value_counts().to_dict(),
             'by_year_rows': long.groupby('year').size().to_dict(),
             'by_year_hit': long.groupby('year').aef_ok.mean().round(4).to_dict()}
    VC.jsave(stats, os.path.join(DEST, 'v7_stats.json'))
    emit(json.dumps({k: stats[k] for k in ('n_points', 'n_rows', 'aef_hit', 'by_grade')}, ensure_ascii=False, indent=1))
    emit('→ %s' % DEST)


if __name__ == '__main__':
    main()
