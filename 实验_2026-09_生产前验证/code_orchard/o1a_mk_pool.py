# -*- coding: utf-8 -*-
"""o1a_mk_pool.py — 乔灌园地：母库 8 年年度子集园地点提取（含年匹配 AEF64 面板，本地零 GEE）

* 输入：F:/lc_work/年度子集_含稀有类/r7_train_{2017..2024}.parquet（8 年年度子集，含 A00-A63）
* 规则源：class_new==11（30 类口径乔灌园地；GLC 原始码 12→11）
* 门槛：逐点按 row_id 去重；报告每年点数、来源构成、点并集规模
* 输出：data/orchard/mk_orchard_points.csv    （row_id,lon,lat,src 首见,years_has 年份位图）
        data/orchard/mk_orchard_yearly.parquet（row_id,year,lon,lat,src,A00-A63 —— 现有家底逐年面板）
* 用法：python o1a_mk_pool.py
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

SUB = 'F:/lc_work/年度子集_含稀有类/r7_train_%d.parquet'
OUTD = os.path.join(VC.DATA, 'orchard')
YEARS = list(range(2017, 2025))


def main():
    os.makedirs(OUTD, exist_ok=True)
    frames, first_seen = [], {}
    for y in YEARS:
        p = SUB % y
        d = pd.read_parquet(p, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + VC.FEATS)
        o = d[d.class_new == 11].copy()
        o['year'] = y
        print('%d: 园地 %d 点（%s）' % (y, len(o), o.src.value_counts().head(4).to_dict()))
        frames.append(o[['row_id', 'year', 'lon', 'lat', 'src'] + VC.FEATS])
        for r in o.itertuples():
            first_seen.setdefault(int(r.row_id), (float(r.lon), float(r.lat), str(r.src)))
    yl = pd.concat(frames, ignore_index=True)
    yl = yl.drop_duplicates(subset=['row_id', 'year'])
    yl.to_parquet(os.path.join(OUTD, 'mk_orchard_yearly.parquet'), index=False)
    pts = pd.DataFrame([{'row_id': k, 'lon': v[0], 'lat': v[1], 'src': v[2]}
                        for k, v in first_seen.items()])
    pts['src_kind'] = pts.src.str.replace(r'_\d{4}.*', '', regex=True)
    pts.to_csv(os.path.join(OUTD, 'mk_orchard_points.csv'), index=False, encoding='utf-8-sig')
    print('\n并集 %d 点；逐年面板 %d 行' % (len(pts), len(yl)))
    print('逐年覆盖：', yl.groupby('year').size().to_dict())
    print('来源构成：', pts.src_kind.value_counts().head(12).to_dict())


if __name__ == '__main__':
    main()
