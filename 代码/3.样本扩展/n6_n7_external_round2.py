# -*- coding: utf-8 -*-
"""
n6_eglc_hdvl_amur.py — 第二轮外部样本入库
* EGLC 谐和参考（2260 万行 parquet）: 中国 2017-2020 → 类名映射 → eglc_china.parquet
* HDLV-XJ 新疆验证 20,932: 类码表(1耕地/2森林/4草地/5水/7裸地) → 独立验证库存放（不入训练）
* Amur 黑龙江 7,755: 类码未知 → 原样存放待确认
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EXT = '数据/外部样本源'
SRC_BJ = r'数据/边界/china_100000_full.json'

EGLC_MAP = {
    'Natural Semi-Natural Grassland': 130,
    'Cultivated Grassland/Pasture': 130,
    'Water': 202,
    'Developed/Urban': 190,
    'Evergreen Forest': 51,      # 郁闭/疏闭未知，默认取常绿阔叶50系起点，conf 降级
    'Deciduous Forest': 61,
    'Mixed Forest': 91,
    'Wetland': 181,
    'Permanent Snow/Ice': 220,
    'Shrubland': 121,
    'Rock Outcrops': 201,
    'Shifting Sand': 201,
    'Bare Soil': 201,
    'Cropland': 10,
}
EGLC_CONF = {'Evergreen Forest': 0.5, 'Deciduous Forest': 0.5, 'Mixed Forest': 0.5,
             'Wetland': 0.5, 'Shrubland': 0.6}

def load_polys():
    import json, shapely
    d = json.load(open(SRC_BJ, encoding='utf-8'))
    ps = []
    for f in d['features']:
        sg = shapely.geometry.shape(f['geometry'])
        ps += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return ps

def china_filter(xs, ys, polys):
    from shapely import contains_xy
    keep = np.zeros(len(xs), dtype=bool)
    m = (xs >= 73) & (xs <= 136) & (ys >= 17) & (ys <= 54)
    km = np.zeros(m.sum(), dtype=bool)
    km |= contains_xy(polys[0], xs[m], ys[m])
    for p in polys[1:]:
        km |= contains_xy(p, xs[m], ys[m])
    keep[m] = km
    return keep

def eglc():
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(EXT, 'eglc_harmonized_c.parquet'),
                      columns=['dataset', 'class', 'year', 'lon', 'lat'],
                      filters=[('lon', '>=', 73), ('lon', '<=', 136),
                               ('lat', '>=', 17), ('lat', '<=', 54),
                               ('year', '>=', 2017), ('year', '<=', 2020)])
    df = t.to_pandas()
    print('EGLC 中国 2017-2020:', len(df))
    polys = load_polys()
    keep = china_filter(df.lon.to_numpy(), df.lat.to_numpy(), polys)
    df = df[keep].reset_index(drop=True)
    print('国界内:', len(df))
    df['class_new'] = df['class'].map(EGLC_MAP).fillna(-1).astype(int)
    df['src'] = 'eglc_' + df['dataset']
    df['src_conf'] = df['class'].map(lambda c: EGLC_CONF.get(c, 0.8))
    df['year'] = df['year'].astype(int)
    out = df[df.class_new > 0][['lon', 'lat', 'class_new', 'src', 'src_conf', 'year']]
    out.to_parquet(os.path.join(EXT, 'eglc_china.parquet'), index=False)
    print('EGLC 可映射:', len(out), dict(out.class_new.value_counts()))

def hdlv():
    xl = pd.read_excel(os.path.join(EXT, 'HDLV_XJ_ValidationDataset.xls'),
                       sheet_name=0, header=1)
    xl.columns = ['OBJECTID', 'value', 'lon', 'lat'][:len(xl.columns)]
    m = {1: 10, 2: 2, 4: 130, 5: 202, 7: 201}   # 2=森林→存 level0 组码，作验证
    xl['class_new'] = xl['value'].map(m).fillna(-1).astype(int)
    xl['src'] = 'hdlv_xj_2020'
    xl['year'] = 2020
    xl = xl[xl.class_new > 0]
    xl.to_parquet(os.path.join(EXT, 'hdlv_xj_validation.parquet'), index=False)
    print('HDLV-XJ 验证:', len(xl), dict(xl.class_new.value_counts()))

def amur():
    am = pd.read_csv(os.path.join(EXT, 'amur_samples-locs-2021.csv'))
    am['class_new'] = -1   # 类码表未知，暂不入训练
    am['src'] = 'amur_basin_2021'
    am['year'] = 2021
    am.to_parquet(os.path.join(EXT, 'amur_raw.parquet'), index=False)
    print('Amur 原样存放:', len(am), dict(am.type_2020.value_counts()))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('eglc', 'all'):
        eglc()
    if which in ('hdlv', 'all'):
        hdlv()
    if which in ('amur', 'all'):
        amur()

