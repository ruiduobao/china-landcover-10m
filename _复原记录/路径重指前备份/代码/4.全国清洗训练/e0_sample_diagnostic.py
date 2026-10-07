# -*- coding: utf-8 -*-
"""
e0_sample_diagnostic.py — 评审阶段A：r7 样本库基线诊断（只读，不修改任何样本）
* 对应外部评审意见 §五阶段A（第1步：不提取嵌入，先完成样本基线诊断）
* 输入: r7_pool / r7_train（样本重建）+ r1_validation（独立验证池）
* A1 来源构成: 类别×来源单源占比(>50%/70%预警)、tier/year/agree_n、省级单源依赖
* A2 空间覆盖: 1°格覆盖、50km 等面积格(Albers)覆盖+Gini、0.25°格分布、省份×类别
* A3 空间自相关: 同类最近邻距离、121 类集中度、验证-训练空间距离（泄漏检查）
* 输出: 数据/本地处理/全国清洗训练/样本诊断/*.csv + e0_summary.json
* 用法: python e0_sample_diagnostic.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

PROJ = r'Z:/Mywork/论文/中国土地覆盖数据'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '5.样本重建'))
import s0_conf as C
from lc_conf import CLASSES
from s1_geom import china_contains

WORK2 = os.path.join(PROJ, '数据/本地处理/全国清洗训练/样本诊断')
os.makedirs(WORK2, exist_ok=True)
POOL = os.path.join(C.WORK, 'r7_pool.parquet')
TRAIN = os.path.join(C.WORK, 'r7_train.parquet')
VALID = C.R1_VALID

# 中国区域 Albers 等面积投影（50km 诊断网格与米制距离统一用）
TR = Transformer.from_crs('EPSG:4326',
                          '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 '
                          '+datum=WGS84 +units=m', always_xy=True)

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)

def xy(lon, lat):
    return TR.transform(np.asarray(lon, float), np.asarray(lat, float))

def src_share(df, tag):
    """类别×来源：单源占比与预警"""
    ct = pd.crosstab(df.class_new, df.src)
    share = ct.div(ct.sum(1), axis=0)
    rows = []
    for c in ct.index:
        top = share.loc[c].sort_values(ascending=False)
        n = int(ct.loc[c].sum())
        rows.append({'class': int(c), 'name': C.class_name(c), 'n': n,
                     'top_src': top.index[0], 'top_share': round(float(top.iloc[0]), 3),
                     'n_src_ge5pct': int((top >= 0.05).sum()),
                     'flag_70': bool(top.iloc[0] > 0.7), 'flag_50': bool(top.iloc[0] > 0.5)})
    out = pd.DataFrame(rows).sort_values('top_share', ascending=False)
    out.to_csv(os.path.join(WORK2, f'source_class_share_{tag}.csv'),
               index=False, encoding='utf-8-sig')
    return out

def coverage(df, tag):
    """A2: 1°格 / 50km 等面积格 / 0.25°格"""
    x, y = xy(df.lon, df.lat)
    res = {'tag': tag, 'n': len(df)}
    # 1° 格
    g1 = (np.floor(df.lon).astype(int) + 1000) * 1000 + np.floor(df.lat).astype(int)
    res['cells_1deg'] = int(g1.nunique())
    # 0.25° 格
    q = (np.floor(df.lon * 4).astype(int) + 10000) * 10000 + np.floor(df.lat * 4).astype(int)
    qcnt = q.value_counts().to_numpy()
    res['gini_025'] = round(gini(qcnt), 4)
    res['max_cell_025'] = int(qcnt.max())
    res['median_cell_025'] = float(np.median(qcnt))
    # 50km 等面积格
    g50 = pd.Series((np.floor(x / 50000).astype(int) + 10000) * 10000 + np.floor(y / 50000).astype(int))
    g50c = g50.value_counts()
    res['cells_50km'] = int(len(g50c))
    res['gini_50km'] = round(gini(g50c.to_numpy()), 4)
    res['max_cell_50km'] = int(g50c.max())
    res['median_cell_50km'] = float(g50c.median())
    return res, pd.DataFrame({'lon': df.lon, 'lat': df.lat, 'x': x, 'y': y,
                              'class_new': df.class_new, 'g1': g1, 'g50': g50})

def china_1deg_cells():
    """1°格计入中国：中心+4角任一测试点在境内（格中心法会漏掉中心在境外的边缘格）"""
    lon = np.arange(73, 136)
    lat = np.arange(17, 54)
    LO, LA = np.meshgrid(lon, lat)
    tests = []
    for dx, dy in [(0.5, 0.5), (0.01, 0.01), (0.99, 0.01), (0.01, 0.99), (0.99, 0.99)]:
        tests.append(np.column_stack([(LO + dx).ravel(), (LA + dy).ravel()]))
    hits = np.zeros(LO.size, dtype=bool)
    for t in tests:
        hits |= china_contains(t[:, 0], t[:, 1])
    keys = np.unique((LO.ravel()[hits].astype(int) * 1000 + LA.ravel()[hits].astype(int)))
    return set(keys.tolist())

def main():
    t0 = time.time()
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src', 'tier', 'year'])
    train = pd.read_parquet(TRAIN)
    valid = pd.read_parquet(VALID, columns=['lon', 'lat', 'class_new', 'src'])
    print(f'pool {len(pool):,} / train {len(train):,} / valid {len(valid):,}', flush=True)
    flags = {}

    # ---------- A1 来源构成 ----------
    print('A1 来源构成…', flush=True)
    sh_train = src_share(train, 'train')
    src_share(pool, 'pool')
    flags['class_single_src_gt70_train'] = sh_train[sh_train.flag_70][
        ['class', 'name', 'n', 'top_src', 'top_share']].to_dict('records')
    # tier / year / agree_n（索引类型不同，分列输出）
    dist = pd.concat([
        train.tier.value_counts().rename('tier').to_frame()
            .assign(key=lambda d: d.index.astype(str)).set_index('key'),
        train.year.value_counts().rename('year').to_frame()
            .assign(key=lambda d: d.index.astype(str)).set_index('key'),
        train.agree_n.value_counts().rename('agree_n').to_frame()
            .assign(key=lambda d: d.index.astype(str)).set_index('key')], axis=1)
    dist.to_csv(os.path.join(WORK2, 'tier_year_agree_distribution.csv'),
                encoding='utf-8-sig')
    # 省级单源依赖
    pv = []
    for p, d in train.groupby('province'):
        vc = d.src.value_counts(normalize=True)
        pv.append({'province': p, 'n': len(d), 'top_src': vc.index[0],
                   'top_share': round(float(vc.iloc[0]), 3),
                   'n_classes': int(d.class_new.nunique())})
    pv = pd.DataFrame(pv).sort_values('top_share', ascending=False)
    pv.to_csv(os.path.join(WORK2, 'province_source_share.csv'),
              index=False, encoding='utf-8-sig')
    flags['province_single_src_gt70'] = pv[pv.top_share > 0.7].to_dict('records')

    # ---------- A2 空间覆盖 ----------
    print('A2 空间覆盖…', flush=True)
    cov_pool, _ = coverage(pool[['lon', 'lat', 'class_new']], 'pool')
    cov_train, gtrain = coverage(train, 'train')
    china_cells = china_1deg_cells()
    # 类别级 1° / 50km / 省份覆盖（train）
    lvl0 = {c: CLASSES[c][3] for c in CLASSES}
    rows = []
    train_arr = train[['class_new', 'province']].reset_index(drop=True)
    for c, d in gtrain.groupby('class_new'):
        rows.append({'class': int(c), 'name': C.class_name(c), 'n': len(d),
                     'cells_1deg': int(d.g1.nunique()),
                     'cells_50km': int(d.g50.nunique()),
                     'n_province': int(train_arr.loc[train_arr.class_new == c,
                                                     'province'].nunique())})
    cov_class = pd.DataFrame(rows).sort_values('cells_50km')
    cov_class.to_csv(os.path.join(WORK2, 'coverage_per_class.csv'),
                     index=False, encoding='utf-8-sig')
    # 一级类 50km 覆盖（对照评审门槛：主要一级类 ≥90% 有样本格中占优）
    cov_class['lvl0'] = cov_class['class'].map(lvl0)
    lvl_cov = cov_class.groupby('lvl0').agg(n=('n', 'sum'), cells_50km=('cells_50km', 'sum'),
                                            cells_1deg=('cells_1deg', 'sum'))
    lvl_cov.to_csv(os.path.join(WORK2, 'coverage_per_level0.csv'), encoding='utf-8-sig')
    flags['coverage'] = {
        'china_1deg_land_cells': len(china_cells),
        'pool_cells_1deg': cov_pool['cells_1deg'],
        'train_cells_1deg': cov_train['cells_1deg'],
        'train_1deg_cover_ratio': round(cov_train['cells_1deg'] / len(china_cells), 3),
        'train_gini_50km': cov_train['gini_50km'],
        'train_gini_025': cov_train['gini_025'],
        'train_max_cell_025': cov_train['max_cell_025'],
        'level0_coverage': lvl_cov.to_dict('index')}
    # 省份×类别计数
    pc = pd.crosstab(train.province, train.class_new)
    pc.to_csv(os.path.join(WORK2, 'province_class_counts.csv'), encoding='utf-8-sig')
    # 省份样本量极差（评审点：最高/最低差约4个数量级）
    pcount = train.province.value_counts()
    flags['province_imbalance'] = {'top': {'p': pcount.index[0], 'n': int(pcount.iloc[0])},
                                   'bottom': {'p': pcount.index[-1], 'n': int(pcount.iloc[-1])},
                                   'ratio': round(float(pcount.iloc[0] / max(pcount.iloc[-1], 1)), 1)}

    # ---------- A3 空间自相关 ----------
    print('A3 空间自相关…', flush=True)
    XT, YT = xy(train.lon, train.lat)
    rng = np.random.default_rng(42)
    rows = []
    for c, d in train.groupby('class_new'):
        idx = np.where(train.class_new.to_numpy() == c)[0]
        if len(idx) > 30000:
            idx = rng.choice(idx, 30000, replace=False)
        P = np.column_stack([XT[idx], YT[idx]])
        t = cKDTree(P)
        dd, _ = t.query(P, k=2)
        nn = dd[:, 1]
        rows.append({'class': int(c), 'name': C.class_name(c), 'n': int(len(idx)),
                     'nn_median_m': round(float(np.median(nn)), 1),
                     'nn_p10_m': round(float(np.percentile(nn, 10)), 1),
                     'nn_p90_m': round(float(np.percentile(nn, 90)), 1)})
    nn = pd.DataFrame(rows).sort_values('nn_median_m')
    nn.to_csv(os.path.join(WORK2, 'nn_distance_per_class.csv'),
              index=False, encoding='utf-8-sig')
    # 121 集中度（工作版 parquet province 可能未填充，先统计缺失）
    d121 = train[train.class_new == 121]
    prov_na = float(d121.province.isna().mean()) if 'province' in d121 else 1.0
    prov121 = d121.province.fillna('未填充').value_counts(normalize=True).head(5)
    flags['class121'] = {'n': len(d121), 'province_na_ratio': round(prov_na, 3),
                         'top5_province_share': {k: round(float(v), 3) for k, v in prov121.items()}}
    # 验证-训练距离（空间泄漏）
    XV, YV = xy(valid.lon, valid.lat)
    ttree = cKDTree(np.column_stack([XT, YT]))
    dv, _ = ttree.query(np.column_stack([XV, YV]), k=1)
    flags['valid_train_dist'] = {
        'median_m': round(float(np.median(dv)), 1),
        'lt_1km': round(float((dv < 1000).mean()), 3),
        'lt_5km': round(float((dv < 5000).mean()), 3),
        'lt_10km': round(float((dv < 10000).mean()), 3)}
    pd.DataFrame({'dist_m': dv}).to_csv(os.path.join(WORK2, 'valid_train_distance.csv'),
                                        index=False)

    # ---------- 汇总 ----------
    summary = {'time': time.strftime('%Y-%m-%d %H:%M'),
               'pool': len(pool), 'train': len(train), 'valid': len(valid),
               'flags': flags, 'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(WORK2, 'e0_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print('\n===== 关键结论 =====')
    print('单源>70%类别:', [f"{r['class']}{r['name']}:{r['top_src']}{r['top_share']:.0%}"
                           for r in flags['class_single_src_gt70_train']][:12])
    print('覆盖:', flags['coverage']['train_1deg_cover_ratio'], '1°格 | Gini50km',
          flags['coverage']['train_gini_50km'], '| 0.25°格最大',
          flags['coverage']['train_max_cell_025'])
    print('省份极差:', flags['province_imbalance'])
    print('121集中度:', flags['class121'])
    print('验证-训练距离:', flags['valid_train_dist'])
    print('输出目录:', WORK2, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
