# -*- coding: utf-8 -*-
"""e10_pool_ecoaudit.py — 训练样本 × V 档生态硬约束 审计（回答"样本是否需要调整"）

* 输入：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（2023 训练池 2,060,577 点，含 class_new/src）
        A 段（四川精查）：data/eco_gate/_dem/dem_<tile>.tif（60 m）+ bio_sichuan.tif
        B 段（全国粗查）：GEE 现取 USGS/SRTMGL1_003（scale=1000）+ WORLDCLIM/V1/BIO（scale=1000）分纬带拼接
* 规则源：V 档规则表中**不需要坡度**的部分（高程类 + 气候类 + 全域类），阈值与 `e2_sichuan_v_mask.py` 一致
        （含 3 处负责人裁决：D1 无 bio12 判据、D2 无盆地<400 判据、D3 沼泽只看坡度故本审计不含沼泽）
* 门槛：精查段统计池内落在四川 24 瓦内的点；粗查段统计全池（1 km DEM，仅用于"量级"判断，误差 ±200 m 需说明）
* 输出：data/eco_gate/pool_audit/pool_ecoaudit_sichuan.json、pool_ecoaudit_national.json
        pool_violations_<scope>.csv（逐规则×逐类计数）
* 用法：python e10_pool_ecoaudit.py [--scope sichuan|national|both]
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import rasterio
import v31_common as VC

POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
DEMD = os.path.join(WORK, 'data', 'eco_gate', '_dem')
OUTD = os.path.join(WORK, 'data', 'eco_gate', 'pool_audit')
NAMES = VC.V31_NAMES()

# 不需要坡度的规则（与 e2 的 19 条一致，取子集）：(rid, 名称, 判据, 回退类)
RULES_NOSLOPE = [
    (1, 'R18_红树林全域否决', 'cls==18', 15),
    (2, 'R19_海岸盐沼全域否决', 'cls==19', 15),
    (3, 'R20_潮滩全域否决', 'cls==20', 16),
    (4, 'R07_落叶针叶_盆地<1500m', '(cls==7)&(elev<1500)', 6),
    (5, 'R07_落叶针叶_>4800m', '(cls==7)&(elev>4800)', 6),
    (6, 'R04_常绿阔叶_>3800m', '(cls==4)&(elev>3800)', 5),
    (7, 'R05_落叶阔叶_>4500m', '(cls==5)&(elev>4500)', 11),
    (8, 'R06_常绿针叶_>4800m', '(cls==6)&(elev>4800)', 11),
    (9, 'R08_混交_>4800m', '(cls==8)&(elev>4800)', 5),
    (10, 'R02_乔灌园地_>4500m', '(cls==2)&(elev>4500)', 11),
    (11, 'R03_灌溉耕地_>4500m', '(cls==3)&(elev>4500)', 1),
    (13, 'R09_常绿灌丛_>5000m', '(cls==9)&(elev>5000)', 10),
    (14, 'R12_地衣苔藓_盆地', '(cls==12)&(elev<1500)&(bio01>150)', 13),
    (17, 'R17_盐渍湿地_湿润低地', '(cls==17)&(bio12>1200)&(elev<1000)', 11),
    (19, 'R24_冰雪_<2500m', '(cls==24)&(elev<2500)', 22),
]



_RAST_CACHE = {}


def _open_cached(fp):
    """返回 (数组, transform, 宽度, 高度)；同一瓦只读一次。"""
    if fp not in _RAST_CACHE:
        with rasterio.open(fp) as ds:
            _RAST_CACHE[fp] = (ds.read(), ds.transform, ds.width)
    return _RAST_CACHE[fp]


def sample_rasters(fps_bands, lon, lat):
    """向量化采样：fps_bands=[(fp, band)...]，band 为 1 基；返回 (n, len) 数组，越界为 nan。"""
    out = np.full((len(lon), len(fps_bands)), np.nan, 'float32')
    for j, (fp, band) in enumerate(fps_bands):
        if not os.path.exists(fp):
            continue
        arr, tfm, w = _open_cached(fp)
        h = arr.shape[1]
        col = (lon - tfm.c) / tfm.a
        row = (tfm.f - lat) / abs(tfm.e)
        ci = np.floor(col).astype('int64')
        ri = np.floor(row).astype('int64')
        ok = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        if ok.any():
            out[ok, j] = arr[band - 1][ri[ok], ci[ok]]
    return out

def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_pool():
    df = pd.read_parquet(POOL, columns=['lon', 'lat', 'class_new', 'src'])
    df['cls'] = VC.to_v31(df['class_new'].to_numpy(int))
    return df


def audit(df, elev, bio01, bio12, tag, note=''):
    """elev/bio 均为与 df 对齐的一维数组（bio01 单位 0.1℃，bio12 mm）。"""
    env = {'np': np, 'cls': df['cls'].to_numpy(), 'elev': elev, 'bio01': bio01, 'bio12': bio12}
    per_rule = collections.Counter()
    per_rule_cls = collections.defaultdict(collections.Counter)
    hit_any = np.zeros(len(df), bool)
    for rid, name, expr, retreat in RULES_NOSLOPE:
        m = eval(expr, env)
        n = int(m.sum())
        if n:
            per_rule[name] = n
            for c, k in collections.Counter(df['cls'].to_numpy()[m]).items():
                per_rule_cls[name][int(c)] = int(k)
        hit_any |= m
    res = dict(scope=tag, note=note, n_points=int(len(df)),
               n_violation=int(hit_any.sum()),
               pct=round(100 * float(hit_any.mean()), 4),
               per_rule={k: v for k, v in per_rule.most_common()},
               per_rule_cls={k: {str(a): b for a, b in v.most_common()} for k, v in per_rule_cls.items()},
               per_class_violated={(NAMES.get(str(c), str(c))): int(n)
                                   for c, n in collections.Counter(df['cls'].to_numpy()[hit_any]).most_common()},
               per_src_violated=dict(collections.Counter(
                   df['src'].fillna('').astype(str).to_numpy()[hit_any]).most_common(12)))
    return res, hit_any


def scope_sichuan(df):
    """精查：池点落在四川 24 瓦内者，取 60 m DEM + 四川气候。"""
    from e2_sichuan_v_mask import load_tiles
    tiles = load_tiles()
    lon, lat = df['lon'].to_numpy(), df['lat'].to_numpy()
    elev = np.full(len(df), np.nan, 'float32')
    bio01 = np.full(len(df), np.nan, 'float32')
    bio12 = np.full(len(df), np.nan, 'float32')
    for t, box in tiles.items():
        x0, y0, x1, y1 = box
        m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1) & np.isnan(elev)
        if not m.any():
            continue
        fp = os.path.join(DEMD, 'dem_%s.tif' % t)
        if not os.path.exists(fp):
            continue
        v = sample_rasters([(fp, 1)], lon[m], lat[m])[:, 0]
        elev[np.where(m)[0]] = v
    bs = sample_rasters([(os.path.join(DEMD, 'bio_sichuan.tif'), 1),
                         (os.path.join(DEMD, 'bio_sichuan.tif'), 2)], lon, lat)
    bio01, bio12 = bs[:, 0], bs[:, 1]
    sel = ~np.isnan(elev)
    sub = df[sel].reset_index(drop=True)
    emit('四川瓦内池点 %d / %d' % (int(sel.sum()), len(df)))
    return audit(sub, elev[sel], bio01[sel], bio12[sel], 'sichuan',
                 '60 m SRTM + 1 km WorldClim；池点落在四川 24 瓦内者')


def scope_national(df, acct='bx15mw', n_sample=80000, seed=20261009):
    """粗查：GEE 服务端点采样（sampleRegions）——不做栅格下载，避免 48 MB 请求上限。
    抽样 n_sample 点（约 10%）后按比例上推全省；SRTM 按 1 km 聚合（先 reduceResolution 再取样）。"""
    import ee
    VC.ensure_ctx(acct)
    rng = np.random.RandomState(seed)
    sub = df.iloc[rng.choice(len(df), size=min(n_sample, len(df)), replace=False)].reset_index(drop=True)
    emit('全国粗查抽样 %d / %d 点，GEE sampleRegions …' % (len(sub), len(df)))
    pts = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                      {'i': i}) for i, r in enumerate(sub.itertuples())]
    fc = ee.FeatureCollection(pts)
    from functools import reduce
    img = ee.Image('USGS/SRTMGL1_003').rename('elev')
    bio = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'])
    stack = img.addBands(bio)
    out = {}
    step = 4000      # GEE 交互式查询上限 5000 要素 + 载荷 10 MB → 每片 4k 点
    for k in range(0, len(pts), step):
        part = ee.FeatureCollection(pts[k:k + step])
        got = stack.sampleRegions(collection=part, scale=1000, geometries=False,
                                  tileScale=4).getInfo()['features']
        for f in got:
            p = f['properties']
            if 'elev' in p and 'bio01' in p and 'bio12' in p:
                out[int(p['i'])] = (float(p['elev']), float(p['bio01']), float(p['bio12']))
        emit('  采样 %d / %d（累计命中 %d）' % (min(k + step, len(pts)), len(pts), len(out)))
    idx = sorted(out)
    sel = np.zeros(len(sub), bool)
    sel[idx] = True
    elev = np.array([out[i][0] for i in idx], 'float32')
    bio01 = np.array([out[i][1] for i in idx], 'float32')
    bio12 = np.array([out[i][2] for i in idx], 'float32')
    sub2 = sub[sel].reset_index(drop=True)
    res, hit = audit(sub2, elev, bio01, bio12, 'national',
                     'GEE sampleRegions @1 km，抽样 %d/%d（%.1f%%）后按比例上推；山区高程误差 ±200-400 m，仅量级' % (
                         len(sub2), len(df), 100 * len(sub2) / len(df)))
    scale = len(df) / max(1, len(sub2))
    res['sampling'] = dict(n_sampled=int(len(sub2)), n_pool=int(len(df)), scale_factor=round(scale, 3),
                           seed=seed)
    res['n_violation_estimated_pool'] = int(round(res['n_violation'] * scale))
    res['per_rule_estimated_pool'] = {k: int(round(v * scale)) for k, v in res['per_rule'].items()}
    res['note'] += '；已按抽样比 %.2f× 上推全池' % scale
    return res, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='both', choices=['sichuan', 'national', 'both'])
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    df = load_pool()
    emit('池 %d 点（分类分布 top: %s）' % (len(df), dict(collections.Counter(
        NAMES.get(str(c), c) for c in df['cls']).most_common(6))))
    for scope, fn in (('sichuan', scope_sichuan), ('national', scope_national)):
        if a.scope not in (scope, 'both'):
            continue
        res, _ = fn(df)
        VC.jsave(res, os.path.join(OUTD, 'pool_ecoaudit_%s.json' % scope))
        with open(os.path.join(OUTD, 'pool_violations_%s.csv' % scope), 'w',
                  encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rule', 'n_points', 'classes(class:count)'])
            for k, v in res['per_rule'].items():
                w.writerow([k, v, '; '.join('%s:%d' % (NAMES.get(c, c), n)
                                            for c, n in res['per_rule_cls'].get(k, {}).items())])
        emit('=== %s：违反 %d / %d = %.4f%% ===' % (scope, res['n_violation'], res['n_points'], res['pct']))
        for k, v in list(res['per_rule'].items())[:8]:
            emit('   %-26s %6d  %s' % (k, v, '; '.join(
                '%s:%d' % (NAMES.get(c, c), n) for c, n in res['per_rule_cls'].get(k, {}).items())))
        emit('   → %s' % os.path.join(OUTD, 'pool_ecoaudit_%s.json' % scope))


if __name__ == '__main__':
    main()
