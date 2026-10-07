# -*- coding: utf-8 -*-
"""
v5_e2e_interactive.py — 小区域端到端验证（样本 → GEE 内训练 → 分类图 → 逐类面积抽查）
为什么在 GEE 内训练：GEE 的 decisionTreeEnsemble 需要把整棵森林内联进请求
（全深度 150 棵树 ≈ 数百 MB，远超请求上限），而 smileRandomForest 可在服务端训练，
产出的分类器是服务端对象，没有载荷问题。因此用 **v-final 母库的 stratified 子样本**
（默认 6000 点，弱类保底）在 GEE 内训练，走通"样本→训练→分类图"全链路。
* 交互式：getThumbURL 出图 + reduceRegion(frequencyHistogram) 出逐类面积，**不用 batch 导出**
* 独立对照：同区 ESA WorldCover v200 的逐类面积，用来判断分类图是否可信（不是当真值）
输出: Z:/.../数据/本地处理/全国清洗训练/端到端验证/{region}_map.png, {region}_area.csv,
      {region}_summary.json
用法: python v5_e2e_interactive.py --acct y30b63ye --proj quick-cache-508211-s9
"""
import os, sys, io, json, time, argparse
import numpy as np
import pandas as pd
import requests

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
from lc_conf import CLASSES
import e6_spatial_eval as E6

KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'
OUTD = os.path.join(KB, '数据/本地处理/全国清洗训练/端到端验证')
WORK = r'F:/lc_work'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
CRED = r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
FEATS = [f'A{i:02d}' for i in range(64)]
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
REGIONS = {'R1_闽赣常绿阔叶': (116.0, 24.0, 118.0, 26.0),
           'R2_黑龙江针阔混交': (129.5, 43.5, 131.5, 45.5)}
WEAK = [52, 62, 91, 140, 11]
N_SUB = 5250
WEAK_MIN = 300
OTHER_MAX = 150
YEAR = 2022
CLASS_LIST = sorted(CLASSES)
PALETTE = ['#1f78b4', '#a6cee3', '#ff7f00', '#33a02c', '#b2df8a', '#006400', '#7fc97f',
           '#66c2a5', '#238b45', '#41ab5d', '#00441b', '#78c679', '#00c957', '#c7e9c0',
           '#ffffcc', '#fdd49e', '#d9d9d9', '#969696', '#525252', '#636363', '#bdbdbd',
           '#fdbf6f', '#a63603', '#ffeda0', '#feb24c', '#c6dbef', '#6baed6', '#3182bd',
           '#08519c', '#ffffff', '#d73027', '#4575b4']


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def bring_up(acct, proj):
    d = os.path.join(CRED, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    os.environ.setdefault('HTTP_PROXY', PROXY['http'])
    os.environ.setdefault('HTTPS_PROXY', PROXY['https'])
    import ee
    for i in range(5):
        try:
            ee.Initialize(project=proj)
            break
        except Exception as x:
            if i == 4:
                raise
            emit('init retry %d: %s' % (i + 1, str(x)[:90]))
            time.sleep(20)
    emit('[ee] %s @ %s ready' % (acct, proj))
    return ee


def build_subsample():
    """v-final 母库 2022 训练集的 stratified 子样本（弱类保底），供 GEE 端训练。"""
    fp = os.path.join(WORK, 'vfinal_sub.csv')
    if os.path.isfile(fp):
        return pd.read_csv(fp)
    src = os.path.join(E6.SUB_DIR, f'r7_train_{YEAR}.parquet')
    df = pd.read_parquet(src, columns=['row_id', 'lon', 'lat', 'class_new'] + FEATS)
    rng = np.random.default_rng(2026)
    keep = []
    per = OTHER_MAX
    for c, g in df.groupby('class_new'):
        n = WEAK_MIN if c in WEAK else per
        n = min(n, len(g))
        keep.append(g.iloc[rng.choice(len(g), n, replace=False)])
    sub = pd.concat(keep, ignore_index=True)
    # 特征压到 3 位小数，缩小内联载荷
    for f in FEATS:
        sub[f] = sub[f].round(3)
    sub.to_csv(fp, index=False)
    emit('子样本 %d 点（弱类保底 %d）→ %s' % (len(sub), WEAK_MIN, fp))
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='y30b63ye')
    ap.add_argument('--proj', default='quick-cache-508211-s9')
    ap.add_argument('--scale', type=int, default=30)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    sub = build_subsample()
    emit('子样本 %d 点，载荷约 %.1f MB' % (len(sub), len(sub) * 64 * 7 / 1e6))

    ee = bring_up(a.acct, a.proj)
    recs = []
    for r in sub[['row_id', 'lon', 'lat', 'class_new'] + FEATS].to_dict('records'):
        d = {'cl': int(r['class_new'])}
        for f in FEATS:
            d[f] = float(r[f])
        recs.append(ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]), d))
    fc = ee.FeatureCollection(recs)
    t0 = time.time()
    clf = ee.Classifier.smileRandomForest(numberOfTrees=100, minLeafPopulation=2).train(
        fc, 'cl', FEATS)
    emit('GEE 内训练完成（服务端）%.0fs，树=100' % (time.time() - t0,))

    summary = {'time': time.strftime('%Y-%m-%d %H:%M'), 'year': YEAR, 'regions': {},
               'n_sub': int(len(sub)), 'class_list': CLASS_LIST}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        reg = ee.Geometry.Rectangle([x0, y0, x1, y1])
        emb = (ee.ImageCollection(AEF).filterDate(f'{YEAR}-01-01', f'{YEAR + 1}-01-01')
               .filterBounds(reg).mosaic().select(FEATS))
        cls = emb.classify(clf).rename('class')
        # ---- 逐类面积（服务端直算，不下载栅格）----
        # 交互式调用有计算时限：2°区 30m = 5400 万像元会超时 → 面积用 100m 抽样，
        # 另取 0.5° 核心窗按 30m 细查（两者都在交互式限额内）
        hist = cls.reduceRegion(ee.Reducer.frequencyHistogram(), reg,
                                scale=100, maxPixels=10 ** 9, bestEffort=True).getInfo()
        h = (hist or {}).get('class', {}) or {}
        px_km2 = (100 ** 2) / 1e6
        rows = [{'class': int(c), 'name': CLASSES[int(c)][1],
                 'n_px': int(v), 'area_km2': round(int(v) * px_km2, 1)} for c, v in h.items()]
        area = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
        tot = area.area_km2.sum()
        area['pct'] = (area.area_km2 / tot * 100).round(3)
        area.to_csv(os.path.join(OUTD, f'{name}_area.csv'), index=False, encoding='utf-8-sig')
        nz = int((area.area_km2 > 0).sum())
        emit(f'{name}: 非零类 {nz}/{len(CLASS_LIST)}  总面积 {tot:,.0f} km²  '
             f'最弱 5 类面积 ' + ', '.join(
                 f'{c}={(area[area["class"] == c].area_km2.sum()):,.0f}' for c in WEAK))

        # ---- 独立对照：WorldCover v200 同区面积 ----
        wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        wh = wc.reduceRegion(ee.Reducer.frequencyHistogram(), reg, scale=100,
                             maxPixels=10 ** 9, bestEffort=True).getInfo()
        wch = (wh or {}).get('Map', {}) or {}
        # 0.5° 核心窗 30m 细查（弱类面积的可信口径）
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        core = ee.Geometry.Rectangle([cx - 0.25, cy - 0.25, cx + 0.25, cy + 0.25])
        ch = cls.reduceRegion(ee.Reducer.frequencyHistogram(), core, scale=30,
                              maxPixels=10 ** 9, tileScale=4, bestEffort=True).getInfo()
        chh = (ch or {}).get('class', {}) or {}
        core_rows = [{'class': int(c), 'name': CLASSES[int(c)][1], 'n_px': int(v),
                      'area_km2': round(int(v) * 0.0009, 2)} for c, v in chh.items()]
        emit(f'  0.5°核心窗(30m): 非零类 {len(core_rows)}')

        # ---- 分类图（缩略图，交互式）----
        idx = cls.remap(CLASS_LIST, list(range(len(CLASS_LIST))))
        url = idx.getThumbURL({'region': reg, 'dimensions': 1400, 'format': 'png',
                               'min': 0, 'max': len(CLASS_LIST) - 1, 'palette': PALETTE})
        png = os.path.join(OUTD, f'{name}_map.png')
        try:
            r = requests.get(url, proxies=PROXY, timeout=900)
            r.raise_for_status()
            open(png, 'wb').write(r.content)
            emit(f'  分类图 → {png}（{len(r.content)/1e6:.2f} MB）')
        except Exception as x:
            emit('  出图失败: %s' % str(x)[:120]); png = None
        summary['regions'][name] = {
            'bbox': [x0, y0, x1, y1], 'scale_m': a.scale,
            'n_nonzero_class': nz, 'total_km2': round(float(tot), 1),
            'weak_area_km2': {str(c): float(area[area['class'] == c].area_km2.sum()) for c in WEAK},
            'area_table': rows, 'worldcover_hist': {str(k): int(v) for k, v in wch.items()},
            'area_scale_m': 100, 'core_window_30m': core_rows,
            'png': png}
    json.dump(summary, open(os.path.join(OUTD, 'e2e_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    emit('输出目录: %s' % OUTD)


if __name__ == '__main__':
    main()
