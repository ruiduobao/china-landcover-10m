# -*- coding: utf-8 -*-
"""深度诊断 3（根因）：我们的**训练标签**本身在独立产品眼里是什么？
取 NE 区域各主力类的训练点，用 ESA WorldCover 逐点交叉 → 判断错在"标签"还是"模型" """
import sys, os, json, time
sys.path.insert(0, '.'); sys.path.insert(0, '../0.本地流水线')
import numpy as np, pandas as pd
import prod_conf as C
from lc_conf import CLASSES

SUB = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类', 'r7_train_2023.parquet')
if not os.path.isfile(SUB):
    SUB = os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集', 'r7_train_2023.parquet')
df = pd.read_parquet(SUB, columns=['row_id', 'lon', 'lat', 'class_new'])
# NE 范围
m = (df.lon >= 118.5) & (df.lon <= 135.0) & (df.lat >= 38.5) & (df.lat <= 54.0)
ne = df[m]
print('NE 2023 训练点 %d / 全区 %d' % (len(ne), len(df)))
rng = np.random.default_rng(3)
smp = []
for c, g in ne.groupby('class_new'):
    if len(g) < 50: continue
    smp.append(g.iloc[rng.choice(len(g), min(len(g), 3000), replace=False)])
s = pd.concat(smp, ignore_index=True)
print('抽样 %d 点（每类 ≤3000）' % len(s))

C.apply_account_env('zixen8v8')
import ee
for _i in range(5):
    try:
        ee.Initialize(project='braided-horizon-508210-a5'); break
    except Exception as _x:
        print('init 重试 %d: %s' % (_i+1, str(_x)[:70])); time.sleep(20 + 15*_i)
else:
    raise SystemExit('初始化失败')
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                     {'cl': int(r.class_new)}) for r in s.itertuples()])
# GEE 算法请求上限 10 MB（实测）→ 必须分小批，每批 ≤6000 点
rows = list(s.itertuples())
parts = []
for i in range(0, len(rows), 6000):
    ch = rows[i:i+6000]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                         {'cl': int(r.class_new)}) for r in ch])
    for att in range(3):
        try:
            rr = wc.sampleRegions(collection=fc, properties=['cl'], scale=10,
                                  tileScale=4).getInfo()
            parts += rr['features']
            break
        except Exception as x:
            print('  批 %d 重试 %d: %s' % (i//6000, att+1, str(x)[:70]))
            time.sleep(10 + 10*att)
    print('  批 %d/%d' % (i//6000+1, (len(rows)+5999)//6000), flush=True)
res = {'features': parts}
got = pd.DataFrame([f['properties'] for f in res['features']])
got = got.rename(columns={'Map': 'wc'})
print('WorldCover 命中 %d / %d（%.0f%%）' % (len(got), len(s), 100*len(got)/len(s)))
WCNAME = {10:'WC乔木', 20:'WC灌丛', 30:'WC草地', 40:'WC耕地', 50:'WC建成',
          60:'WC裸/稀', 70:'WC冰雪', 80:'WC水体', 90:'WC湿地', 100:'WC苔藓'}
ct = pd.crosstab(got.cl, got.wc)
print('\n=== 我们的训练标签 × WorldCover（行归一化 %，只列 ≥5%）===')
print('  %-14s %7s   %s' % ('我们的类', 'n命中', 'WC 组成'))
for c in ct.index:
    row = ct.loc[c]
    r = (100*row/row.sum()).sort_values(ascending=False)
    top = '  '.join('%s %.0f' % (WCNAME.get(int(k), k), v) for k, v in r.items() if v >= 5)[:100]
    print('  %3d %-11s %7d   %s' % (c, CLASSES[int(c)][1], row.sum(), top))
ct.to_csv(r'F:/lc_work/label_vs_wc_ne.csv', encoding='utf-8-sig')
print('\n输出 → F:/lc_work/label_vs_wc_ne.csv')
