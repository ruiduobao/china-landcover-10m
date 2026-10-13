# -*- coding: utf-8 -*-
"""o35_yearly_v6.py — 乔灌园地 v6 全量样本 2017–2024 逐年 AEF64 面板（块级可续跑，多进程分片）

* 输入：v6 组件并集（橡胶 30k + 茶园 30k + 苹果 30k + 多源 2,211 + 锚点 149，去重）
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 逐年 mosaic（filterDate Y-01-01~Y+1-01-01），
          scale=10，逐 band unmask(-9999) 防丢点
* 门槛：每（年,块）5000 点；块落盘 data/orchard/v6_yearly/y{Y}/aef_p{pp}.parquet（存在即跳过）
* 输出：data/orchard/v6_yearly/y{Y}/aef_p{pp}.parquet（uid,lon,lat,subtype,grade,rec,A00-A63,aef_ok）
* 用法：python o35_yearly_v6.py --part 0 --nparts 4 --acct 1yxth5g [--years 2017,2018]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
D6 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v6_含苹果'
OUTD = os.path.join(ORD, 'v6_yearly')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def _grade(df, default):
    for c in ('grade', 'tier', 'conf'):
        if c in df.columns:
            return df[c].astype(str).to_numpy()
    return np.array([default] * len(df))


def load_v6():
    parts = []
    rub = pd.read_parquet(os.path.join(D6, '乔灌园地v6_橡胶样本.parquet'))
    rub['srcset'] = 'rubber'
    parts.append(pd.DataFrame({'lon': rub.lon, 'lat': rub.lat, 'subtype': rub.subtype,
                               'grade': _grade(rub, 'B'), 'srcset': 'rubber',
                               'rec': rub.rec if 'rec' in rub.columns else (rub.get('rec_prov', True) if 'rec_prov' in rub.columns else True)}))
    tea = pd.read_parquet(os.path.join(D6, '乔灌园地v6_茶园样本.parquet'))
    parts.append(pd.DataFrame({'lon': tea.lon, 'lat': tea.lat, 'subtype': tea.subtype,
                               'grade': _grade(tea, 'B_专题产品'), 'srcset': 'tea', 'rec': True}))
    ap = pd.read_parquet(os.path.join(D6, '乔灌园地v6_苹果样本.parquet'))
    parts.append(pd.DataFrame({'lon': ap.lon, 'lat': ap.lat, 'subtype': ap.subtype,
                               'grade': _grade(ap, 'B_产品+面积校准'), 'srcset': 'apple', 'rec': True}))
    mu = pd.read_csv(os.path.join(D6, '乔灌园地v6_多源果园候选.csv'), encoding='utf-8-sig')
    parts.append(pd.DataFrame({'lon': mu.lon, 'lat': mu.lat, 'subtype': mu.subtype.fillna('果园/待判'),
                               'grade': _grade(mu, 'B'), 'srcset': 'multi', 'rec': True}))
    an = pd.read_csv(os.path.join(D6, '乔灌园地v6_目视确认锚点.csv'), encoding='utf-8-sig')
    parts.append(pd.DataFrame({'lon': an.lon, 'lat': an.lat, 'subtype': an.Q2.fillna('待判'),
                               'grade': 'A_目视', 'srcset': 'anchor', 'rec': True}))
    ci = pd.read_parquet(os.path.join(ORD, 'citrus_2020_cn_points.parquet'))
    parts.append(pd.DataFrame({'lon': ci.lon, 'lat': ci.lat, 'subtype': '柑橘',
                               'grade': 'C_专题产品(存疑)', 'srcset': 'citrus', 'rec': True}))
    ho = pd.read_parquet(os.path.join(ORD, 'hainan_official_points.parquet'))
    parts.append(pd.DataFrame({'lon': ho.lon, 'lat': ho.lat, 'subtype': '橡胶',
                               'grade': 'A_官方地块', 'srcset': 'hainan_official', 'rec': True}))
    d = pd.concat(parts, ignore_index=True)
    d['lon6'] = d.lon.round(6)
    d['lat6'] = d.lat.round(6)
    d = d.drop_duplicates(subset=['lon6', 'lat6']).reset_index(drop=True)
    d['uid'] = np.arange(len(d))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--acct', default='1yxth5g')
    ap.add_argument('--years', default='2017,2018,2019,2020,2021,2022,2023,2024')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    d = load_v6()
    if a.part == 0 and a.nparts > 1:
        emit('v6 并集 %d 点；来源 %s' % (len(d), d.srcset.value_counts().to_dict()))
    chunks = [d.iloc[i:i + CHUNK] for i in range(0, len(d), CHUNK)]
    mine = [(yi, y, ci, c) for yi, y in enumerate(years) for ci, c in enumerate(chunks) if ci % a.nparts == a.part]
    emit('本进程：年 %d × 块 %d 中负责 %d 个任务' % (len(years), len(chunks), len(mine)))
    import ee
    VC.ensure_ctx(a.acct)
    for yi, y, ci, c in mine:
        yd = os.path.join(OUTD, 'y%d' % y)
        os.makedirs(yd, exist_ok=True)
        fp = os.path.join(yd, 'aef_p%02d.parquet' % ci)
        if os.path.exists(fp):
            continue
        for attempt in range(4):
            try:
                aef = (ee.ImageCollection(AEF).filterDate('%d-01-01' % y, '%d-01-01' % (y + 1))
                       .mosaic().select(VC.FEATS))
                feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'uid': int(r.uid)})
                         for r in c.itertuples()]
                got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                                        tileScale=4).getInfo()['features']
                df = pd.DataFrame([{'uid': g['properties']['uid'],
                                    **{k: g['properties'].get(k) for k in VC.FEATS}} for g in got])
                m = c.merge(df, on='uid', how='left')
                m['aef_ok'] = m[VC.FEATS].notna().all(axis=1)
                m = m.drop(columns=['lon6', 'lat6'])
                m.to_parquet(fp, index=False)
                emit('  y%d 块%02d: %d/%d 命中' % (y, ci, int(m.aef_ok.sum()), len(c)))
                break
            except Exception as e:
                emit('  y%d 块%02d 重试%d: %s' % (y, ci, attempt, str(e)[:120]))
                time.sleep(8)


if __name__ == '__main__':
    main()
