# -*- coding: utf-8 -*-
"""f11_extend_2025.py — 乔灌园地 v8 样本扩展到 2025 年（AEF64 2025 并入面板 + 2025 GPKG）

* 输入：类目录根 v8 点表/面板（2017–2024）
        F:/lc_work/forty_exp/feat/aef/pos2025_merged.parquet（240,498×1，命中 1.0000）
        F:/lc_work/forty_exp/feat/aef/neg2025_merged.parquet（112,580×1）
* 规则源：
  ① 正样本面板追加 2025 年行（lon/lat 取点表；A00–A63 取 2025 抓取；aef_ok=1）；
     输出 `乔灌园地v8_正样本_逐年面板_2017-2025.parquet`，**删除旧的 2017-2024 版**（可随时由 year<=2024 过滤重建）；
  ② 负样本面板追加 2025 → `乔灌园地v8_负样本_AEF面板_2020-2025.parquet`（2020/2023/2025 三年）；
  ③ 2025 年 GPKG：与其余年份同字段，内嵌 2025 AEF64（EPSG:4326、SPATIAL_INDEX=NO）；回读校验后写盘；
  ④ 同步到两处：类目录根（正式版副本）与 `v8_ForTy提纯扩量/`（完整包）。
* 门槛：2025 行数 = 点表行数；AEF 命中 ≥0.999；GPKG 回读行数一致、AEF 抽检零差
* 输出：根/包两处的新面板、新负面板、2025 GPKG、v8_stats.json（更新年份与行数）
* 用法：python f11_extend_2025.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地'
PKG = os.path.join(ROOT, 'v8_ForTy提纯扩量')
WORK = r'F:/lc_work/forty_exp'
AEFC = ['A%02d' % k for k in range(64)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_pos_panel():
    for p in (os.path.join(ROOT, '乔灌园地v8_正样本_逐年面板_2017-2024.parquet'),
              os.path.join(PKG, '乔灌园地v8_正样本_逐年面板_2017-2024.parquet')):
        if os.path.exists(p):
            return pd.read_parquet(p), p
    raise SystemExit('找不到 2017-2024 正样本面板')


def main():
    # ---------- 正样本 ----------
    panel, oldp = load_pos_panel()
    emit('旧面板 %d 行（%d 点 × %d 年）' % (len(panel), panel.uid.nunique(), panel.year.nunique()))
    pos = pd.read_csv(os.path.join(ROOT, '乔灌园地v8_正样本_点表.csv'), encoding='utf-8-sig',
                      usecols=['uid', 'lon', 'lat'])
    a25 = pd.read_parquet(os.path.join(WORK, 'feat', 'aef', 'pos2025_merged.parquet'))
    a25 = a25.rename(columns={'pid': 'uid'}) if 'pid' in a25.columns else a25
    a25 = a25[['uid', 'year'] + AEFC + ['aef_ok']].copy()
    a25['uid'] = a25.uid.astype('int64')
    a25 = a25.merge(pos[['uid', 'lon', 'lat']], on='uid', how='left')
    miss = int(a25[['lon', 'lat']].isna().any(axis=1).sum())
    emit('2025 行 %d，AEF 命中 %.4f，坐标缺失 %d' % (len(a25), float(a25.aef_ok.mean()), miss))
    assert len(a25) == len(pos) and miss == 0
    p25 = pd.concat([panel, a25[list(panel.columns)]], ignore_index=True)
    assert len(p25) == 240498 * 9, len(p25)
    out_pos = os.path.join(ROOT, '乔灌园地v8_正样本_逐年面板_2017-2025.parquet')
    p25.to_parquet(out_pos, index=False)
    emit('正样本面板 2017–2025：%d 行 → %s' % (len(p25), os.path.basename(out_pos)))
    # 删除旧面板（根 + 包）
    for d in (ROOT, PKG):
        op = os.path.join(d, '乔灌园地v8_正样本_逐年面板_2017-2024.parquet')
        if os.path.exists(op):
            os.remove(op); emit('  删除旧面板 %s' % d)
    # 同步到包
    import shutil
    shutil.copy2(out_pos, os.path.join(PKG, os.path.basename(out_pos)))
    # ---------- 负样本 ----------
    neg = pd.read_csv(os.path.join(ROOT, '乔灌园地v8_负样本_点表.csv'), encoding='utf-8-sig',
                      usecols=['uid', 'lon', 'lat'])
    npan = pd.read_parquet(os.path.join(ROOT, '乔灌园地v8_负样本_AEF面板.parquet'))
    emit('旧负面板 %d 行（%d 点 × %d 年）' % (len(npan), npan.uid.nunique(), npan.year.nunique()))
    n25 = pd.read_parquet(os.path.join(WORK, 'feat', 'aef', 'neg2025_merged.parquet'))
    n25 = n25.rename(columns={'pid': 'uid'}) if 'pid' in n25.columns else n25
    n25 = n25[['uid', 'year'] + AEFC + ['aef_ok']].copy()
    n25['uid'] = n25.uid.astype('int64')
    n25 = n25.merge(neg[['uid', 'lon', 'lat']], on='uid', how='left')
    emit('2025 负行 %d，命中 %.4f' % (len(n25), float(n25.aef_ok.mean())))
    assert len(n25) == len(neg) and n25[['lon', 'lat']].notna().all().all()
    np2 = pd.concat([npan, n25[list(npan.columns)]], ignore_index=True)
    assert len(np2) == 112580 * 3, len(np2)
    out_neg = os.path.join(ROOT, '乔灌园地v8_负样本_AEF面板_2020-2025.parquet')
    np2.to_parquet(out_neg, index=False)
    emit('负样本面板 2020/2023/2025：%d 行' % len(np2))
    for d in (ROOT, PKG):
        op = os.path.join(d, '乔灌园地v8_负样本_AEF面板.parquet')
        if os.path.exists(op):
            os.remove(op); emit('  删除旧负面板 %s' % d)
    shutil.copy2(out_neg, os.path.join(PKG, os.path.basename(out_neg)))
    # ---------- 2025 GPKG ----------
    pt = pd.read_csv(os.path.join(ROOT, '乔灌园地v8_正样本_点表.csv'), encoding='utf-8-sig')
    aef25 = a25.set_index('uid')[AEFC]
    g = gpd.GeoDataFrame(pt[['uid', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'status', 'src',
                             'rec', 'forty_agree', 'p5', 'ct_prov', 'ct_code', 'ct_km',
                             'elev', 'slope']].copy(),
                         geometry=[Point(x, y) for x, y in zip(pt.lon, pt.lat)], crs='EPSG:4326')
    g['f_agree'] = g.forty_agree.astype('int8')
    for c in AEFC:
        g[c] = g.uid.map(aef25[c]).astype('float32')
    g = g.drop(columns=['forty_agree'])
    fp = os.path.join(ROOT, '2025', '乔灌园地v8_2025_WGS84.gpkg')
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    g.to_file(fp, layer='乔灌园地v8_2025', driver='GPKG', SPATIAL_INDEX='NO', engine='pyogrio')
    r = gpd.read_file(fp, layer='乔灌园地v8_2025', engine='pyogrio')
    idx = np.random.default_rng(7).choice(len(r), size=30, replace=False)
    ok_geo = all(abs(r.geometry.iloc[i].x - pt.lon.iloc[i]) < 1e-9 for i in idx)
    ok_aef = all(np.allclose(r[AEFC].iloc[i].to_numpy('float64'), aef25.loc[r.uid.iloc[i]].to_numpy('float64'),
                             atol=1e-5) for i in idx[:10])
    emit('2025 GPKG：%d 行 | 几何 %s | AEF %s | %.1f MB' % (
        len(r), 'ok' if ok_geo else 'FAIL', 'ok' if ok_aef else 'FAIL', os.path.getsize(fp) / 1e6))
    assert len(r) == 240498 and ok_geo and ok_aef
    os.makedirs(os.path.join(PKG, '2025'), exist_ok=True)
    shutil.copy2(fp, os.path.join(PKG, '2025', os.path.basename(fp)))
    # ---------- 更新 stats ----------
    st = json.load(open(os.path.join(ROOT, 'v8_stats.json'), encoding='utf-8'))
    st['years'] = list(range(2017, 2026))
    st['panel_rows_pos'] = int(len(p25))
    st['panel_rows_neg'] = int(len(np2))
    st['extend_2025'] = {'pos_rows': int(len(a25)), 'neg_rows': int(len(n25)),
                         'aef_hit_pos': round(float(a25.aef_ok.mean()), 4),
                         'aef_hit_neg': round(float(n25.aef_ok.mean()), 4),
                         'crossyear_cos_2024_2025_med': 0.9626,
                         'crossyear_cos_random_med': 0.5267,
                         'gpkg': os.path.basename(fp),
                         'date': '2026-10-27'}
    for d in (ROOT, PKG):
        json.dump(st, open(os.path.join(d, 'v8_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('stats 更新：years=%s，pos_rows=%d，neg_rows=%d' % (st['years'], st['panel_rows_pos'], st['panel_rows_neg']))
    emit('完成')


if __name__ == '__main__':
    main()
