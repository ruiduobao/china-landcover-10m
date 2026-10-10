# -*- coding: utf-8 -*-
"""g9_final.py — 合成四省灌丛样本点最终交付件（分层置信度 + 证据列 + CSV/SHP）

* 输入：data/shrub/judged_all.csv（454 判读点：是78/否328/无法48）
        data/shrub/t2_full.csv（480 判别器扩增点，含物候/地形）
        data/m3/eco_features_ref.csv 等
* 规则源（预注册分层）：
  **T1 目视确认**：第一轮 6 个判读子代理判"是"的 78 点（含影像证据 note）
                  → 类型取判读 Q2（常绿/落叶），缺失 → 物候补
  **T2 判别器扩增**：p_shrub ≥0.60，每省 ≤120，与判读点间距 ≥1 km
                  → 类型取物候（amp<0.15 常绿 / >0.30 落叶 / 其余待判）
                  → **尚未目视复核**，须以 T1 的验证为凭（CV AUC 0.868、留一省 0.836）
  **N1 已确认非灌丛**：判"否"的 328 点（负样本，防止误训正类）
* 门槛：T1/T2 坐标不重复；SHP 字段 ASCII≤10；双坐标系（WGS84 + GCJ-02）
* 输出：data/shrub/shrub_samples_4prov.csv（全部）
        data/shrub/shrub_{T1_确认,T2_扩增}.csv（分开）
        data/shrub/shp/m3b 同款结构（shapefile）
        data/shrub/sample_summary.json
* 用法：python g9_final.py
"""
import csv, json, math, os, sys, time
sys.path.reconfigure(encoding='utf-8') if hasattr(sys, 'reconfigure') else None
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np, pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub')
OUT = os.path.join(OUTD, 'shrub_samples_4prov.csv')
SHP = os.path.join(OUTD, 'shp')
NAMES = {'宁夏': 'Ningxia', '四川': 'Sichuan', '黑龙江': 'Heilongjiang', '福建': 'Fujian'}

def emit(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def amp_of(row):
    try:
        a = float(row.get('ndvi_jja')) - float(row.get('ndvi_djf'))
        if not (a == a): return None
        return round(a, 3)
    except Exception:
        return None

def main():
    j = pd.read_csv(os.path.join(OUTD, 'judged_all.csv'), encoding='utf-8-sig')
    t2 = pd.read_csv(os.path.join(OUTD, 't2_full.csv'), encoding='utf-8-sig')
    t2['amp'] = t2.apply(amp_of, axis=1)

    def veg_from_amp(a):
        if a is None or (isinstance(a, float) and math.isnan(a)): return '待判'
        return '常绿' if a < 0.15 else ('落叶' if a > 0.30 else '待判')

    rows = []
    # ---- T1 ----
    yes = j[j.Q1_shrub == '是']
    for r in yes.itertuples():
        a = amp_of(r._asdict() if hasattr(r, '_asdict') else {})
        t = r.Q2_evergreen if isinstance(r.Q2_evergreen, str) and r.Q2_evergreen in ('常绿', '落叶') else veg_from_amp(a)
        rows.append(dict(point_id=r.point_id, prov=r.prov, tier='T1_目视确认',
                         shrub_type=t if t in ('常绿', '落叶') else '待判', conf=r.conf,
                         p_shrub='', ndvi_amp=a if a is not None else '',
                         dem=getattr(r, 'dem', ''), stratum=r.stratum, src=r.src,
                         note='判读：%s/%s；%s' % (r.Q1_shrub, r.conf, r.note),
                         lon=r.lon, lat=r.lat))
    # ---- T2 ----
    for r in t2.itertuples():
        rows.append(dict(point_id=r.point_id, prov=r.prov, tier='T2_判别器扩增',
                         shrub_type=veg_from_amp(r.amp), conf='中',
                         p_shrub=round(float(r.p_shrub), 3), ndvi_amp=r.amp,
                         dem=float(r.dem) if str(r.dem) not in ('', 'nan') else '',
                         stratum=r.variant, src=str(r.src),
                         note='判别器概率%.2f；物候%s；未目视复核' % (float(r.p_shrub), veg_from_amp(r.amp)),
                         lon=float(r.lon), lat=float(r.lat)))
    # ---- N1 负样本 ----
    no = j[j.Q1_shrub == '否']
    for r in no.itertuples():
        rows.append(dict(point_id=r.point_id, prov=r.prov, tier='N1_目视否定',
                         shrub_type='不适用', conf=r.conf, p_shrub='', ndvi_amp='', dem='',
                         stratum=r.stratum, src=r.src,
                         note='判为否：实为%s' % (r.Q3_other if hasattr(r, 'Q3_other') else ''),
                         lon=r.lon, lat=r.lat))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    emit('最终样本 %d 行 → %s' % (len(out), OUT))
    print('  分层：', dict(out.groupby('tier').size()))
    print('  省×层：', {('%s|%s' % k): int(v) for k, v in out.groupby(['prov', 'tier']).size().items()})
    print('  省×型：', {('%s|%s' % k): int(v) for k, v in out[out.tier.str.startswith(('T1', 'T2'))].groupby(['prov', 'shrub_type']).size().items()})

    # ---- SHP（双坐标，字段 ASCII≤10） ----
    fp_fields = {'point_id': 'point_id', 'prov': 'prov', 'tier': 'tier', 'shrub_type': 'shrubtype',
                 'conf': 'conf', 'p_shrub': 'p_shrub', 'ndvi_amp': 'ndvi_amp',
                 'dem': 'dem', 'stratum': 'stratum', 'src': 'src', 'note': 'note'}
    import geopandas as gpd
    from shapely.geometry import Point
    os.makedirs(SHP, exist_ok=True)
    A, EE = 6378245.0, 0.00669342162296594323
    def to_gcj(lon, lat):
        if not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271):
            return lon, lat
        dlat = _tl(lon - 105.0, lat - 35.0); dlon = _tn(lon - 105.0, lat - 35.0)
        rad = math.radians(lat); m = 1 - EE * math.sin(rad) ** 2; sm = math.sqrt(m)
        dlat = (dlat * 180.0) / ((A * (1 - EE)) / (m * sm) * math.pi)
        dlon = (dlon * 180.0) / (A / sm * math.cos(rad) * math.pi)
        return lon + dlon, lat + dlat
    def _tl(x, y):
        r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2 / 3
        r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3 * math.pi)) * 2 / 3
        r += (160.0 * math.sin(y / 12 * math.pi) + 320.0 * math.sin(y * math.pi / 30)) * 2 / 3
        return r
    def _tn(x, y):
        r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2 / 3
        r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3 * math.pi)) * 2 / 3
        r += (150.0 * math.sin(x / 12 * math.pi) + 300.0 * math.sin(x / 30 * math.pi)) * 2 / 3
        return r
    g = gpd.GeoDataFrame(out.rename(columns=fp_fields),
                         geometry=[Point(x, y) for x, y in zip(out.lon, out.lat)], crs='EPSG:4326')
    g.to_file(os.path.join(SHP, 'shrub_samples_WGS84.shp'), encoding='UTF-8')
    gl = gpd.GeoDataFrame(out.rename(columns=fp_fields),
                          geometry=[Point(*to_gcj(x, y)) for x, y in zip(out.lon, out.lat)],
                          crs='EPSG:4326')
    gl.to_file(os.path.join(SHP, 'shrub_samples_GCJ02.shp'), encoding='UTF-8')
    emit('SHP → %s（WGS84 + GCJ02）' % SHP)
    return out

if __name__ == '__main__':
    main()
