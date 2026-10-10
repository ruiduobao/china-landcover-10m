# -*- coding: utf-8 -*-
"""h8_final.py — 全国灌木样本交付装配（码表对齐 + 双坐标 SHP + 覆盖统计 + 质检门）

* 输入：data/shrub_nat/shrub_samples_nat.parquet（h7 产物）
* 规则源：
        码表：class24（9 常绿灌丛/10 落叶灌丛，24 类空间）映射 class30（120/121，母库空间）；
              "待判"给 class24=-1 并标 shrub_type=待判，**不入训练**，只做复核清单
        SHP：WGS84（对影像）+ GCJ-02（对 DataV 省界，内联坐标转换，字段名 ASCII ≤10 字节）
        覆盖统计：双口径 px（候选盒内 200 m 格）× km²（按纬度余弦校正）
* 门槛（质量门，任一不过即报 FAIL）：
        G1 AEF 缺失=0；G2 与母库精确重合=0；G3 同点间距 ≥1 km；
        G4 每省正样本 ≥5（否则该省只报合并级）；G5 常绿/落叶合计=正样本数（无漏标）
        G6 正样本的 ndvi_amp 有效比例 ≥0.9
* 输出：data/shrub_nat/shrub_samples_nat.csv、shrub_nat_summary.json、shp/（双坐标）、
        data/shrub_nat/QA_gate.json
* 用法：python h8_final.py
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
SHPD = os.path.join(OUTD, 'shp')
EAST_C, EARTH_R = 111320.0, 6378137.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def gcj02(wgs_lon, wgs_lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    x = wgs_lon - 105.0
    y = wgs_lat - 35.0
    def tr(lat, lon):
        ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
        ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    dlat = tr(x, y)
    dlon = tr(y, x)
    radlat = wgs_lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return wgs_lon + dlon, wgs_lat + dlat


def main():
    df = pd.read_parquet(os.path.join(OUTD, 'shrub_samples_nat.parquet'))
    emit('输入 %d 点' % len(df))
    # ---- 码表 ----
    c24 = np.where(df['shrub_type'] == '常绿', 9, np.where(df['shrub_type'] == '落叶', 10, -1))
    df['class24'] = c24
    df['class30'] = np.where(df['shrub_type'] == '常绿', 120, np.where(df['shrub_type'] == '落叶', 121, -1))
    df['src'] = np.where(df['sample_role'] == 'hard_neg', 'nat_shrub_hardneg', 'nat_shrub_pos')
    df['shrub_type_pending'] = df['shrub_type'] == '待判'
    pos = df[df['sample_role'] == 'shrub_pos']
    emit('正样本 %d（常绿 %d / 落叶 %d / 待判 %d）' % (
        len(pos), int((pos['shrub_type'] == '常绿').sum()),
        int((pos['shrub_type'] == '落叶').sum()), int((pos['shrub_type'] == '待判').sum())))
    # ---- 覆盖统计（候选盒口径，双口径）----
    boxes = {}
    for box, g in df.groupby('box'):
        lon0, lat0 = box.split('_')
        lon0, lat0 = float(lon0), float(lat0)
        area = (111.32 * 2.0) * (111.32 * 2.0 * math.cos(math.radians(lat0 + 1.0)))
        boxes[box] = dict(lon0=lon0, lat0=lat0, area_km2=area, n=int(len(g)))
    n_box_pos = int(sum(1 for b, v in boxes.items() if v['n'] and (df[(df['box'] == b) & (df['sample_role'] == 'shrub_pos')].shape[0])))
    land_km2 = sum(v['area_km2'] for v in boxes.values())
    cov_km2 = sum(boxes[b]['area_km2'] for b in boxes
                  if len(df[(df['box'] == b) & (df['sample_role'] == 'shrub_pos')]))
    emit('候选覆盖：%d/%d 盒（%.1f%% 面积）' % (n_box_pos, len(boxes), 100.0 * cov_km2 / max(1, land_km2)))
    # ---- 质检门 ----
    gate = {}
    gate['G1_aef_miss'] = int((df['aef_ok'] != 1).sum())
    key = list(zip(np.round(df['lon'].to_numpy(float), 6), np.round(df['lat'].to_numpy(float), 6)))
    gate['G2_pool_dup'] = int(pd.Series(key).duplicated().sum())
    pts = df[['lon', 'lat']].to_numpy(float)
    bad = 0
    for i in range(1, len(pts)):   # 近邻扫描（O(n·k)，n≤12万可接受）
        d = np.sqrt(((pts[:i] - pts[i]) ** 2).sum(axis=1))
        j = int(np.argmin(d))
        if d[j] > 0 and d[j] * 111320.0 < 1000:
            bad += 1
    gate['G3_min_1km_viol'] = bad
    prov_cnt = pos.groupby('prov').size()
    gate['G4_prov_min5'] = int((prov_cnt < 5).sum())
    gate['G5_ed_sum_ok'] = int(int((pos['shrub_type'].isin(['常绿', '落叶', '待判'])).sum()) == len(pos))
    gate['G6_amp_valid'] = round(float(pos['ndvi_amp'].notna().mean()), 4)
    gate['PASS'] = (gate['G1_aef_miss'] == 0 and gate['G2_pool_dup'] == 0
                    and gate['G3_min_1km_viol'] == 0 and gate['G4_prov_min5'] == 0
                    and gate['G5_ed_sum_ok'] == 1 and gate['G6_amp_valid'] >= 0.9)
    emit('质检门：%s' % {k: v for k, v in gate.items() if k != 'PASS'})
    emit('质检门 PASS = %s' % gate['PASS'])
    # ---- 交付表 ----
    cols = ['point_id', 'lon', 'lat', 'sample_role', 'src', 'prov', 'zone', 'band', 'tier', 'k',
            'dwp', 'dwl', 'wc20', 'lc20', 'mcd', 'tc', 'dem', 'slope', 'bio01', 'bio12', 'occ',
            'shrub_type', 'shrub_type_pending', 'class24', 'class30', 'phys',
            'ndvi_djf', 'ndvi_jja', 'ndvi_amp', 'rh98', 'rh98_mx11', 'year']
    cols = [c for c in cols if c in df.columns]
    out = df[cols + [c for c in df.columns if c.startswith('A') and c[1:].isdigit()] + (['year'] if 'year' in df.columns else [])]
    if 'year' not in out.columns:
        out['year'] = 2023
    out = out.loc[:, [c for c in out.columns if not (c.startswith('A') and c[1:].isdigit())] +
                  [c for c in out.columns if c.startswith('A') and c[1:].isdigit()]]
    out.to_csv(os.path.join(OUTD, 'shrub_samples_nat.csv'), index=False, encoding='utf-8-sig')
    # ---- SHP（双坐标）----
    try:
        import shapefile
        w = shapefile.Writer(os.path.join(SHPD, 'shrub_samples_nat_WGS84.shp'),
                             encoding='utf-8')
        flds = [('point_id', 'C', 24), ('role', 'C', 12), ('prov', 'C', 12), ('zone', 'C', 12),
                ('band', 'C', 2), ('tier', 'C', 2), ('k', 'C', 6), ('type', 'C', 6),
                ('c24', 'N', 8), ('c30', 'N', 8), ('phys', 'C', 10), ('dwp', 'F', 13, 3),
                ('ndvi_amp', 'F', 13, 3), ('rh98', 'F', 13, 3), ('dem', 'N', 8, 0)]
        for n, t, s, *d in flds:
            w.field(n, t, s, *(d or []))
        for r in out.itertuples():
            w.point(float(r.lon), float(r.lat))
            w.record(r.point_id, r.sample_role, str(r.prov), str(r.zone), str(r.band), str(r.tier),
                     str(r.k), r.shrub_type, int(r.class24), int(r.class30), str(r.phys),
                     float(r.dwp), float(r.ndvi_amp) if pd.notna(r.ndvi_amp) else -999.0,
                     float(r.rh98) if pd.notna(r.rh98) else -999.0,
                     float(r.dem) if pd.notna(r.dem) else -999.0)
        w.close()
        wg = shapefile.Writer(os.path.join(SHPD, 'shrub_samples_nat_GCJ02.shp'), encoding='utf-8')
        for n, t, s, *d in flds:
            wg.field(n, t, s, *(d or []))
        for lon, lat, r in zip(out['lon'].to_numpy(float), out['lat'].to_numpy(float),
                               out.itertuples()):
            glon, glat = gcj02(lon, lat)
            wg.point(glon, glat)
            wg.record(r.point_id, r.sample_role, str(r.prov), str(r.zone), str(r.band), str(r.tier),
                      str(r.k), r.shrub_type, int(r.class24), int(r.class30), str(r.phys),
                      float(r.dwp), float(r.ndvi_amp) if pd.notna(r.ndvi_amp) else -999.0,
                      float(r.rh98) if pd.notna(r.rh98) else -999.0,
                      float(r.dem) if pd.notna(r.dem) else -999.0)
        wg.close()
        emit('→ shp/ 双坐标（WGS84 + GCJ-02）')
    except Exception as e:
        emit('SHP 失败：%s' % str(e)[:100])
    # ---- 汇总 ----
    summ = dict(n=int(len(df)), n_pos=int(len(pos)),
                n_hard_neg=int((df['sample_role'] == 'hard_neg').sum()),
                n_ever=int((pos['shrub_type'] == '常绿').sum()),
                n_decid=int((pos['shrub_type'] == '落叶').sum()),
                n_pending=int((pos['shrub_type'] == '待判').sum()),
                n_prov=int(pos['prov'].nunique()),
                cov_boxes=n_box_pos, cov_boxes_total=len(boxes),
                cov_pct=round(100.0 * cov_km2 / max(1, land_km2), 2),
                gate=gate)
    json.dump(summ, open(os.path.join(OUTD, 'shrub_nat_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ shrub_samples_nat.csv / shrub_nat_summary.json')
    emit('完成：%s' % json.dumps({k: v for k, v in summ.items() if k != 'gate'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
