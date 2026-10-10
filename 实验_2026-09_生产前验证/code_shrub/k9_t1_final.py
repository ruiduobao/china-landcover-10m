# -*- coding: utf-8 -*-
"""k9_t1_final.py — 装配「目视确认灌木样本」T1 交付件（本地，零 GEE）

* 输入：data/shrub_nat/judged_feats.parquet（1,852 判读点 + 全量特征）+ judged_pool.csv（标签）
* 规则源：
        T1 = is_shrub=='是'（目视确认）→ 正样本，shrub_type 取判读的常绿/落叶（"待判"的单独标注）
        N1 = is_shrub=='否'（目视否定）→ 负样本，Q3 实际类别原样带出
        质量门：AEF 缺失剔除；同点去重；与母库精确重合剔除
* 输出：data/shrub_nat/T1_shrub_samples.csv（正样本，含 AEF64/物候/GEDI/地形/分层源）
        data/shrub_nat/N1_hard_negs.csv（负样本）
        data/shrub_nat/T1_shrub.shp / T1_shrub_GCJ02.shp（双坐标）
        data/shrub_nat/T1_stats.json
* 用法：python k9_t1_final.py
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
SHPD = os.path.join(OUTD, 'shp')
POOL23 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'


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
    magic = 1 - ee * math.sin(radlat) ** 2
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return wgs_lon + dlon, wgs_lat + dlat


def main():
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    j['point_id'] = j['point_id'].astype(str)
    n0 = len(j)
    j = j[j['aef_ok'] == 1].copy()
    emit('AEF 剔除 %d → %d' % (n0 - len(j), len(j)))
    key = list(zip(np.round(j['lon'].to_numpy(float), 6), np.round(j['lat'].to_numpy(float), 6)))
    dup = pd.Series(key).duplicated().sum()
    j = j[~pd.Series(key).duplicated().to_numpy()].reset_index(drop=True)
    emit('同点去重 %d → %d' % (dup, len(j)))
    try:
        pool = pd.read_parquet(POOL23, columns=['lon', 'lat'])
        pk = set(zip(np.round(pool['lon'].to_numpy(float), 6), np.round(pool['lat'].to_numpy(float), 6)))
        k2 = list(zip(np.round(j['lon'].to_numpy(float), 6), np.round(j['lat'].to_numpy(float), 6)))
        inpool = np.isin(k2, list(pk))
        emit('与母库精确重合 %d 点（保留但标记 in_pool23）' % int(inpool.sum()))
        j['in_pool23'] = inpool
    except Exception as e:
        emit('母库去重跳过 %s' % str(e)[:60])
        j['in_pool23'] = False

    def zone_of(dem, bio01, bio12):
        if bio01 <= -999:
            return '缺失'
        if dem >= 3000:
            return '青藏高原高山'
        t = bio01 / 10.0
        if t >= 20:
            return '热带'
        if t >= 13 and bio12 >= 900:
            return '亚热带'
        if t <= -2:
            return '寒温带'
        if bio12 < 300:
            return '干旱'
        if bio12 < 500:
            return '半干旱'
        return '温带'
    j['zone'] = [zone_of(r.dem, r.bio01, r.bio12) for r in j.itertuples()]
    j['ndvi_amp'] = (j['ndvi_jja'].astype(float) - j['ndvi_djf'].astype(float)).abs()

    pos = j[j['is_shrub'] == '是'].copy()
    neg = j[j['is_shrub'] == '否'].copy()
    pos['class24'] = np.where(pos['shrub_type'] == '常绿', 9, np.where(pos['shrub_type'] == '落叶', 10, -1))
    pos['class30'] = np.where(pos['shrub_type'] == '常绿', 120, np.where(pos['shrub_type'] == '落叶', 121, -1))
    pos['tier'] = 'T1_visual'
    pos['src'] = 'judged_pool_T1'
    pos['note'] = '' if 'note' not in pos.columns else pos['note']
    neg['tier'] = 'N1_visual'
    neg['src'] = 'judged_pool_N1'
    neg['class24'] = -1
    neg['class30'] = -1
    neg['note'] = '' if 'note' not in neg.columns else neg['note']
    cols = ['point_id', 'lon', 'lat', 'tier', 'src', 'src_exp', 'zone', 'shrub_type', 'conf',
            'class24', 'class30', 'dwp', 'dwl', 'wc20', 'lc20', 'mcd', 'tc', 'dem', 'slope',
            'bio01', 'bio12', 'occ', 'ndvi_djf', 'ndvi_jja', 'ndvi_amp', 'rh98', 'rh98_mx11',
            'q1_raw', 'q2_raw', 'note', 'in_pool23'] + ['A%02d' % i for i in range(64)]
    pos[cols].to_csv(os.path.join(OUTD, 'T1_shrub_samples.csv'), index=False, encoding='utf-8-sig')
    neg[cols].to_csv(os.path.join(OUTD, 'N1_hard_negs.csv'), index=False, encoding='utf-8-sig')
    emit('T1 正样本 %d ｜ N1 负样本 %d' % (len(pos), len(neg)))
    # ---- SHP（双坐标）----
    try:
        import shapefile
        flds = [('point_id', 'C', 26), ('tier', 'C', 12), ('type', 'C', 6), ('conf', 'C', 4),
                ('zone', 'C', 12), ('c24', 'N', 8), ('c30', 'N', 8), ('src', 'C', 20),
                ('dwp', 'F', 13, 3), ('ndvi_amp', 'F', 13, 3), ('rh98', 'F', 13, 3),
                ('dem', 'N', 8, 0), ('inpool', 'L', 1)]
        w = shapefile.Writer(os.path.join(SHPD, 'T1_shrub_WGS84.shp'), encoding='utf-8')
        for n, t, s, *d in flds:
            w.field(n, t, s, *(d or []))
        g = shapefile.Writer(os.path.join(SHPD, 'T1_shrub_GCJ02.shp'), encoding='utf-8')
        for n, t, s, *d in flds:
            g.field(n, t, s, *(d or []))
        for r in pos.itertuples():
            w.point(float(r.lon), float(r.lat))
            w.record(r.point_id, r.tier, str(r.shrub_type), str(r.conf), str(r.zone),
                     int(r.class24), int(r.class30), str(r.src_exp), float(r.dwp),
                     float(r.ndvi_amp) if pd.notna(r.ndvi_amp) else -999.0,
                     float(r.rh98) if pd.notna(r.rh98) else -999.0,
                     float(r.dem) if pd.notna(r.dem) else -999.0, bool(r.in_pool23))
            gl, ga = gcj02(float(r.lon), float(r.lat))
            g.point(gl, ga)
            g.record(r.point_id, r.tier, str(r.shrub_type), str(r.conf), str(r.zone),
                     int(r.class24), int(r.class30), str(r.src_exp), float(r.dwp),
                     float(r.ndvi_amp) if pd.notna(r.ndvi_amp) else -999.0,
                     float(r.rh98) if pd.notna(r.rh98) else -999.0,
                     float(r.dem) if pd.notna(r.dem) else -999.0, bool(r.in_pool23))
        w.close(); g.close()
        emit('→ shp/T1_shrub_WGS84.shp + _GCJ02.shp')
    except Exception as e:
        emit('SHP 失败 %s' % str(e)[:100])
    # ---- 统计 ----
    st = dict(n_T1=int(len(pos)), n_N1=int(len(neg)),
              by_type={t: int(n) for t, n in pos.groupby('shrub_type').size().items()},
              by_type_prov={p: {t: int(n) for t, n in g.groupby('shrub_type').size().items()}
                            for p, g in pos.groupby('zone')},
              by_conf={t: int(n) for t, n in pos.groupby('conf').size().items()},
              by_srcexp={t: int(n) for t, n in pos.groupby('src_exp').size().items()},
              neg_by_type={t: int(n) for t, n in neg.groupby('q2_raw').size().items()},
              in_pool23=int(pos['in_pool23'].sum()))
    json.dump(st, open(os.path.join(OUTD, 'T1_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ T1_shrub_samples.csv / N1_hard_negs.csv / T1_stats.json')
    emit('类型分布 %s' % st['by_type'])


if __name__ == '__main__':
    main()
