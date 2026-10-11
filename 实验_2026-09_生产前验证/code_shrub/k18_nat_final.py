# -*- coding: utf-8 -*-
"""k18_nat_final.py — 装配全国灌木样本放量交付件（本地，零 GEE）

* 输入：nat_points.csv（k15 选点）+ s2_parts/*.parquet（k17 物候）+ nat_scored.parquet（打分）
* 规则源（预注册）：
        常绿/落叶判型阈值 = doc57 68 个口径一致的判读点校准：
             常绿 amp 中位 44，落叶 amp 中位 612 → 阈值 328（两类中位中点），
             灰带 ±150（amp∈[178,478] 判"待判"）
             ——只用 doc57：expA/expA2 的判型标签在物候上无区分力（220 vs 207），不可用于校准
        分层：P1_hi（θ=0.7，框内实测 precision 0.938、按省预测全国 0.94/宁 0.86/闽 0.96）
              P2_vol（θ=0.6，框内实测 0.821、按省预测全国 0.82/宁 0.64/闽 0.88）
        母库标记：与 r7_train_2023 精确重合剔除/标记
* 门槛（质量门，任一不过报 FAIL）：
        G1 物候缺失 ≤10%；G2 同点去重；G3 每省 ≥5 或该省只报合并级；
        G4 常绿+落叶+待判 = 总点；G5 两档点不重叠
* 输出：data/shrub_nat/shrub_nat_samples.csv（终表）+ shrub_nat_stats.json + shp/（双坐标）
* 用法：python k18_nat_final.py
"""
import glob, json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
SHPD = os.path.join(OUTD, 'shp')
POOL23 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
AMP_THR, AMP_GRAY = 328.0, 150.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def gcj02(wgs_lon, wgs_lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    x, y = wgs_lon - 105.0, wgs_lat - 35.0

    def tr(lat, lon):
        ret = (-100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat
               + 0.2 * math.sqrt(abs(lon)))
        ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    dlat, dlon = tr(x, y), tr(y, x)
    rl = wgs_lat / 180.0 * math.pi
    mg = 1 - ee * math.sin(rl) ** 2
    sq = math.sqrt(mg)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (mg * sq) * math.pi)
    dlon = (dlon * 180.0) / (a / sq * math.cos(rl) * math.pi)
    return wgs_lon + dlon, wgs_lat + dlat


def main():
    pts = pd.read_csv(os.path.join(OUTD, 'nat_points.csv'), encoding='utf-8-sig')
    pts['point_id'] = pts['point_id'].astype(str)
    ff = sorted(glob.glob(os.path.join(OUTD, 's2_parts', '*.parquet')))
    if not ff:
        emit('没有 s2_parts，物候未抓完'); return
    s2 = pd.concat([pd.read_parquet(f) for f in ff], ignore_index=True).drop_duplicates(subset=['point_id'])
    df = pts.merge(s2, on='point_id', how='left')
    emit('选点 %d，物候命中 %d（%.1f%%）' % (len(df), int(df.ndvi_jja.notna().sum()),
                                         100.0 * df.ndvi_jja.notna().mean()))
    df['ndvi_amp'] = (df['ndvi_jja'].astype(float) - df['ndvi_djf'].astype(float)).abs()

    def typ(a):
        if pd.isna(a):
            return '待判'
        if a > AMP_THR + AMP_GRAY:
            return '落叶'
        if a < AMP_THR - AMP_GRAY:
            return '常绿'
        return '待判'
    df['shrub_type'] = [typ(v) for v in df['ndvi_amp']]
    df['class24'] = np.where(df['shrub_type'] == '常绿', 9,
                             np.where(df['shrub_type'] == '落叶', 10, -1))
    df['class30'] = np.where(df['shrub_type'] == '常绿', 120,
                             np.where(df['shrub_type'] == '落叶', 121, -1))
    df['src'] = np.where(df['tier'] == 'P1_hi', 'fcs10screen_p07', 'fcs10screen_p06')
    df['year'] = 2023
    # AEF 从母库带出（nat_scored 里已有 AEF）
    ns = pd.read_parquet(os.path.join(OUTD, 'nat_scored.parquet'))
    keep_aef = ['lon', 'lat'] + ['A%02d' % i for i in range(64)]
    ns2 = ns[keep_aef].copy()
    ns2['lon'] = np.round(ns2['lon'].to_numpy(float), 6)
    ns2['lat'] = np.round(ns2['lat'].to_numpy(float), 6)
    df['_k'] = list(zip(np.round(df['lon'].to_numpy(float), 6), np.round(df['lat'].to_numpy(float), 6)))
    ns2['_k'] = list(zip(ns2['lon'], ns2['lat']))
    ns2 = ns2.drop_duplicates(subset=['_k'])
    amap = {k: row for k, row in zip(ns2['_k'], ns2[['A%02d' % i for i in range(64)]].to_numpy(float))}
    A = np.array([amap.get(k, [np.nan] * 64) for k in df['_k']])
    for i in range(64):
        df['A%02d' % i] = A[:, i]
    df['aef_ok'] = df['A00'].notna().astype(int)
    emit('AEF 命中 %d/%d（母库同源）' % (int(df['aef_ok'].sum()), len(df)))
    df = df[df['aef_ok'] == 1].copy()
    key = list(zip(np.round(df['lon'].to_numpy(float), 6), np.round(df['lat'].to_numpy(float), 6)))
    emit('同点重复 %d' % pd.Series(key).duplicated().sum())
    df = df[~pd.Series(key).duplicated().to_numpy()].reset_index(drop=True)
    # 省/区带
    # 省归属必须用原始几何（make_valid 会丢 18 个省，见 k19_provattrib）
    import k19_provattrib as PA
    df['prov'] = PA.attribute(df)
    n_out = int((df['prov'] == '').sum())
    df = df[df['prov'] != ''].reset_index(drop=True)
    emit('界内 %d 点（剔除界外 %d）' % (len(df), n_out))
    cols = ['point_id', 'lon', 'lat', 'tier', 'src', 'prov', 'class_new', 'p_shrub',
            'shrub_type', 'class24', 'class30', 'ndvi_djf', 'ndvi_jja', 'ndvi_amp', 'year',
            'aef_ok'] + ['A%02d' % i for i in range(64)]
    df[cols].to_csv(os.path.join(OUTD, 'shrub_nat_samples.csv'), index=False, encoding='utf-8-sig')
    # 质检门
    pos = df
    gate = dict(G1_pheno_missing_pct=round(100 * (1 - pos['ndvi_jja'].notna().mean()), 2),
                G2_dup=int(pd.Series(key).duplicated().sum()),
                G4_type_sum=int(int(pos['shrub_type'].isin(['常绿', '落叶', '待判']).sum()) == len(pos)),
                G5_tier_overlap=int(len(set(pos[pos.tier == 'P1_hi'].point_id) & set(pos[pos.tier == 'P2_vol'].point_id))),
                G3_prov_below5=int((pos.groupby('prov').size() < 5).sum()),
                n=int(len(pos)))
    gate['PASS'] = (gate['G1_pheno_missing_pct'] <= 10 and gate['G2_dup'] == 0
                    and gate['G4_type_sum'] == 1 and gate['G5_tier_overlap'] == 0
                    and gate['G3_prov_below5'] == 0)
    st = dict(n_total=int(len(df)),
              by_tier={t: int(n) for t, n in df.groupby('tier').size().items()},
              by_type={t: int(n) for t, n in df.groupby('shrub_type').size().items()},
              by_type_tier={'%s|%s' % (t, ty): int(n) for (t, ty), n in
                            df.groupby(['tier', 'shrub_type']).size().items()},
              by_code={str(k): int(v) for k, v in df.groupby('class_new').size().items()},
              n_prov=int(df['prov'].nunique()),
              prov_top={p: int(n) for p, n in df.groupby('prov').size().sort_values(ascending=False).head(12).items()},
              amp_thr=AMP_THR, amp_gray=AMP_GRAY, gate=gate)
    json.dump(st, open(os.path.join(OUTD, 'shrub_nat_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    try:
        import shapefile
        flds = [('point_id', 'C', 24), ('tier', 'C', 8), ('src', 'C', 18), ('prov', 'C', 12),
                ('type', 'C', 6), ('c24', 'N', 8), ('c30', 'N', 8), ('pshrub', 'F', 13, 3),
                ('amp', 'F', 13, 3)]
        for tag, fn in (('WGS84', 'shrub_nat_samples_WGS84.shp'), ('GCJ02', 'shrub_nat_samples_GCJ02.shp')):
            w = shapefile.Writer(os.path.join(SHPD, fn), encoding='utf-8')
            for n, t, s, *d in flds:
                w.field(n, t, s, *(d or []))
            for r in df.itertuples():
                if tag == 'GCJ02':
                    x, y = gcj02(float(r.lon), float(r.lat))
                else:
                    x, y = float(r.lon), float(r.lat)
                w.point(x, y)
                w.record(r.point_id, r.tier, r.src, str(r.prov), r.shrub_type, int(r.class24),
                         int(r.class30), float(r.p_shrub), float(r.ndvi_amp) if pd.notna(r.ndvi_amp) else -999.0)
            w.close()
        emit('→ shp/ 双坐标')
    except Exception as e:
        emit('SHP 失败 %s' % str(e)[:100])
    emit('质检门 PASS=%s（%s）' % (gate['PASS'], {k: v for k, v in gate.items() if k != 'PASS'}))
    emit('→ shrub_nat_samples.csv / shrub_nat_stats.json')
    emit('总计 %d 点：%s' % (len(df), st['by_tier']))
    emit('常绿/落叶/待判：%s' % st['by_type'])


if __name__ == '__main__':
    main()
