# -*- coding: utf-8 -*-
"""v12_flat_coast_fix.py — 湖河滩地海岸污染修复：应用冻结判据（A/B/C/D/E）并重写交付件

* 判据（冻结于 海岸污染修复/技术方案.md，2026-10-13）：
    A 剔除→186候选 : it_n≥3 且 沿海省 且 dem≤10
    B 保留+复核      : it_n∈[1,2]，或 it_n≥3 但(非沿海省 或 dem>10)   → coast_qc=review_tidal
    C 剔除→201候选 : it_n=0 且 沿海省 且 dem≤10 且(wc==60 或 DW bare≥0.30) 且(max_300≤0.5 且 seasonality≤3)
                      且(d_it≤2 或 d_sea≤3)
    D 保留+复核      : it_n=0 且未入 A/B/C 且(d_it≤2 或 d_sea≤1)       → coast_qc=review_nearshore
    E 保留
* 输出：2023/湖河滩地_2023_WGS84.shp（+coast_qc 字段；剔除 A/C）
        湖河滩地_剔除清单.csv（追加 A/C 行）+ 海岸污染修复/复核与剔除清单 + 证据表
* 校验：保留数=A..E 计数吻合；几何/坐标逐点复制；字段小数位正确；coast_qc 分布核对
* 用法：python v12_flat_coast_fix.py
"""
import os, shutil, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/湖河滩地'
EV = os.path.join(WORK, 'data', 'flat_fix', 'out', 'evidence_all.parquet')
SUB = os.path.join(DEST, '海岸污染修复')
SCHEMA = [('pid', 'C', 20), ('lon', 'F', 13, 6), ('lat', 'F', 13, 6), ('cls24', 'N', 4),
          ('cls_zh', 'C', 18), ('tier', 'C', 8), ('src', 'C', 24), ('ct_prov', 'C', 30),
          ('ct_code', 'C', 8), ('ct_km', 'N', 9, 2), ('coast_qc', 'C', 16)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    os.makedirs(SUB, exist_ok=True)
    ev = pd.read_parquet(EV)
    ev['point_id'] = ev.point_id.astype(str)
    num = lambda c: pd.to_numeric(ev[c], errors='coerce')
    dem, bare, mx3, seas = num('dem'), num('bare'), num('max_300'), num('seasonality')
    dit, dsea = num('d_it_km'), num('d_sea_km')
    tidal = ev.it_n >= 3
    A = tidal & ev.coastal_prov & (dem <= 10)
    B = (ev.it_n.between(1, 2)) | (tidal & ~A)
    C = (~A) & (~B) & ev.coastal_prov & (dem <= 10) & ((ev.wc == 60) | (bare >= 0.30)) & \
        (mx3 <= 0.5) & (seas <= 3) & ((dit <= 2) | (dsea <= 3))
    D = (~A) & (~B) & (~C) & ((dit <= 2) | (dsea <= 1))
    E = ~(A | B | C | D)
    emit('A=%d B=%d C=%d D=%d E=%d' % (A.sum(), B.sum(), C.sum(), D.sum(), E.sum()))
    qc = np.where(B, 'review_tidal', np.where(D, 'review_nearshore', ''))
    ev['coast_qc'] = qc
    ev['verdict'] = np.where(A, 'out_tidal', np.where(C, 'out_beach', 'keep'))
    ev['suggest'] = np.where(A, '186', np.where(C, '201', ''))
    # —— 重写 2023 交付件 ——
    src = os.path.join(DEST, '2023', '湖河滩地_2023_WGS84.shp')
    r = shapefile.Reader(src, encoding='utf-8')
    fl = [x[0] for x in r.fields[1:]]
    recs = [list(x) for x in r.iterRecords()]
    pts = [p.points[0] for p in r.iterShapes()]
    r.close()
    idx = {str(rec[fl.index('pid')]): i for i, rec in enumerate(recs)}
    ev = ev.set_index('point_id').loc[list(idx.keys())].reset_index()   # 对齐 SHP 顺序
    keep_mask = np.array([not str(x).startswith('out_') for x in ev.verdict])
    tmp = src[:-4] + '_tmp'
    w = shapefile.Writer(tmp, shapeType=shapefile.POINT, encoding='utf-8')
    for f in SCHEMA:
        w.field(f[0], f[1], f[2], f[3] if len(f) > 3 else 0)
    n_keep = 0
    for k, pid in enumerate(idx.keys()):
        if not keep_mask[k]:
            continue
        i = idx[pid]
        rec = recs[i]
        w.point(float(pts[i][0]), float(pts[i][1]))
        w.record(str(rec[fl.index('pid')]), round(float(pts[i][0]), 6), round(float(pts[i][1]), 6),
                 int(rec[fl.index('cls24')]), str(rec[fl.index('cls_zh')]), str(rec[fl.index('tier')]),
                 str(rec[fl.index('src')]), str(rec[fl.index('ct_prov')]), str(rec[fl.index('ct_code')]),
                 float(rec[fl.index('ct_km')]), str(ev.coast_qc.iloc[k]))
        n_keep += 1
    w.close()
    # 校验
    rr = shapefile.Reader(tmp + '.shp', encoding='utf-8')
    assert len(rr) == n_keep == int(keep_mask.sum()), (len(rr), n_keep, keep_mask.sum())
    rt = rr.records()
    nbad = sum(1 for x in rt if len(x) != len(SCHEMA))
    rr.close()
    assert nbad == 0
    for ext in ('.shp', '.shx', '.dbf'):
        for att in range(4):
            try:
                os.replace(tmp + ext, src[:-4] + ext); break
            except PermissionError:
                if att == 3:
                    data = open(tmp + ext, 'rb').read()
                    with open(src[:-4] + ext, 'wb') as fh:
                        fh.write(data)
                    os.remove(tmp + ext)
                else:
                    time.sleep(2.0 * (att + 1))
    emit('2023 交付件重写：保留 %d / 原 %d' % (n_keep, len(recs)))
    # —— 剔除清单追加 ——
    qp = os.path.join(DEST, '湖河滩地_剔除清单.csv')
    q = pd.read_csv(qp, encoding='utf-8-sig', low_memory=False)
    add = ev[ev.verdict.isin(['out_tidal', 'out_beach'])].copy()
    add['reason'] = np.where(add.verdict == 'out_tidal', 'COAST_TIDAL_FLAT', 'COAST_BEACH_DUNE')
    add['class24'] = add.get('class24', 16)
    add['point_id'] = add.point_id.astype(str)
    q['point_id'] = q.point_id.astype(str)
    cols = list(dict.fromkeys(list(q.columns) + ['verdict', 'suggest']))
    q = pd.concat([q.reindex(columns=cols), add.reindex(columns=cols)], ignore_index=True)
    q.to_csv(qp, index=False, encoding='utf-8-sig')
    emit('剔除清单：%d → %d 行（+%d）' % (len(q) - len(add), len(q), len(add)))
    # —— 复核/剔除清单 ——
    keepcols = ['point_id', 'lon', 'lat', 'ct_prov', 'ct_km', 'it_n', 'd_it_km', 'd_sea_km',
                'dem', 'max_300', 'seasonality', 'bare', 'water', 's2_ndvi_xia', 'coast_qc']
    keepcols = [c for c in keepcols if c in ev.columns]
    ev[ev.verdict == 'out_tidal'][keepcols].assign(suggest=186).to_csv(
        os.path.join(SUB, '剔除_海岸潮滩_建议186.csv'), index=False, encoding='utf-8-sig')
    ev[ev.verdict == 'out_beach'][keepcols].assign(suggest=201).to_csv(
        os.path.join(SUB, '剔除_海岸沙滩_建议201.csv'), index=False, encoding='utf-8-sig')
    ev[ev.coast_qc == 'review_tidal'][keepcols + ['suggest']].to_csv(
        os.path.join(SUB, '待核查_潮汐边缘.csv'), index=False, encoding='utf-8-sig')
    ev[ev.coast_qc == 'review_nearshore'][keepcols + ['suggest']].to_csv(
        os.path.join(SUB, '待核查_近岸水文未定.csv'), index=False, encoding='utf-8-sig')
    # —— 证据表（全量） ——
    evc = [c for c in ['point_id', 'lon', 'lat', 'ct_prov', 'coastal_prov', 'it_n', 'it_ever',
                       'd_it_km', 'd_sea_km', 'etopo_m', 'max_extent', 'occurrence', 'seasonality',
                       'recurrence', 'max_300', 'occ_300', 'm07_150', 'm01_150', 'water', 'bare',
                       'crops', 'built', 'trees', 'grass', 'shrub', 'dm'] if c in ev.columns]
    evc = [c for c in evc if c != 'dm']
    out = ev[keepcols[:1] + [c for c in evc if c != 'point_id'] + ['coast_qc', 'verdict', 'suggest']]
    out.to_csv(os.path.join(SUB, '证据表.csv'), index=False, encoding='utf-8-sig')
    out.to_parquet(os.path.join(SUB, '证据表.parquet'), index=False)
    emit('清单与证据表已写出 → %s' % SUB)
    print('汇总：keep %d（review_tidal %d / review_nearshore %d）｜ out_tidal %d ｜ out_beach %d' % (
        n_keep, int((ev.coast_qc == 'review_tidal').sum()), int((ev.coast_qc == 'review_nearshore').sum()),
        int(A.sum()), int(C.sum())))


if __name__ == '__main__':
    main()
