# -*- coding: utf-8 -*-
"""k14_modis_pheno.py — 用 MODIS 16 天 NDVI 代替 S2 合成做常绿/落叶判据（先验证再上量）

* 为什么：放量要点（预计 10 万级）都要判常绿/落叶，S2 四季中位合成要现算（约 1.1 点/秒/进程），
        10 万点要 2 小时以上；MODIS MOD13Q1 是**预合成的 250 m 16 天 NDVI**，采样成本低一个量级。
        先在已判读的 210 个可分型灌丛点上验证 MODIS 振幅与 S2 振幅的判型一致性。
* 输入：judged_feats.parquet（含 S2 ndvi_djf/jja 与 shrub_type）
* 规则源：MOD13Q1 2023 年，取 1 月（DOY 1-16）与 7 月（DOY 193-208）两个旬的 NDVI 均值（×1e4）；
        amp_modis = NDVI_7月 − NDVI_1月；与 S2 的 (ndvi_jja−ndvi_djf) 对齐后比较
* 门槛：两法对"常绿/落叶"的二分类一致率 ≥0.85，且各自 AUC ≥0.95 才用于放量
* 输出：data/shrub_nat/modi_pheno.json + judged_feats.parquet 增加 modi_amp 列
* 用法：python k14_modis_pheno.py [--acct bx15mw]
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
import v31_common as VC

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='bx15mw')
    a = ap.parse_args()
    global ee
    import ee as _ee
    ee = _ee
    VC.ensure_ctx(a.acct)
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    pts = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': pid})
           for pid, x, y in zip(j['point_id'], j['lon'].to_numpy(float), j['lat'].to_numpy(float))]
    col = (ee.ImageCollection('MODIS/061/MOD13Q1')
           .filterDate('2023-01-01', '2024-01-01').select('NDVI'))
    jan = col.filter(ee.Filter.dayOfYear(1, 16)).mean().rename('ndvi_jan').multiply(0.0001)
    jul = col.filter(ee.Filter.dayOfYear(193, 208)).mean().rename('ndvi_jul').multiply(0.0001)
    img = ee.Image.cat([jan, jul]).toFloat()
    rows = []
    for k in range(0, len(pts), 2000):
        for att in range(5):
            try:
                r = img.sampleRegions(collection=ee.FeatureCollection(pts[k:k + 2000]), scale=250,
                                      geometries=False, tileScale=4).getInfo()['features']
                rows += [f['properties'] for f in r]
                break
            except Exception as e:
                emit('  块 %d 重试%d %s' % (k, att + 1, str(e)[:70]))
                time.sleep(8 * (att + 1))
        emit('  MODIS %d/%d' % (min(k + 2000, len(pts)), len(pts)))
    fm = pd.DataFrame(rows).drop_duplicates(subset=['point_id'])
    j = j.merge(fm, on='point_id', how='left')
    j['modi_amp'] = (j['ndvi_jul'].astype(float) - j['ndvi_jan'].astype(float)).abs()
    # ---- 在可分型灌丛点上两法对比 ----
    sub = j[(j['is_shrub'] == '是') & j['shrub_type'].isin(['常绿', '落叶'])].copy()
    sub = sub[sub['modi_amp'].notna() & sub['ndvi_amp'].notna() & sub['ndvi_jul'].notna()
              & sub['ndvi_jan'].notna()]
    y = (sub['shrub_type'] == '落叶').to_numpy(int)
    from sklearn.metrics import roc_auc_score
    auc_s2 = roc_auc_score(y, sub['ndvi_amp'].to_numpy(float))
    auc_m = roc_auc_score(y, sub['modi_amp'].to_numpy(float))
    # 各自最优阈值（Youden）与一致率
    def best_thr(v, yy):
        best, bt = -1, None
        for t in np.quantile(v, np.linspace(0.05, 0.95, 60)):
            pred = (v >= t).astype(int)
        # 直接用中位分裂近似：取使 (TPR-FPR) 最大的阈值
        for t in np.quantile(v, np.linspace(0.05, 0.95, 60)):
            p = (v >= t).astype(int)
            tp = ((p == 1) & (yy == 1)).sum(); fp = ((p == 1) & (yy == 0)).sum()
            tn = ((p == 0) & (yy == 0)).sum(); fn = ((p == 0) & (yy == 1)).sum()
            j_ = tp / max(1, tp + fn) - fp / max(1, fp + tn)
            if j_ > best:
                best, bt = j_, t
        return bt
    ts2 = best_thr(sub['ndvi_amp'].to_numpy(float), y)
    tm = best_thr(sub['modi_amp'].to_numpy(float), y)
    p2 = (sub['ndvi_amp'].to_numpy(float) >= ts2).astype(int)
    pm = (sub['modi_amp'].to_numpy(float) >= tm).astype(int)
    agree = float((p2 == pm).mean())
    res = dict(n=int(len(sub)), auc_s2=round(float(auc_s2), 4), auc_modis=round(float(auc_m), 4),
               thr_s2=round(float(ts2), 5), thr_modis=round(float(tm), 5),
               agree=round(agree, 4),
               usable=bool(agree >= 0.85 and auc_m >= 0.95))
    emit('可分型灌丛点 n=%d：S2 AUC=%.4f｜MODIS AUC=%.4f｜阈值 %.4f/%.4f｜一致率 %.3f → 可用=%s' % (
        len(sub), auc_s2, auc_m, ts2, tm, agree, res['usable']))
    j.to_parquet(os.path.join(OUTD, 'judged_feats.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'modi_pheno.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ modi_pheno.json（judged_feats 已加 modi_amp）')


if __name__ == '__main__':
    main()
