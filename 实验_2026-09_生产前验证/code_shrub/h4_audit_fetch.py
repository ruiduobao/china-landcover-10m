# -*- coding: utf-8 -*-
"""h4_audit_fetch.py — 审计点贵层特征 + z18 切片 + 联系表 + 判读索引

* 输入：audit_points.csv（h3）；h2b_drive.py 抓 AEF64+S2+GEDI；m5j 抓 z18
* 规则源：rubric = imagery/SHRUB_RUBRIC_NAT.md；判读表头 point_id,Q1_shrub,Q2_evergreen,
        Q3_other,conf,note
* 门槛：切片并发 10（doc46 实测甜点）；联系表 2×2；索引含 k/zone/prov/dwp/dwl/hint
* 输出：audit_feats*.parquet、imagery/chips/*.png、imagery/sheets/sheet_*.png、
        imagery/audit_index.csv、imagery/sheet_map.json
* 用法：python h4_audit_fetch.py [--skip-feats]
"""
import argparse, json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
CODE = os.path.join(WORK, 'code')
PY = r'F:/anaconda/envs/geeMAP_env/python.exe'
OUTD = os.path.join(WORK, 'data', 'shrub_nat')
IMGD = os.path.join(OUTD, 'imagery')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-feats', action='store_true')
    a = ap.parse_args()
    os.makedirs(IMGD, exist_ok=True)
    audit = os.path.join(OUTD, 'audit_points.csv')
    if not a.skip_feats:
        emit('== 贵层特征（AEF64+S2+GEDI）==')
        r = subprocess.run([PY, os.path.join(CODE, 'h2b_drive.py'), '--in', audit,
                            '--out', os.path.join(OUTD, 'audit_feats.parquet'),
                            '--procs', '4', '--nbr', '1'],
                           cwd=CODE, capture_output=True, text=True, encoding='utf-8')
        emit('  h2b_drive rc=%d %s' % (r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout else ''))
    # ---- 索引：把特征合回审计点 ----
    import glob
    import pandas as pd
    import numpy as np
    ap_df = pd.read_csv(audit, encoding='utf-8-sig')
    ap_df['point_id'] = ap_df['point_id'].astype(str)
    ff = sorted(glob.glob(os.path.join(OUTD, 'audit_feats_p*.parquet')))
    if ff:
        fe = pd.concat([pd.read_parquet(f) for f in ff], ignore_index=True)
        cols = ['point_id', 'aef_ok', 'ndvi_jja', 'ndvi_djf', 'ndvi_amp', 'rh98', 'rh98_mx11']
        cols = [c for c in cols if c in fe.columns]
        ap_df = ap_df.merge(fe[cols], on='point_id', how='left')
    else:
        emit('没有 audit_feats 分片，索引仅含便宜层')
    ap_df['hint'] = ['物候振幅%.2f→%s' % (r.ndvi_amp, '常绿倾向' if r.ndvi_amp < 0.2 else
                                        ('落叶倾向' if r.ndvi_amp > 0.35 else '中间'))
                     if pd.notna(getattr(r, 'ndvi_amp', np.nan)) else '无物候' for r in ap_df.itertuples()]
    ap_df['rh98m'] = [('邻域冠层%.1fm' % r.rh98_mx11) if pd.notna(getattr(r, 'rh98_mx11', np.nan))
                      else '' for r in ap_df.itertuples()]
    ap_df['order'] = range(len(ap_df))
    ap_df.to_csv(os.path.join(IMGD, 'audit_index.csv'), index=False, encoding='utf-8-sig')
    emit('索引 %d 行 → %s' % (len(ap_df), os.path.join(IMGD, 'audit_index.csv')))
    # ---- z18 切片 ----
    emit('== z18 切片 ==')
    cmd = [PY, os.path.join(CODE, 'm5j_m3b_imagery.py'), 'fetch', '--in',
           os.path.join(IMGD, 'audit_index.csv'), '--outdir', IMGD, '--workers', '10']
    r = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True, encoding='utf-8')
    emit('  fetch rc=%d：%s' % (r.returncode, ' ; '.join(r.stdout.strip().splitlines()[-3:]) if r.stdout else r.stderr[-200:]))
    # ---- 联系表 + sheet_map ----
    emit('== 联系表 ==')
    r = subprocess.run([PY, os.path.join(CODE, 'm5j_m3b_imagery.py'), 'sheet', '--outdir', IMGD],
                       cwd=CODE, capture_output=True, text=True, encoding='utf-8')
    emit('  sheet rc=%d：%s' % (r.returncode, ' ; '.join(r.stdout.strip().splitlines()[-2:]) if r.stdout else r.stderr[-200:]))
    import PIL.Image as Image
    chips = sorted(f[:-4] for f in os.listdir(os.path.join(IMGD, 'chips')) if f.endswith('.png'))
    per = 4
    smap = {}
    for k in range(0, len(chips), per):
        smap['sheet_%02d.png' % (k // per)] = chips[k:k + per]
    json.dump(smap, open(os.path.join(IMGD, 'sheet_map.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('联系表 %d 张，覆盖 %d 点 → sheet_map.json' % (len(smap), len(chips)))


if __name__ == '__main__':
    main()
