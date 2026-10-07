# -*- coding: utf-8 -*-
"""
b1_beijing_subset.py — 试验区样本子集生成（配置驱动，可复用于任何区域/全国）
* 从 r1_train 按 生产配置.json 的 bbox 裁剪子集
* 划分: 按 0.25° 格整格划分 train/holdout（~25%，防空间泄漏，确定性）
* 输出: 配置 train_file / holdout_file 指定的路径 + {out_dir}/b1_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2.北京试点'))
import s0_conf as C
import pilot_common as PC   # 复用生产配置（bbox/out_dir/train_file/holdout_file）

def main():
    t0 = time.time()
    df = pd.read_parquet(C.R1_TRAIN, columns=['lon', 'lat', 'class_new', 'tier', 'year',
                                              'src', 'src_conf', 'agree_n', 'province'])
    lon0, lat0, lon1, lat1 = PC.BBOX
    m = (df.lon >= lon0) & (df.lon <= lon1) & (df.lat >= lat0) & (df.lat <= lat1)
    sub = df[m].reset_index(drop=True)
    print(f'配置 BBOX {PC.BBOX} 内 r1_train 点: {len(sub):,}', flush=True)

    ck = (np.floor(sub.lon * 4).astype(np.int64) * 100000 +
          np.floor(sub.lat * 4).astype(np.int64))
    is_hold = (ck % 4 == 0)
    hold = sub[is_hold].reset_index(drop=True)
    tr = sub[~is_hold].reset_index(drop=True)
    print(f'格划分: train {len(tr):,} / holdout {len(hold):,}（{len(hold)/max(len(sub),1):.1%}）')

    for p in [PC.TRAIN_FILE, PC.HOLDOUT_FILE]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    tr.to_parquet(PC.TRAIN_FILE, index=False)
    hold.to_parquet(PC.HOLDOUT_FILE, index=False)
    vc = tr.class_new.value_counts().sort_index()
    print(f'train 类别覆盖: {tr.class_new.nunique()} 类')
    print(vc.to_string())

    summary = {'bbox': list(PC.BBOX), 'total': int(len(sub)),
               'train': int(len(tr)), 'holdout': int(len(hold)),
               'train_classes': int(tr.class_new.nunique()),
               'by_class_train': {str(k): int(v) for k, v in vc.items()}}
    json.dump(summary, open(os.path.join(PC.OUT_DIR, 'b1_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print(f'\n输出: {PC.TRAIN_FILE} / {PC.HOLDOUT_FILE} ({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
