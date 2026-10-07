# -*- coding: utf-8 -*-
"""s6_w5_build.py — 湿地窗 w5（三江平原 133-135E/47-49N）构建 + 窗训练样本上传 + 合并 + 共享
切分规则与 w1-w4 完全一致（exp1b_window.py 套路）：
  h=(row_id*2654435761)%5==0 且不在 Baseline(sqrt 200k) 训练集 → 留出评估（本地落 data/eval_w5.parquet）
  其余 → 训练样本（3000点/片 上传 uploader 名下 v31s_w5_<iii>，全齐后 merge 成 v31s_w5_merged）
非阻塞：每轮最多提交 6 片；资产存在性判断幂等。
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd

SHARD = 3000
W = 'w5'
WIN = VC.cfg('windows.json')['windows'][W]
PER_RUN = 6


def build_tables():
    """本地切分（幂等）。返回 (n_train, n_eval, 类分布)。"""
    ev_fp = os.path.join(VC.DATA, 'eval_w5.parquet')
    plan_fp = os.path.join(VC.DATA, 'w5_plan.json')
    if os.path.exists(plan_fp):
        p = VC.jload(plan_fp)
        return p['n_train'], p['n_eval'], p['dist']
    x0, y0, x1, y1 = WIN['bbox']
    d = pd.read_parquet(VC.SRC2023, columns=['row_id'] + VC.FEATS + ['lon', 'lat', 'class_new'])
    d['row_id'] = d['row_id'].astype('int64')
    sub = d[(d.lon >= x0) & (d.lon <= x1) & (d.lat >= y0) & (d.lat <= y1)].copy()
    VC.emit('w5 窗内 %d 点（源=%s）' % (len(sub), VC.SRC2023))
    nat = pd.read_csv(VC.SQRT_CSV, usecols=['row_id'])
    nat_ids = set(nat.row_id.astype('int64'))
    h = (sub.row_id * 2654435761) % 5
    eval_mask = (h == 0) & (~sub.row_id.isin(nat_ids))
    ev, tr = sub[eval_mask].copy(), sub[~eval_mask].copy()
    ev.to_parquet(ev_fp, index=False)
    dist = {str(int(k)): int(v) for k, v in tr.class_new.value_counts().sort_index().items()}
    VC.jsave(dict(bbox=WIN['bbox'], n_train=int(len(tr)), n_eval=int(len(ev)),
                  dist=dist, src=VC.SRC2023, built=time.strftime('%Y-%m-%d %H:%M')),
             plan_fp)
    VC.emit('w5 训练 %d / 留出 %d ｜类分布 %s' % (len(tr), len(ev), dist))
    return len(tr), len(ev), dist


def main():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    n_tr, n_ev, dist = build_tables()
    up = pool['uploader']
    ee, upid = VC.ctx(up)
    up_root = 'projects/%s/assets' % upid
    have = VC.list_assets(up, up_root)
    if have is None:
        raise SystemExit('NEED_MANUAL: listAssets 失败')
    n_ch = (n_tr + SHARD - 1) // SHARD
    missing = [i for i in range(n_ch) if ('v31s_w5_%03d' % i) not in have]
    if ('v31s_w5_merged' not in have) and missing:
        VC.emit('w5 训练 %d 片（待传 %d）' % (n_ch, len(missing)))
        d = pd.read_parquet(VC.SRC2023, columns=['row_id'] + VC.FEATS + ['lon', 'lat', 'class_new'])
        d['row_id'] = d['row_id'].astype('int64')
        x0, y0, x1, y1 = WIN['bbox']
        sub = d[(d.lon >= x0) & (d.lon <= x1) & (d.lat >= y0) & (d.lat <= y1)].copy()
        nat = pd.read_csv(VC.SQRT_CSV, usecols=['row_id'])
        h = (sub.row_id * 2654435761) % 5
        tr = sub[~((h == 0) & (~sub.row_id.isin(set(nat.row_id.astype('int64')))))].copy()
        for i in missing[:PER_RUN]:
            n_q, err = VC.qdepth(up)
            if n_q is None or n_q > 5:
                VC.emit('  队列=%s（%s），下轮续' % (n_q, err)); return
            ch = tr.iloc[i * SHARD:(i + 1) * SHARD]
            recs = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                               {**{f: float(r[f]) for f in VC.FEATS},
                                 'cl': int(r['class_new']), 'lon': float(r['lon']),
                                 'lat': float(r['lat'])}) for r in ch.to_dict('records')]
            aid = '%s/v31s_w5_%03d' % (up_root, i)
            desc = 'v31s_w5_%s_%03d' % (time.strftime('%m%d%H%M%S'), i)
            t = ee.batch.Export.table.toAsset(collection=ee.FeatureCollection(recs),
                                              description=desc, assetId=aid)
            t.start()
            VC.emit('  w5 训练片 %03d 提交 %s' % (i, t.id))
            time.sleep(3)
        return
    if ('v31s_w5_merged' not in have):
        parts = sorted(x for x in have if x.startswith('v31s_w5_'))
        fc = ee.FeatureCollection('%s/%s' % (up_root, parts[0]))
        for p in parts[1:]:
            fc = fc.merge(ee.FeatureCollection('%s/%s' % (up_root, p)))
        desc = 'v31smrg_w5_%s' % time.strftime('%m%d%H%M%S')
        t = ee.batch.Export.table.toAsset(collection=fc, description=desc,
                                          assetId='%s/v31s_w5_merged' % up_root)
        t.start()
        VC.emit('w5 训练合并任务 %s' % t.id)
        return
    VC.reg_set('train_w5', '%s/v31s_w5_merged' % up_root)
    VC.emit('w5 训练合并资产已就绪（评估资产由 s1 接手）')


if __name__ == '__main__':
    main()
