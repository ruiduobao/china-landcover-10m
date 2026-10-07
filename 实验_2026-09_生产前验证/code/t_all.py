# -*- coding: utf-8 -*-
"""t_all.py — §四.3 代表瓦片栅格试产（生产语义：本瓦片 ±2° 样本局部训练 → 10m 栅格导出）
子命令：
  pools  : 从 2023 年度子集切出各瓦片 ±2° 训练池（≤2 万点）并上传（3000 点/片）
  submit : 逐瓦片训练局部 RF + 导出分类栅格到资产（记录真实 EECU / 墙钟）
  qa     : 轮询导出 → 逐瓦片类直方图（scale=30）+ 0 值占比 + 秦岭对瓦片接缝比较
瓦片：秦岭 107–109/33–35、秦岭东 109–111/33–35（接缝对）、松嫩 123–125/45–47、三江 133–135/47–49
"""
import os, sys, time, math, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TILES = {
    'qinling':    dict(box=[107.0, 33.0, 109.0, 35.0], note='森林（秦岭）'),
    'qinling_e':  dict(box=[109.0, 33.0, 111.0, 35.0], note='森林（秦岭东，接缝对）'),
    'songnen':    dict(box=[123.0, 45.0, 125.0, 47.0], note='农林（松嫩）'),
    'sanjiang':   dict(box=[133.0, 47.0, 135.0, 49.0], note='湿地（三江）'),
}
HOSTS = ['5631jn8', '744s27z', 'e5h08k', 'hj74bml']
YEAR = 2023
SRC = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
SHARD = 3000
P = VC.cfg('frozen_params.json')
OUTD = os.path.join(VC.DATA, 'tiles')
REG = os.path.join(VC.DATA, 'asset_registry.json')


def host_of(t):
    return HOSTS[list(TILES).index(t) % len(HOSTS)]


def build_pools():
    """本地切池（流式），只保留瓦片 ±2° 盒内点，cap 20000。"""
    os.makedirs(OUTD, exist_ok=True)
    boxes = {t: [b - 2.0 for b in [v['box'][0], v['box'][1]]] +
                [v['box'][2] + 2.0, v['box'][3] + 2.0] for t, v in TILES.items()}
    cols = ['row_id', 'lon', 'lat', 'class_new'] + VC.FEATS
    acc = {t: [] for t in TILES}
    pf = pq.ParquetFile(SRC)
    for b in pf.iter_batches(batch_size=200_000, columns=cols):
        d = b.to_pandas()
        for t, bx in boxes.items():
            m = ((d.lon >= bx[0]) & (d.lon <= bx[2]) & (d.lat >= bx[1]) & (d.lat <= bx[3])).to_numpy()
            if m.any():
                acc[t].append(d[m])
    for t in TILES:
        if not acc[t]:
            VC.emit('%s 无点' % t); continue
        d = pd.concat(acc[t], ignore_index=True)
        d['row_id'] = d['row_id'].astype('int64')
        h = (d.row_id.to_numpy() * 2654435761) % (2 ** 31)
        d = d.iloc[np.argsort(h, kind='stable')[:20000]]
        d.to_parquet(os.path.join(OUTD, 'pool_%s.parquet' % t), index=False)
        VC.emit('%s 池 %d 点（±2° 盒）' % (t, len(d)))


def upload_pools():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    reg = VC.jload(REG, {})
    only = sys.argv[sys.argv.index('--host') + 1] if '--host' in sys.argv else None
    for t in TILES:
        if only and host_of(t) != only:
            continue
        rk = 'tile_pool_%s' % t
        if reg.get(rk):
            VC.emit('%s 已注册，跳过' % t); continue
        fp = os.path.join(OUTD, 'pool_%s.parquet' % t)
        if not os.path.exists(fp):
            VC.emit('%s 缺池文件' % t); continue
        acct = host_of(t)
        base = 'projects/%s/assets' % VC.pid_of(acct)
        name = 'v31t_%s' % t
        have = VC.list_assets(acct, base)
        if have is None:
            VC.emit('%s 列表失败' % acct); continue
        if (name + '_merged') in have:
            reg[rk] = '%s/%s_merged' % (base, name); VC.jsave(reg, REG)
            VC.emit('%s 已存在，登记' % name); continue
        d = pd.read_parquet(fp)
        nch = math.ceil(len(d) / SHARD)
        missing = [i for i in range(nch) if ('%s_%03d' % (name, i)) not in have]
        if missing:
            ee, _ = VC.ctx(acct)
            for i in missing:
                ch = d.iloc[i * SHARD:(i + 1) * SHARD]
                recs = [ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                                   {'cl': int(r['class_new']), 'lon': float(r['lon']), 'lat': float(r['lat']),
                                    **{f: float(r[f]) for f in VC.FEATS}}) for r in ch.to_dict('records')]
                t2 = ee.batch.Export.table.toAsset(
                    collection=ee.FeatureCollection(recs), assetId='%s/%s_%03d' % (base, name, i),
                    description='v31t_%s_%03d_%s' % (name, i, time.strftime('%m%d%H%M%S')))
                t2.start()
                VC.emit('  %s 片%03d 提交 %s' % (name, i, t2.id))
                time.sleep(2)
            return
        ee, _ = VC.ctx(acct)
        fc = ee.FeatureCollection('%s/%s_000' % (base, name))
        for i in range(1, nch):
            fc = fc.merge(ee.FeatureCollection('%s/%s_%03d' % (base, name, i)))
        t2 = ee.batch.Export.table.toAsset(collection=fc, assetId='%s/%s_merged' % (base, name),
                                           description='v31tm_%s_%s' % (name, time.strftime('%m%d%H%M%S')))
        t2.start()
        reg[rk] = '%s/%s_merged' % (base, name); VC.jsave(reg, REG)
        VC.emit('  %s 合并提交 %s' % (name, t2.id))
        return


def submit():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    reg = VC.jload(REG, {})
    for t, meta in TILES.items():
        rk_out = 'tile_raster_%s' % t
        if reg.get(rk_out):
            VC.emit('%s 栅格已提交过' % t); continue
        src = reg.get('tile_pool_%s' % t)
        if not src:
            VC.emit('%s 池未就绪' % t); continue
        acct = host_of(t)
        # 资产真实存在才提交（注册表是"合并已提交"时置位的，合并可能仍在跑）
        base, nm = src.rsplit('/', 1)
        have = VC.list_assets(acct, base)
        if have is None or nm not in have:
            VC.emit('%s 池资产尚未生成（合并中），本轮跳过' % t); continue
        n_q, err = VC.qdepth(acct)
        if n_q is None or n_q > 3:
            VC.emit('%s 队列=%s，稍后' % (acct, n_q)); continue
        ee, pid = VC.ctx(acct)
        box = ee.Geometry.Rectangle(meta['box'])
        tr = VC.remap_fc_v31(ee.FeatureCollection(src))
        clf = ee.Classifier.smileRandomForest(
            numberOfTrees=P['n_trees'], minLeafPopulation=P['min_leaf'],
            maxNodes=P['max_nodes'], seed=7).train(tr, 'cl', VC.FEATS)
        aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
               .filterBounds(box).mosaic().select(VC.FEATS))
        cls = aef.classify(clf).rename('class').uint8().clip(box)
        aid = 'projects/%s/assets/v31r_%s_%d' % (pid, t, YEAR)
        desc = 'v31r_%s_%d_%s' % (t, YEAR, time.strftime('%m%d%H%M%S'))
        t2 = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
                                           scale=10, crs='EPSG:4326', region=meta['box'],
                                           maxPixels=10 ** 12, shardSize=16)
        t2.start()
        reg[rk_out] = dict(asset=aid, desc=desc, acct=acct, host=acct,
                           submitted=time.strftime('%Y-%m-%d %H:%M:%S'), task=t2.id)
        VC.jsave(reg, REG)
        VC.emit('%s %d 导出提交 @%s → %s' % (t, YEAR, acct, t2.id))
        time.sleep(3)


def task_state(info):
    """查询任务状态；失败返回 '?'"""
    try:
        e, _ = VC.ctx(info['host'])
        ts = e.data.getTaskStatus(info['task'])
        return (ts[0]['state'], round(float(ts[0].get('batch_eecu_usage_seconds') or 0) / 3600, 2))             if ts else ('?', 0.0)
    except Exception:
        return ('?', 0.0)


def winner_of(t):
    """返回该瓦片的有效成品信息：任一路 COMPLETED 即胜（赛跑优先看先完成的）。"""
    reg = VC.jload(REG, {})
    cands = []
    valid = set()
    for m in ('', 'b', 'c', 'd', 'e'):
        valid.add('tile_raster_%s%s' % (t, m))
        valid.add('tile_race_%s%s' % (t, m))
    for k in valid:                      # 精确匹配（避免 qinling 匹配到 qinling_e）
        info = reg.get(k)
        if isinstance(info, dict):
            st, eecu = task_state(info)
            cands.append((k, info, st, eecu))
    rank = {'COMPLETED': 0, 'RUNNING': 1, 'READY': 2, 'PENDING': 3}
    cands.sort(key=lambda x: rank.get(x[2], 4))
    if cands:
        k, info, st, eecu = cands[0]
        return dict(key=k, info=info, state=('COMPLETED' if st == 'COMPLETED' else st), eecu=eecu)
    return dict(key=None, info=None, state='未提交', eecu=0.0)


def qa():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    reg = VC.jload(REG, {})
    L = ['# 代表瓦片试产 QA %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '| 瓦片 | 说明 | 状态 | EECU·h | 覆盖率 | 类数 | Top5 类 |', '|---|---|---|---|---|---|---|']
    for t, meta in TILES.items():
        win = winner_of(t)
        info, st, eecu = win['info'], win['state'], win['eecu']
        if not info:
            L.append('| %s | %s | 未提交 | | | | |' % (t, meta['note'])); continue
        acct = info['host']
        ee, pid = VC.ctx(acct)
        if st != 'COMPLETED':
            L.append('| %s | %s | %s | %.2f | | | |' % (t, meta['note'], st, eecu)); continue
        try:
            img = ee.Image(info['asset'])
            box = ee.Geometry.Rectangle(meta['box'])
            hist = img.reduceRegion(reducer=ee.Reducer.frequencyHistogram(), geometry=box,
                                    scale=30, maxPixels=10 ** 10, tileScale=4).getInfo()
            h = list(hist.values())[0] if hist else {}
            h = {int(float(k)): int(v) for k, v in h.items()}
            tot = sum(h.values())
            zero = h.get(0, 0)
            cov = (tot - zero) / tot if tot else 0
            valid = sorted(((k, v) for k, v in h.items() if k > 0), key=lambda x: -x[1])
            n_cls = len(valid)
            names = VC.V31_NAMES()
            tops = ' '.join('%s:%.0f%%' % (names.get(str(k), k), 100.0 * v / max(1, tot - zero)) for k, v in valid[:5])
            L.append('| %s | %s | ✅ | %.2f | %.2f%% | %d | %s |' % (
                t, meta['note'], eecu, cov * 100, n_cls, tops))
            VC.emit('%s QA：覆盖率 %.2f%% 类数 %d' % (t, cov * 100, n_cls))
        except Exception as e:
            L.append('| %s | %s | ✅(QA失败) | %.2f | | | %s |' % (t, meta['note'], eecu, str(e)[:60]))
    fp = os.path.join(VC.REPT, '瓦片试产QA_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)
    return fp


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'pools'
    {'pools': build_pools, 'upload': upload_pools, 'submit': submit, 'qa': qa}[cmd]()
