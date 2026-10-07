# -*- coding: utf-8 -*-
"""y2_upload.py — 年度匹配检查 · 表上传（跨账号分摊、幂等、可断点续跑）
资产命名：v31y_tr_<year>_<w>_<iii> → _merged ；v31y_ev_<year>_<w>_<iii> → _merged
宿主：按 combo 序号轮流分到 healthy 账号（上传并行化 + 配额分摊）
每轮最多提交 PER_RUN 个任务；合并任务在分片齐后提交；最后由 s1 式逻辑做 ACL 共享。
"""
import os, sys, time, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
SHARD = 3000          # 3000 点/片（10MB 请求上限实测安全值，R1 同口径）
PER_RUN = 60
OUTD = os.path.join(VC.DATA, 'yearly')


def hosts(pool):
    hs = [pool['uploader']] + [w for w in pool['workers']]
    return [a for a in dict.fromkeys(hs) if a]


def host_of(pool, year, w):
    idx = YEARS.index(year) * len(WS) + WS.index(w)
    return hosts(pool)[idx % len(hosts(pool))]


def asset_ids(pool, year, w, kind):
    base = 'projects/%s/assets' % VC.pid_of(host_of(pool, year, w))
    return base, 'v31y_%s_%d_%s' % (kind, year, w)


def shard_fc(ee, df, feats, with_cl):
    recs = []
    for r in df.to_dict('records'):
        p = {f: float(r[f]) for f in feats}
        p['rid'] = int(r['row_id'])
        p['lon'] = float(r['lon']); p['lat'] = float(r['lat'])
        if with_cl:
            p['cl'] = int(r['class_new'])
        recs.append(ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]), p))
    return ee.FeatureCollection(recs)


def main():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'), {})
    if not pool or not pool.get('healthy'):
        raise SystemExit('NEED_MANUAL: 账号池无 healthy（先跑 s0_sweep + s0b_health）')
    only = sys.argv[sys.argv.index('--host') + 1] if '--host' in sys.argv else None
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'), {})
    n_sub = 0
    for year in YEARS:
        for w in WS:
            if only and host_of(pool, year, w) != only:
                continue
            for kind, with_cl in [('tr', True), ('ev', False)]:
                base, name = asset_ids(pool, year, w, kind)
                rk = 'y_%s_%d_%s' % (kind, year, w)
                if reg.get(rk):
                    continue
                fp = os.path.join(OUTD, '%s_%d_%s.parquet' % ('train' if with_cl else 'eval', year, w))
                if not os.path.exists(fp):
                    continue
                acct = host_of(pool, year, w)
                have = VC.list_assets(acct, base)
                if have is None:
                    VC.emit('  %s 列表失败，跳过' % acct); continue
                if (name + '_merged') in have:
                    VC.reg_set(rk, '%s/%s_merged' % (base, name))
                    VC.emit('  %s 已存在，登记' % name); continue
                df = pd.read_parquet(fp)
                if len(df) == 0:
                    VC.emit('  %s 空表，跳过' % name); continue
                nch = math.ceil(len(df) / SHARD)
                missing = [i for i in range(nch) if ('%s_%03d' % (name, i)) not in have]
                if missing and n_sub < PER_RUN:
                    n_q, err = VC.qdepth(acct)
                    if n_q is None or n_q > 8:
                        VC.emit('  %s 队列=%s，跳过' % (acct, n_q)); continue
                    ee, _ = VC.ctx(acct)
                    for i in missing:
                        if n_sub >= PER_RUN:
                            break
                        ch = df.iloc[i * SHARD:(i + 1) * SHARD]
                        # 训练表要 64 维；评估表同样需要（分类器要用）
                        fc = shard_fc(ee, ch, VC.FEATS, with_cl)
                        aid = '%s/%s_%03d' % (base, name, i)
                        t = ee.batch.Export.table.toAsset(
                            collection=fc, assetId=aid,
                            description='v31y_%s_%s_%03d_%s' % (name, 'a', i, time.strftime('%m%d%H%M%S')))
                        t.start(); n_sub += 1
                        VC.emit('  上传 %s 片%03d (%d 点) → %s' % (name, i, len(ch), t.id))
                        time.sleep(1.5)
                if n_sub >= PER_RUN:
                    VC.emit('本轮提交达上限 %d，余下轮继续' % PER_RUN)
                    return
                have = VC.list_assets(acct, base) or set()
                if all(('%s_%03d' % (name, i)) in have for i in range(nch)):
                    ee, _ = VC.ctx(acct)
                    fc = ee.FeatureCollection('%s/%s_000' % (base, name))
                    for i in range(1, nch):
                        fc = fc.merge(ee.FeatureCollection('%s/%s_%03d' % (base, name, i)))
                    t = ee.batch.Export.table.toAsset(
                        collection=fc, assetId='%s/%s_merged' % (base, name),
                        description='v31y_mrg_%s_%s_%s' % (name, 'a', time.strftime('%m%d%H%M%S')))
                    t.start(); n_sub += 1
                    VC.emit('  合并提交 %s_merged → %s' % (name, t.id))
                    reg[rk] = '%s/%s_merged' % (base, name)
                    VC.jsave(reg, os.path.join(VC.DATA, 'asset_registry.json'))
    VC.emit('y2 本轮提交 %d' % n_sub)


if __name__ == '__main__':
    main()
