# -*- coding: utf-8 -*-
"""t_race.py — 卡住瓦片的赛跑补投（原任务保留不取消，谁先完成用谁）
用法：python t_race.py <tile> <host_acct>
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import t_all as T

REG = os.path.join(VC.DATA, 'asset_registry.json')


def main():
    t = sys.argv[1]
    host = sys.argv[2]
    year = 2023
    reg = VC.jload(REG, {})
    pool_full = reg.get('tile_pool_%s' % t)
    assert pool_full, '池未注册'
    src_holder = next((a for a in [T.host_of(t)] + VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))['healthy']
                       if VC.pid_of(a) in pool_full), None)
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    emails = ['user:' + pool['emails'][a] for a in pool['workers'] + [pool['uploader']]
              if a in pool.get('emails', {})]
    # ① 池资产共享给赛跑账号
    ee0, _ = VC.ctx(src_holder)
    ee0.data.setAssetAcl(pool_full, {'readers': emails})
    VC.emit('已共享 %s 给 %d 个账号' % (pool_full.split('/')[-1], len(emails)))
    # ② 赛跑提交
    ee, pid = VC.ctx(host)
    box = ee.Geometry.Rectangle(T.TILES[t]['box'])
    tr = VC.remap_fc_v31(ee.FeatureCollection(pool_full))
    clf = ee.Classifier.smileRandomForest(numberOfTrees=T.P['n_trees'], minLeafPopulation=T.P['min_leaf'],
                                          maxNodes=T.P['max_nodes'], seed=7).train(tr, 'cl', VC.FEATS)
    aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1)).filterBounds(box)
           .mosaic().select(VC.FEATS))
    cls = aef.classify(clf).rename('class').uint8().clip(box)
    aid = 'projects/%s/assets/v31r_%s_%d_%s' % (pid, t, year, sys.argv[3] if len(sys.argv) > 3 else 'b')
    desc = 'v31r_%s_%d_%s_%s' % (t, year, sys.argv[3] if len(sys.argv) > 3 else 'b', time.strftime('%m%d%H%M%S'))
    task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
                                         scale=10, crs='EPSG:4326', region=T.TILES[t]['box'],
                                         maxPixels=10 ** 12, shardSize=16)
    task.start()
    reg['tile_race_%s%s' % (t, sys.argv[3] if len(sys.argv) > 3 else '')] = dict(asset=aid, desc=desc, host=host, task=task.id,
                                   submitted=time.strftime('%Y-%m-%d %H:%M:%S'))
    VC.jsave(reg, REG)
    VC.emit('赛跑提交 %s @%s → %s' % (t, host, task.id))


if __name__ == '__main__':
    main()
