# -*- coding: utf-8 -*-
"""
p5_submit.py — 北京试点 batch 任务生成与提交（8 年逐年分类）
* 路线（试点修订）: 逐年全量分类（PIF 漂移使两点检测不可靠，见 p1）
* 每任务: {year} 年嵌入 mosaic → classify(treeEnsemble) → toAsset (UInt8, 10m, 4326)
* 分配: zsi8emo=2020+样本提取(已完成); s4ezbd=2017/2018/2019/2021; w2qe4hiu=2022/2023/2024 + 2020 验证
* 队列: 每账号 1 RUNNING + 1 PENDING（账号级并发=1）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

TASKS_JSONL = '数据/本地处理/北京试点/tasks_pilot.jsonl'
PLAN = [
    # (year, account, task_name)
    (2020, 'zsi8emo', 'CNLC10_BJ_2020'),
    (2017, 's4ezbd', 'CNLC10_BJ_2017'),
    (2018, 's4ezbd', 'CNLC10_BJ_2018'),
    (2019, 's4ezbd', 'CNLC10_BJ_2019'),
    (2021, 's4ezbd', 'CNLC10_BJ_2021'),
    (2022, 'w2qe4hiu', 'CNLC10_BJ_2022'),
    (2023, 'w2qe4hiu', 'CNLC10_BJ_2023'),
    (2024, 'w2qe4hiu', 'CNLC10_BJ_2024'),
]

def build_and_submit(year, acct, name):
    import ee
    cfg = json.load(open('数据/本地处理/北京试点/models_pilot/trees_bj.json'))
    FEATS_M = cfg['features']
    trees = cfg['trees']
    clf = ee.Classifier.decisionTreeEnsemble(trees)
    BJ = ee.Geometry(PC.beijing_geojson())
    emb = (ee.ImageCollection(PC.EMB_COL)
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .filterBounds(BJ).mosaic().select(FEATS_M))
    cls = (emb.classify(clf).rename('class')
           .clip(BJ)
           .uint8())
    # 仓库轮换：按该账号已用任务数选仓
    used = sum(1 for t in json.load(open(TASKS_JSONL))
               if t['account'] == acct) if os.path.exists(TASKS_JSONL) else 0
    pid = PC.ACCOUNTS[acct]['pids'][used % len(PC.ACCOUNTS[acct]['pids'])]
    asset = f'projects/{pid}/assets/{name}'
    task = ee.batch.Export.image.toAsset(
        image=cls, description=name, assetId=asset,
        region=BJ, scale=10, crs='EPSG:4326',
        maxPixels=1e13)
    task.start()
    return {'id': name, 'year': year, 'account': acct, 'pid': pid,
            'asset': asset, 'status': 'SUBMITTED', 'taskId': task.id,
            'submit_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def main():
    os.makedirs(os.path.dirname(TASKS_JSONL), exist_ok=True)
    done = []
    if os.path.exists(TASKS_JSONL):
        done = json.load(open(TASKS_JSONL))
    done_names = {t['id'] for t in done}
    for year, acct, name in PLAN:
        if name in done_names:
            print(f'[skip] {name} 已提交')
            continue
        # 用对应账号凭据提交
        PC.load_account(acct)
        info = build_and_submit(year, acct, name)
        done.append(info)
        json.dump(done, open(TASKS_JSONL, 'w'), ensure_ascii=False, indent=2)
        print(f'[submit] {name} -> {info["pid"]}  taskId={info["taskId"][:20]}…')
        time.sleep(5)
    print(f'\n共 {len(done)} 任务，清单: {TASKS_JSONL}')

if __name__ == '__main__':
    main()

