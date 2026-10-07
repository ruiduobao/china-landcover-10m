# -*- coding: utf-8 -*-
"""
p5b_monitor.py — 轮询 8 个年度任务状态（listOperations），失败自动换仓重提
* 用法: python p5b_monitor.py [loop]   （loop=持续轮询，缺省=查一轮）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

TASKS_JSONL = '数据/本地处理/北京试点/tasks_pilot.jsonl'

def poll_once():
    tasks = json.load(open(TASKS_JSONL))
    by_acct = {}
    for t in tasks:
        by_acct.setdefault(t['account'], []).append(t)
    changed = False
    for acct, ts in by_acct.items():
        PC.load_account(acct)
        import ee
        ops = ee.data.listOperations()
        opmap = {}
        for op in ops:
            md = op.get('metadata', {})
            desc = md.get('description', '')
            opmap[desc] = op
        for t in ts:
            op = opmap.get(t['id'])
            if op:
                st = op.get('metadata', {}).get('state', '?')
                eecu = op.get('metadata', {}).get('batchEecuUsageSeconds', 0)
            else:
                st, eecu = 'NOT_FOUND', 0
            if st != t['status']:
                print(f"  {t['id']}: {t['status']} -> {st}  (EECU {int(eecu)}s)")
                t['status'] = st
                t['eecu_seconds'] = int(eecu)
                changed = True
            else:
                print(f"  {t['id']}: {st} (EECU {int(eecu)}s)")
    if changed:
        json.dump(tasks, open(TASKS_JSONL, 'w'), ensure_ascii=False, indent=2)
    return tasks

def resubmit_failed():
    """FAILED 任务换仓重提（沿用 p5 逻辑，改 pid）"""
    tasks = json.load(open(TASKS_JSONL))
    failed = [t for t in tasks if t['status'] == 'FAILED']
    if not failed:
        return
    print(f'{len(failed)} 个 FAILED，换仓重提…')
    import subprocess
    for t in failed:
        t['status'] = 'RETRY_PENDING'
        t['retry'] = t.get('retry', 0) + 1
    json.dump(tasks, open(TASKS_JSONL, 'w'), ensure_ascii=False, indent=2)
    # 重提交逻辑在 p5 的基础上改 pid：直接调用内部函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    import p5_submit
    importlib.reload(p5_submit)
    for t in failed:
        acct = t['account']
        PC.load_account(acct)
        used = sum(1 for x in tasks if x['account'] == acct)
        pid = PC.ACCOUNTS[acct]['pids'][(used + t['retry']) % len(PC.ACCOUNTS[acct]['pids'])]
        import ee
        cfg = json.load(open('数据/本地处理/北京试点/models_pilot/trees_bj.json'))
        clf = ee.Classifier.decisionTreeEnsemble(cfg['trees'])
        BJ = ee.Geometry(PC.beijing_geojson())
        emb = (ee.ImageCollection(PC.EMB_COL)
               .filterDate(f"{t['year']}-01-01", f"{t['year']+1}-01-01")
               .filterBounds(BJ).mosaic().select(cfg['features']))
        cls = emb.classify(clf).rename('class').clip(BJ).uint8()
        name = f"{t['id']}_r{t['retry']}"
        asset = f'projects/{pid}/assets/{name}'
        task = ee.batch.Export.image.toAsset(
            image=cls, description=name, assetId=asset,
            region=BJ, scale=10, crs='EPSG:4326', maxPixels=1e13)
        task.start()
        t['id'] = name
        t['pid'] = pid
        t['asset'] = asset
        t['status'] = 'SUBMITTED'
        t['taskId'] = task.id
        json.dump(tasks, open(TASKS_JSONL, 'w'), ensure_ascii=False, indent=2)
        print(f'  重提 {name} -> {pid}')

if __name__ == '__main__':
    loop = len(sys.argv) > 1 and sys.argv[1] == 'loop'
    while True:
        tasks = poll_once()
        states = [t['status'] for t in tasks]
        resubmit_failed()
        if not loop:
            break
        if all(s == 'SUCCEEDED' for s in states):
            print('\n全部 SUCCEEDED!')
            break
        print('--- 60s 后再查 ---')
        time.sleep(60)
