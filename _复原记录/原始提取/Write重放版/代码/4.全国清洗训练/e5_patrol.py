# -*- coding: utf-8 -*-
"""
e5_patrol.py — 全国嵌入提取巡检（每小时定时任务调用）
* 功能:
  1) 进度统计（完成块/709、速率、ETA）
  2) 收集各 worker 日志中的 FINAL-FAIL 块 → 自动重试（tileScale=8，3 次；仍败→hard_fail 档案）
  3) worker 存活检查：python 进程 <4 且未完成 → 自动重启正向 worker
  4) 状态快照 patrol_status.json + 巡检日志 patrol.log
* 用法: python e5_patrol.py
"""
import sys, os, re, glob, json, time, subprocess
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'Z:\Mywork\论文\中国土地覆盖数据\代码\2.北京试点')

BASE = '数据/本地处理/全国清洗训练'
PARTS = os.path.join(BASE, 'emb_parts')
IDX = os.path.join(BASE, 'chunks_index.parquet')
ACCOUNTS = ['zsi8emo', 's4ezbd', 'w2qe4hiu']
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e1_worker.py')

def count_done():
    return len(glob.glob(os.path.join(PARTS, 'chunk_*.parquet')))

def total_chunks():
    import pyarrow.parquet as pq
    return pq.read_table(IDX, columns=['chunk_id']).to_pandas().chunk_id.nunique()

def collect_failed():
    failed = {}
    for f in glob.glob(os.path.join(LOG_DIR, 'w_*.log')) + \
             glob.glob(os.path.join(LOG_DIR, 'wr_*.log')):
        m2 = re.match(r'(wr_)?(w_)?(.+)\.log', os.path.basename(f))
        acct = m2.group(3) if m2 else ''
        with open(f, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                m = re.search(r'chunk(\d+) FINAL-FAIL', line)
                if m:
                    cid = int(m.group(1))
                    fp = os.path.join(PARTS, f'chunk_{cid:04d}.parquet')
                    if not os.path.exists(fp):
                        failed[(acct, cid)] = True
    return list(failed.keys())

LOG_DIR = BASE

def retry_chunk(acct, cid, idx_df):
    sys.path.insert(0, r'Z:\Mywork\论文\中国土地覆盖数据\代码\2.北京试点')
    import pilot_common as PC
    # 关键：先切账号环境再 import/初始化 ee（否则用全局默认凭据 → 权限错）
    acc_dir = os.path.join(r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts', acct)
    os.environ['HOME'] = acc_dir
    os.environ['USERPROFILE'] = acc_dir
    PC.set_proxy()
    import ee
    ee.Initialize(project=PC.ANCHOR_PID[acct])
    FEATS = [f'A{i:02d}' for i in range(64)]
    g = idx_df[idx_df.chunk_id == cid]
    rows = g[['row_id', 'lon', 'lat']].to_dict('records')
    year = int(g.year.iloc[0])
    fp = os.path.join(PARTS, f'chunk_{cid:04d}.parquet')
    for attempt in range(3):
        try:
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
                           {'row_id': int(r['row_id'])})
                for r in rows])
            ic = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                  .filterDate(f'{year}-01-01', f'{year+1}-01-01').filterBounds(fc))
            emb = ic.mosaic().select(FEATS)
            samp = emb.sampleRegions(collection=fc, properties=['row_id'],
                                     scale=10, tileScale=8)
            url = samp.getDownloadURL(filetype='csv', selectors=['row_id'] + FEATS)
            resp = requests.get(url, proxies=PC.PROXY, timeout=2400)
            resp.raise_for_status()
            out = pd.read_csv(io.BytesIO(resp.content))
            out['emb_year'] = year
            out['chunk_id'] = cid
            out.to_parquet(fp, index=False)
            return True
        except Exception as e:
            print(f'  retry {acct} chunk{cid} attempt{attempt+1}: {str(e)[:90]}', flush=True)
            time.sleep(30)
    return False

def workers_alive():
    try:
        r = subprocess.run(['powershell', '-Command',
                            '(Get-Process python -ErrorAction SilentlyContinue | '
                            'Where-Object {$_.StartTime -gt (Get-Date).AddHours(-30)} | '
                            'Measure-Object).Count'],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip())
    except Exception:
        return -1

def relaunch_workers():
    for acct in ACCOUNTS:
        subprocess.Popen([sys.executable, '-u', WORKER, acct],
                         stdout=open(os.path.join(LOG_DIR, f'w_{acct}.log'), 'a'),
                         stderr=subprocess.STDOUT, cwd='Z:/Mywork/论文/中国土地覆盖数据')
        print(f'  [重启] {acct} 正向 worker', flush=True)
        time.sleep(5)

def main():
    total = total_chunks()
    done = count_done()
    print(f'== 巡检 {time.strftime("%m-%d %H:%M")} ==', flush=True)
    print(f'进度: {done}/{total} ({100*done/total:.1f}%)', flush=True)
    snap_fp = os.path.join(BASE, 'patrol_status.json')
    prev = {}
    if os.path.exists(snap_fp):
        try:
            prev = json.load(open(snap_fp))
        except Exception:
            pass
    eta_h = ''
    if 'done' in prev and 'ts' in prev:
        dt = time.time() - prev['ts']
        if dt > 300:
            rate = (done - prev['done']) / (dt / 3600)
            remain = (total - done) / max(rate, 0.5)
            eta_h = f'{remain:.1f}h'
            print(f'速率: {rate:.1f} 块/h  预计剩余: {eta_h}', flush=True)
    failed = collect_failed()
    print(f'FINAL-FAIL 块: {len(failed)}', flush=True)
    retried_ok = 0
    still_fail = []
    if failed:
        idx_df = pd.read_parquet(IDX)
        hard_fp = os.path.join(BASE, 'hard_fail_chunks.json')
        hard = json.load(open(hard_fp)) if os.path.exists(hard_fp) else []
        for acct, cid in failed:
            key = f'{acct}_{cid}'
            if key in hard:
                continue
            ok = retry_chunk(acct, cid, idx_df)
            if ok:
                retried_ok += 1
                print(f'  [重试成功] {acct} chunk{cid}', flush=True)
            else:
                still_fail.append(key)
                if key not in hard:
                    hard.append(key)
                    json.dump(hard, open(hard_fp, 'w'))
                print(f'  [hard-fail] {acct} chunk{cid} 已存档', flush=True)
    alive = workers_alive()
    print(f'python 进程数(30h内): {alive}', flush=True)
    restarted = False
    if 0 <= alive < 4 and done < total - 5:
        print('警告: worker 不足，自动重启', flush=True)
        relaunch_workers()
        restarted = True
    json.dump({'done': done, 'total': total, 'ts': time.time(),
               'failed_retry_ok': retried_ok, 'still_fail': still_fail,
               'workers_alive': alive, 'restarted': restarted, 'eta': eta_h},
              open(snap_fp, 'w'), ensure_ascii=False, indent=2)
    with open(os.path.join(LOG_DIR, 'patrol.log'), 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%m-%d %H:%M")} done={done}/{total} '
                f'retry_ok={retried_ok} hard_fail={len(still_fail)} alive={alive}\n')

if __name__ == '__main__':
    main()
