# -*- coding: utf-8 -*-
"""s3_poll_download.py — 轮询各 worker 的 v31_ 任务；SUCCEEDED 即下载预测 CSV；超3h停滞赛跑加注
判据：只认本实验 desc（v31_<jobid>_ 前缀）；EECU 从 batchEecuUsageSeconds 取。
僵尸纪律（doc35）：不 cancel，停滞>3h 追加赛跑尝试（不同账号新 desc），谁先成用谁。
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd
from s2_submit import submit_one

STALL_H = 1.5  # 小表格任务（分钟级）不应等 3h；受限/卡死账号 1.5h 即赛跑加注


def fetch_states(acct, tries=4):
    """{desc: (state, eecu_h)}；只取 v31_ 前缀。代理抖动 → 退避重试。"""
    last = ''
    for i in range(tries):
        try:
            s, pid = VC.sess(acct)
            r = s.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                      params={'pageSize': 100}, headers={'X-Goog-User-Project': pid}, timeout=60)
            r.raise_for_status()
            out = {}
            for o in r.json().get('operations', []):
                md = o.get('metadata', {})
                d = md.get('description', '')
                if not d.startswith('v31_'):
                    continue
                e = 0.0
                try:
                    e = float(md.get('batchEecuUsageSeconds', 0)) / 3600.0
                except Exception:
                    pass
                out[d] = (md.get('state') or ('DONE' if o.get('done') else '?'), round(e, 3))
            return out
        except Exception as e:
            last = str(e)[:80]
            time.sleep(8 * (i + 1))
    VC.emit('⚠️ %s 轮询 %d 次仍失败：%s' % (acct, tries, last))
    return None


def download_csv(acct, asset, fp, tries=3):
    import requests
    import prod_conf as PC
    last = ''
    for i in range(tries):
        try:
            e, _ = VC.ctx(acct)
            fc = e.FeatureCollection(asset)
            url = fc.getDownloadURL(filetype='csv', selectors=['rid', 'classification'])
            r = requests.get(url, proxies=PC.PROXY, timeout=600)
            r.raise_for_status()
            with open(fp, 'wb') as f:
                f.write(r.content)
            return os.path.getsize(fp)
        except Exception as ex:
            last = str(ex)[:90]
            time.sleep(10 * (i + 1))
    raise RuntimeError('下载重试 %d 次仍失败: %s' % (tries, last))


def main():
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    jobs = VC.jload(VC.JOBS_FP, [])
    if not jobs:
        VC.emit('无任务'); return
    need = {a for j in jobs for a in
            ([j['attempts'][-1]['acct']] if j.get('attempts') and
             j['attempts'][-1].get('state') in ('SUBMITTED', 'RUNNING') else [])}
    stmap = {}
    for acct in need:
        r = fetch_states(acct)
        if r is not None:
            stmap[acct] = r
    n_done = n_open = 0
    for j in jobs:
        atts = j.get('attempts', [])
        if not atts:
            continue
        att = atts[-1]
        if att.get('state') == 'SUCCEEDED':
            n_done += 1
            continue
        if att.get('state') == 'DOWNLOAD_FAIL' and att.get('asset'):
            # 云端已成功，直接重下原资产（不重投任务）
            fp = os.path.join(VC.RAW, j['job_id'] + '.csv')
            try:
                sz = download_csv(att['acct'], att['asset'], fp)
                att['state'] = 'SUCCEEDED'
                att['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
                j['result_csv'] = fp
                j['done'] = True
                n_done += 1
                VC.emit('✅ %s 重下成功 %dKB' % (j['job_id'], sz // 1024))
            except Exception as e:
                VC.emit('❌ %s 重下仍失败: %s' % (j['job_id'], str(e)[:80]))
            continue
        if att.get('state') not in ('SUBMITTED', 'RUNNING', 'PENDING'):
            continue
        st = stmap.get(att['acct'], {}).get(att['desc'])
        if st:
            att['state'], att['eecu_h'] = st[0], st[1]
        if att['state'] == 'SUCCEEDED':
            fp = os.path.join(VC.RAW, j['job_id'] + '.csv')
            try:
                sz = download_csv(att['acct'], att['asset'], fp)
                att['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
                j['result_csv'] = fp
                j['done'] = True
                n_done += 1
                VC.emit('✅ %s 下载 %dKB (EECU=%.3fh)' % (j['job_id'], sz // 1024,
                                                          att.get('eecu_h') or 0))
            except Exception as e:
                att['state'] = 'DOWNLOAD_FAIL'
                VC.emit('❌ %s 下载失败: %s' % (j['job_id'], str(e)[:100]))
        elif att['state'] in ('FAILED', 'CANCELLED'):
            VC.emit('❌ %s 任务 %s' % (j['job_id'], att['state']))
        else:
            age_h = (time.time() - time.mktime(time.strptime(
                att['submitted'], '%Y-%m-%d %H:%M:%S'))) / 3600.0
            if age_h > STALL_H:
                VC.emit('⏳ %s 已 %.1fh 未完成 → 赛跑加注' % (j['job_id'], age_h))
                try:
                    att2 = submit_one(pool, j['window'], j)
                    att2['state'] = 'SUBMITTED'
                    att2['race_of'] = att['desc']
                    atts.append(att2)
                except Exception as e:
                    VC.emit('  赛跑提交失败: %s' % str(e)[:90])
            else:
                n_open += 1
    VC.jsave(jobs, VC.JOBS_FP)
    VC.emit('s3: 完成 %d / 在飞 %d / 总 %d' % (
        n_done, n_open, sum(1 for j in jobs if j.get('attempts'))))


if __name__ == '__main__':
    main()
