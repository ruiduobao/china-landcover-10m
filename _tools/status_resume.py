# -*- coding: utf-8 -*-
"""status_resume.py — 一次性进度/健康状态（供 CRON 与人工调用，只读）
输出: 总进度 / 备份数 / 监督器与各 worker 存活 / 每账号最新日志行 / 代理连通 / 台账摘要
"""
import os, glob, json, subprocess, datetime

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
OUT_DIR = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024\数据\本地处理\全国清洗训练\emb_parts_r7'
BK_DIR = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024\数据\本地处理\全国清洗训练\emb_parts_r7'
LOGD = r'F:\r7_prod\logs'
LEDGER = os.path.join(LOGD, '资源台账.json')
TARGET = 4758


def main():
    done = len(glob.glob(os.path.join(OUT_DIR, 'chunkr7_*.parquet')))
    bk = len(glob.glob(os.path.join(BK_DIR, 'chunkr7_*.parquet')))
    print(f'进度: {done}/{TARGET}  备份(K盘): {bk}')
    # 监督器存活（优先 pid 文件，wmic 仅兜底）
    sup = False
    workers = set()
    pf = os.path.join(LOGD, 'supervisor.pid')
    if os.path.isfile(pf):
        try:
            old = int(open(pf).read().strip())
            import psutil
            p = psutil.Process(old)
            if 'supervise_resume' in ' '.join(p.cmdline() or []):
                sup = True
        except Exception:
            sup = False
    try:
        import psutil
        for pr in psutil.process_iter(['cmdline']):
            cl = ' '.join(pr.info.get('cmdline') or [])
            if 'run_e1b_resume.py' in cl:
                try:
                    workers.add(cl.split('run_e1b_resume.py')[1].strip().split()[0])
                except Exception:
                    pass
    except Exception as e:
        print(f'进程检查失败: {str(e)[:80]}')
    print(f'监督器: {"存活" if sup else "❌不在运行"}   worker 存活: {sorted(workers) if workers else "无"}')
    # 各账号最新日志行
    for f in sorted(glob.glob(os.path.join(LOGD, 'resume_*.log'))):
        a = os.path.basename(f)[7:-4]
        try:
            lines = [l for l in open(f, encoding='utf-8', errors='ignore').read().splitlines() if l.strip()]
            print(f'  {a:14s} {lines[-1][:110] if lines else "(空)"}')
        except Exception:
            pass
    # 代理
    try:
        import requests
        r = requests.get('https://earthengine.googleapis.com', timeout=10,
                         proxies={'http': 'socks5h://127.0.0.1:7890',
                                  'https': 'socks5h://127.0.0.1:7890'})
        print(f'代理: 通 (HTTP {r.status_code})')
    except Exception as e:
        print(f'代理: ❌ {str(e)[:80]}')
    # 台账
    if os.path.isfile(LEDGER):
        j = json.load(open(LEDGER, encoding='utf-8'))
        print(f"台账: {j.get('updated')}  运行 {j.get('elapsed_h')}h")

    if done >= TARGET:
        print('✅ 全部块齐——可接链 e2b 年度 QC')
    elif not sup:
        print('⚠️ 监督器不在运行——需重启: python _tools/supervise_resume.py (从项目根)')


if __name__ == '__main__':
    main()
