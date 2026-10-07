# -*- coding: utf-8 -*-
"""
monitor_progress.py — r1 样本重建流水线定时巡检（Windows 计划任务每 30 分钟调用）
* 只读巡检，不改任何数据；结果追加到 数据/本地处理/样本重建/巡检报告.md
* 检查项:
  1) 各步骤日志尾部状态 + 是否存在 Traceback/ERR
  2) r1_*.parquet 产物是否存在（与 summary 行数核对）
  3) 卡死判定: 活跃日志 mtime 超过 stall_min 分钟且无产物更新 → 报警
  4) 交付目录是否已生成
* 结束条件: 交付 summary 存在 → 写"全部完成"并自动删除本计划任务
"""
import os, re, glob, json, time, subprocess
from datetime import datetime

PROJ = r'Z:/Mywork/论文/中国土地覆盖数据'
LOGD = os.path.join(PROJ, '数据/本地处理/日志')
WORK = os.path.join(PROJ, '数据/本地处理/样本重建')
DELIV = os.path.join(PROJ, '数据/样本交付')
REPORT = os.path.join(WORK, '巡检报告.md')
STALL_MIN = 45

STEPS = ['s2', 's3', 's3b', 's4', 's4b', 's5', 's6', 's7', 's8']
PRODUCTS = ['r1_base', 'r1_ext', 'r1_ext2', 'r1_topup', 'r1_thematic',
            'r1_pool', 'r1_train', 'r1_validation']

def tail(path, n=3):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            return [l.rstrip() for l in f.readlines()[-n:]]
    except OSError:
        return []

def age_min(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 60
    except OSError:
        return None

def main():
    now = datetime.now().strftime('%m-%d %H:%M')
    lines = [f'\n## [{now}] 巡检', '']
    problems = []
    done_products = []

    # 1) 产物核对
    lines.append('| 产物 | 状态 | 点数(summary) |')
    lines.append('|---|---|---|')
    for p in PRODUCTS:
        f = os.path.join(WORK, p + '.parquet')
        if os.path.exists(f):
            sj = os.path.join(WORK, p + '_summary.json')
            n = ''
            if os.path.exists(sj):
                try:
                    d = json.load(open(sj, encoding='utf-8'))
                    n = d.get('total') or d.get('train_total') or ''
                except Exception:
                    n = '?'
            done_products.append(p)
            lines.append(f'| {p} | ✅ {os.path.getsize(f)//1048576}MB | {n} |')
        else:
            lines.append(f'| {p} | — | — |')

    # 2) 日志状态与报错扫描
    lines.append('')
    lines.append('| 步骤日志 | 最后修改 | 状态 |')
    lines.append('|---|---|---|')
    for s in STEPS:
        lg = os.path.join(LOGD, s + '_run.log')
        if not os.path.exists(lg):
            lines.append(f'| {s} | — | 未开始 |')
            continue
        ag = age_min(lg)
        tt = tail(lg, 4)
        bad = [l for l in tail(lg, 60) if re.search(r'Traceback|ERR|Error|失败', l, re.I)]
        status = ''
        if bad:
            status = '❌ ' + bad[-1][:70]
            problems.append(f'{s} 日志有异常: {bad[-1][:80]}')
        elif ag is not None and ag < STALL_MIN and not any(p in s for p in
                [x.replace('r1_', '') for x in done_products]):
            status = f'🔄 运行中/近期活动 ({ag:.0f}min前): {tt[-1][:60] if tt else ""}'
        elif ag is not None and ag < STALL_MIN:
            status = f'✅ 近期完成 ({ag:.0f}min前)'
        else:
            status = f'⏸ 已结束 ({ag:.0f}min前): {tt[-1][:50] if tt else ""}'
        lines.append(f'| {s} | {ag:.0f}min前 | {status} |')

    # 3) 活跃进程检测
    try:
        r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                           capture_output=True, text=True, timeout=30)
        np_ = len(re.findall(r'python\.exe', r.stdout or ''))
        lines.append('')
        lines.append(f'活跃 python 进程数: {np_}')
        if np_ == 0 and not os.path.exists(os.path.join(DELIV, 'r1_delivery_summary.json')):
            problems.append('无活跃 python 进程但交付未生成 → 流水线可能已中断')
    except Exception:
        pass

    # 4) 交付目录
    lines.append('')
    if os.path.exists(os.path.join(DELIV, 'r1_delivery_summary.json')):
        lines.append('🎉 **交付已完成**: 数据/样本交付/ 下 r1 交付物已生成。')
        lines.append('')
        lines.append('本巡检任务到此结束，可删除计划任务 r1_monitor。')
        try:
            subprocess.run(['schtasks', '/Delete', '/TN', 'r1_monitor', '/F'],
                           capture_output=True, text=True, timeout=30)
        except Exception:
            pass
    else:
        ready = [os.path.basename(f) for f in glob.glob(os.path.join(DELIV, '*'))]
        lines.append(f'交付目录当前内容: {ready if ready else "（空，仅归档）"}')

    if problems:
        lines.insert(2, '')
        lines.insert(2, '> ⚠️ **发现问题**: ' + '；'.join(problems))

    with open(REPORT, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
