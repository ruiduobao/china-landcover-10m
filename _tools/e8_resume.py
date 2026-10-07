# -*- coding: utf-8 -*-
"""
e8_resume.py — e8 GPKG 导出的自愈驱动（抗宿主周期性清理）

为什么要它：e8 单次要导 22 个 GPKG（29 GB），宿主会周期性清理后台 python 进程，
每次都在中途被杀。e8 本身是幂等的（已存在且行数一致就跳过），所以只要"反复拉起"即可收敛。
本驱动循环调用 e8，直到连续两轮都没有新增产出为止（最多 N 轮）。

用法: python _tools/e8_resume.py [--max-rounds 30]
"""
import os
import re
import sys
import time
import subprocess
import argparse

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
E8 = os.path.join(ROOT, '代码/4.全国清洗训练/e8_export_gpkg.py')
LOG = os.path.join(ROOT, '_tools/log_e8_resume.txt')


def run_once():
    t0 = time.time()
    p = subprocess.run([sys.executable, '-u', E8, '--all'], cwd=ROOT,
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=10800)
    out = (p.stdout or '') + (p.stderr or '')
    wrote = len(re.findall(r'→.*\.gpkg', out))
    skipped = out.count('跳过')
    return wrote, skipped, time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-rounds', type=int, default=30)
    a = ap.parse_args()
    idle = 0
    for r in range(1, a.max_rounds + 1):
        wrote, skipped, dt, out = run_once()
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f'\n===== 第 {r} 轮 {time.strftime("%H:%M:%S")} '
                    f'写出 {wrote} / 跳过 {skipped} / {dt:.0f}s =====\n')
            f.write(out[-4000:])
        print(f'第 {r} 轮：写出 {wrote} 个，跳过 {skipped} 个（{dt:.0f}s）', flush=True)
        if wrote == 0:
            idle += 1
            if idle >= 2:
                print('连续两轮无新增产出 → 收敛完成', flush=True)
                return 0
        else:
            idle = 0
        time.sleep(5)
    print(f'达到最大轮数 {a.max_rounds}，请检查', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
