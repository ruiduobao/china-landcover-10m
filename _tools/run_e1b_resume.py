# -*- coding: utf-8 -*-
"""run_e1b_resume.py — T3 主嵌入续跑包装器（e1b_worker_f 的 FLEET 替换版）
* 索引/输出/断点仍用 F:\\r7_prod（862 块续跑）；
* FLEET 从 gee_accounts\\fleet_new_13accounts.json 加载（13 个空闲账号，禁动 19 个之外）；
* 每账号一个进程: python _tools/run_e1b_resume.py <账号名>；多账号由 launch_resume.bat 拉起。
"""
import os, sys, json

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
sys.path.insert(0, r'F:\r7_prod')

import e1b_worker_f as W

_new = json.load(open(os.path.join(
    r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts',
    'fleet_new_13accounts.json'), encoding='utf-8'))
W.FLEET = {a: (pid, fp) for a, (pid, fp) in _new.items()}
W.NACC = len(W.FLEET)

acct = sys.argv[1]
assert acct in W.FLEET, f'{acct} 不在新编队'
sys.argv = ['e1b_worker_f', acct]
W.main()
