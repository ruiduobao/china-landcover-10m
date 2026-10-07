# -*- coding: utf-8 -*-
"""run_e1c_rare.py — T2：用空闲账号提取稀有类 9 块嵌入（e1c 的 FLEET 收缩版包装）
用法: python _tools/run_e1c_rare.py <账号名> [第二账号]
* 从项目根运行；E1_IDX/E1_OUT 固定指向项目内稀有类索引与输出目录。
* FLEET 收缩为传入账号 → NACC=len(FLEET)，块按 chunk_id%NACC 分配（9 块 1-2 账号全覆盖）。
"""
import os, sys

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, '代码', '4.全国清洗训练'))

os.environ['E1_IDX'] = '数据/本地处理/全国清洗训练/chunks_index_rare.parquet'
os.environ['E1_OUT'] = '数据/本地处理/全国清洗训练/emb_parts_rare'

import e1c_worker_generic as W

accts = sys.argv[1:]
assert accts, '用法: python _tools/run_e1c_rare.py <账号名> [第二账号]'
for a in accts:
    assert a in W.FLEET, f'{a} 不在 e1c FLEET（新账号需先在 FLEET 补 anchor+指纹）'
W.FLEET = {a: W.FLEET[a] for a in accts}
W.NACC = len(W.FLEET)
sys.argv = ['e1c'] + accts
W.main()
