# -*- coding: utf-8 -*-
"""rescue_copy.py — 把 Z 盘幸存的 chunkr7_*.parquet 增量拷到 F:\r7_prod\emb_parts_r7
* 容错逐文件拷贝（4号盘故障教训）；Z: 不可读时打印提示并正常退出（worker 会重提）
* 作用：把"已完成块"集合从 862 尽量恢复到 3785，将重提量从 ~3900 块降到 ~973 块
"""
import os, shutil, time

SRC = r'Z:\Mywork\论文\中国土地覆盖数据\数据\本地处理\全国清洗训练\emb_parts_r7'
DST = r'F:\r7_prod\emb_parts_r7'
os.makedirs(DST, exist_ok=True)
t0 = time.time()
try:
    files = [f for f in os.listdir(SRC)
             if f.startswith('chunkr7_') and f.endswith('.parquet')]
except Exception as e:
    print('Z: 不可读（%s），跳过抢救，worker 将重提缺失块' % str(e)[:80])
    raise SystemExit(0)
ok = fail = skip = 0
for n, f in enumerate(files):
    try:
        s, d = os.path.join(SRC, f), os.path.join(DST, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            skip += 1
            continue
        shutil.copyfile(s, d)
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 5:
            print('FAIL', f, str(e)[:60], flush=True)
    if (n + 1) % 1000 == 0:
        print(f'{n+1}/{len(files)} {time.time()-t0:.0f}s', flush=True)
print(f'抢救完成: 新拷 {ok}, 已在F盘 {skip}, 失败 {fail}, 用时 {time.time()-t0:.0f}s', flush=True)
