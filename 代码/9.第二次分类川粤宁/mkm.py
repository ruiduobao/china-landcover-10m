# -*- coding: utf-8 -*-
"""mkm.py — 生成代码目录 MANIFEST.md（MD5 清单），一次运行，可重跑刷新。"""
import hashlib, os, io
D = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁'
NOTE = {
    'c0_health.py': '账号健康复检（串行；并发会串号）',
    'c1_upload_pool_v2.py': '池上传链（分片→组→宿主 merge；QCAP=3）',
    'c2_prep_prod.py': '生产前置（prod2nd 目录 + ACL 共享 + 跨账号校验）',
    'c3_driver_prod2.py': '生产驱动运行副本（工作区 F:/lc_work/prod2nd_2023/run/prod2_driver.py 为执行体，两者同 MD5）',
    'ov_build_副本.py': '概览重建+PIL 验收（工作区 run/ 同 MD5，doc43 纪律）',
    'c4_eval_second.py': '评估（同窗同口径 + 与第一次对比）',
    'c5_mosaic_compare.py': '拼接对比（mosaic/compare/qa）',
}
rows = ['# 代码 MD5 清单（2026-10-14 建，生产收官后再补终稿）', '',
        '| 文件 | MD5 | 字节 | 说明 |', '|---|---|---|---|']
for f in sorted(os.listdir(D)):
    if not f.endswith('.py'):
        continue
    p = os.path.join(D, f)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()
    rows.append('| `%s` | `%s` | %d | %s |' % (f, h, os.path.getsize(p), NOTE.get(f, '')))
io.open(os.path.join(D, 'MANIFEST.md'), 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
