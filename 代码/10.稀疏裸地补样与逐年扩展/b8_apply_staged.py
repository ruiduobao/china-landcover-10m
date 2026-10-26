# -*- coding: utf-8 -*-
"""b8_apply_staged.py — 把暂存的南方裸地 GPKG 应用到类目录（QGIS 关闭后运行）

* 背景：QGIS 打开着裸地 GPKG 时 Windows 锁定文件，b6 只能把新文件写入
  `裸地/_staging_2026-10-26/`；本脚本在目标可写时完成替换（备份已在 b6 用 copy 完成）。
* 用法：python b8_apply_staged.py
"""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
CDIR = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/裸地'
STAGE = os.path.join(CDIR, '_staging_2026-10-26')
BK = os.path.join(CDIR, '旧样本_2026-10-26_补样前')


def main():
    applied, pending = [], []
    for y in range(2017, 2025):
        stg = os.path.join(STAGE, '裸地_%d_WGS84.gpkg' % y)
        fp = os.path.join(CDIR, str(y), '裸地_%d_WGS84.gpkg' % y)
        bk = os.path.join(BK, str(y), '裸地_%d_WGS84.gpkg' % y)
        if not os.path.isfile(stg):
            continue
        if not os.path.isfile(bk):
            os.makedirs(os.path.dirname(bk), exist_ok=True)
            shutil.copy2(fp, bk)
        try:
            os.replace(stg, fp)
            applied.append(y)
        except PermissionError as e:
            pending.append((y, str(e)[:80]))
    print('已应用：', applied)
    print('仍被占用：', pending)
    if not pending:
        print('全部完成，可删除暂存目录：', STAGE)


if __name__ == '__main__':
    main()
