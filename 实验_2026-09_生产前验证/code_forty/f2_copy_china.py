# -*- coding: utf-8 -*-
"""f2_copy_china.py — ForTy 中国相关**文件** → D 盘工作副本（E 盘只读，逐文件字节校验）

* 输入：F:/lc_work/forty_exp/map/forty_file_index.csv（f1 产出，in_china=True 的行）
* 规则源：
  ① 顶层目录是"UTM 带-纬度带"粗格（如 raster_48R，跨 6°×8°），跨境粗格内含境外块 →
     因此**按文件 bbox 与省界相交**逐个挑选，而非整目录拷贝；
  ② 只拷 in_china 文件 + 数据介绍.txt；拷后逐文件比对字节数；E 盘零写入。
* 门槛：校验失败文件数必须为 0
* 输出：D:/lcwork_forty/ForTy_china/<tile>/<file>（工作副本）+ copy_manifest.json
* 用法：python f2_copy_china.py [--workers 6]
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

SRC = r'E:/地理所/论文/博士论文/数据/ForTy_v1/32301177'
WORKMAP = r'F:/lc_work/forty_exp/map'
DEST = r'D:/lcwork_forty/ForTy_china'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def copy_one(rel):
    sf = os.path.join(SRC, rel.replace('/', os.sep))
    df = os.path.join(DEST, rel.replace('/', os.sep))
    os.makedirs(os.path.dirname(df), exist_ok=True)
    if not (os.path.exists(df) and os.path.getsize(df) == os.path.getsize(sf)):
        shutil.copy2(sf, df)
    return rel if os.path.getsize(df) != os.path.getsize(sf) else None


def main():
    idx = pd.read_csv(os.path.join(WORKMAP, 'forty_file_index.csv'), encoding='utf-8-sig')
    files = idx[idx.in_china == True].relpath.tolist()
    emit('中国相关文件 %d 个（%.2f GB）→ %s' % (
        len(files), idx[idx.in_china == True].size_bytes.sum() / 1e9, DEST))
    os.makedirs(DEST, exist_ok=True)
    t0, bad = time.time(), []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(copy_one, files)):
            if r:
                bad.append(r)
            if (i + 1) % 500 == 0:
                emit('  进度 %d/%d（坏 %d）' % (i + 1, len(files), len(bad)))
    intro = os.path.join(os.path.dirname(SRC), '数据介绍.txt')
    if os.path.exists(intro):
        shutil.copy2(intro, os.path.join(DEST, '数据介绍.txt'))
    tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DEST) for f in fs)
    man = {'src': SRC, 'dest': DEST, 'files': len(files), 'bad': bad[:20], 'n_bad': len(bad),
           'size_gb': round(tot / 1e9, 2), 'seconds': round(time.time() - t0, 1)}
    json.dump(man, open(os.path.join(WORKMAP, 'copy_manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('拷贝完成：%d 文件 / %.1f GB / %.0f 秒；坏 %d' % (
        len(files), tot / 1e9, time.time() - t0, len(bad)))
    if bad:
        raise SystemExit('校验失败：%d 个文件字节不一致' % len(bad))


if __name__ == '__main__':
    main()
