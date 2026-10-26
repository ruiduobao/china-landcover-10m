# -*- coding: utf-8 -*-
"""h4_apply_fix.py — 混淆对定向修复执行：备份 → 按年删除 → 统计重算 → 留痕

* 输入：out/删除清单_<类>.csv（h3 冻结判据产出；逐点 reason）
* 步骤（逐类）：
    1) **备份**：<类>/ 下全部 *.gpkg → 数据/备份/单类样本_混淆对优化前_2026-10-14/<类>/（保持同名）
    2) 对每个年份 GPKG：删去清单内点（几何/AEF 原样保留），tmp 写入 → 校验（行数差=删除数、幸存者抽检）
    3) 重算 <类>/逐年质检统计.csv（n / strong / weak / note=混淆对修复删 N）
    4) 池类重算 逐年池成员数.csv（新 2023 文件 ∩ r7_train_<年>）
    5) 删除清单复制到 <类>/；README 追加修复说明
* 用法：python h4_apply_fix.py --classes 木本沼泽,草本沼泽 [--no-backup]
"""
import argparse, glob, os, shutil, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile  # noqa

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
BK = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/备份/单类样本_混淆对优化前_2026-10-14'
ANN = r'F:/lc_work/年度子集_含稀有类'
ALLDIR = {'草本沼泽': '草本沼泽', '木本沼泽': '木本沼泽', '湖河滩地': '湖河滩地', '灌木_常绿': '常绿灌木',
          '灌木_落叶': '落叶灌木', '草地': '草地', '乔灌园地': '乔灌园地', '稀疏植被': '稀疏植被',
          '裸地': '裸地', '灌溉耕地': '灌溉耕地', '草本旱地': '草本旱地', '水体': '水体'}
POOL = {'草本沼泽', '木本沼泽', '湖河滩地', '草地', '稀疏植被', '裸地', '灌溉耕地', '草本旱地', '水体'}
IDCOL = {'灌木_常绿': 'point_id', '灌木_落叶': 'point_id', '乔灌园地': 'uid'}
SPECS = {  # h3 用的 source 映射（删清单里 point_id 的语义）
    '草本沼泽': 'pid', '木本沼泽': 'pid', '湖河滩地': 'pid', '草地': 'pid', '稀疏植被': 'pid',
    '裸地': 'pid', '灌溉耕地': 'pid', '草本旱地': 'pid', '水体': 'pid',
    '灌木_常绿': 'point_id', '灌木_落叶': 'point_id', '乔灌园地': 'uid'}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--no-backup', action='store_true')
    a = ap.parse_args()
    for cls in [x.strip() for x in a.classes.split(',') if x.strip()]:
        folder = os.path.join(ROOT, ALLDIR[cls])
        lst = pd.read_csv(os.path.join(WORK, 'data', 'confuse_fix', 'out', '删除清单_%s.csv' % cls),
                          encoding='utf-8-sig')
        if len(lst) == 0:
            emit('%s：删除清单为空，跳过' % cls); continue
        dele = set(lst.point_id.astype(str))
        idcol = SPECS[cls]
        files = sorted(glob.glob(os.path.join(folder, '[0-9][0-9][0-9][0-9]', '*.gpkg')))
        # 1) 备份
        if not a.no_backup:
            for f in files:
                rel = os.path.relpath(f, folder)
                dst = os.path.join(BK, cls, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(f, dst)
            emit('%s：备份 %d 个 GPKG → %s' % (cls, len(files), os.path.join(BK, cls)))
        # 2) 逐文件删除
        stats = []
        for f in files:
            y = int(os.path.basename(os.path.dirname(f)))
            g = gpd.read_file(f, engine='pyogrio')
            idv = g[idcol].astype(str)
            m = idv.isin(dele)
            k = int(m.sum())
            n0 = len(g)
            if k:
                keep = g[~m].reset_index(drop=True)
                tmp = f + '.tmp.gpkg'
                if os.path.exists(tmp):
                    os.remove(tmp)
                keep.to_file(tmp, driver='GPKG', layer=os.path.basename(f)[:-5], engine='pyogrio',
                             SPATIAL_INDEX='NO')
                b = gpd.read_file(tmp, engine='pyogrio')
                assert len(b) == n0 - k, (len(b), n0, k)
                if len(b):
                    rng = np.random.RandomState(5).permutation(len(b))[:20]
                    ids_ok = set(b[idcol].astype(str)) & dele
                    assert not ids_ok, '残留删除点 %d' % len(ids_ok)
                    assert int(pd.to_numeric(b['A00'], errors='coerce').isna().sum()) == 0, 'AEF 出现空值!'
                del b, g, keep
                os.replace(tmp, f)
            stats.append(dict(year=y, n=n0 - k, removed=k))
            emit('  %s %d：%d → %d（删 %d）' % (cls, y, n0, n0 - k, k))
        # 3) 逐年质检统计
        st = []
        for f in files:
            y = int(os.path.basename(os.path.dirname(f)))
            g = gpd.read_file(f, columns=[c for c in ('tier',) if True], engine='pyogrio') if 'tier' in \
                gpd.read_file(f, rows=1, engine='pyogrio').columns else None
            cnt = stats[[s['year'] for s in stats].index(y)]
            st.append(dict(year=y, n=cnt['n'], removed=cnt['removed'], note='混淆对定向修复 v1'))
        pd.DataFrame(st).to_csv(os.path.join(folder, '逐年质检统计.csv'), index=False, encoding='utf-8-sig')
        # 4) 池成员数重算
        if cls in POOL:
            f23 = os.path.join(folder, '2023', '%s' % os.path.basename(
                glob.glob(os.path.join(folder, '2023', '*.gpkg'))[0]))
            g23 = gpd.read_file(f23, columns=[idcol], engine='pyogrio')
            pid = pd.to_numeric(g23[idcol].astype(str), errors='coerce').dropna().to_numpy(np.int64)
            rows = []
            for y in range(2017, 2025):
                s = pd.read_parquet(os.path.join(ANN, 'r7_train_%d.parquet' % y), columns=['row_id'])
                m = s.row_id.to_numpy(np.int64)
                n = int(np.isin(pid, m).sum())
                rows.append(dict(year=y, n_in_pool=n, pct=round(100.0 * n / max(1, len(pid)), 2)))
            pd.DataFrame(rows).to_csv(os.path.join(folder, '逐年池成员数.csv'), index=False, encoding='utf-8-sig')
        # 5) 清单与 README
        lst.to_csv(os.path.join(folder, '删除清单_混淆对修复_%s.csv' % cls), index=False, encoding='utf-8-sig')
        rp = os.path.join(folder, 'README_样本文件说明.md')
        if os.path.exists(rp):
            with open(rp, 'a', encoding='utf-8') as fh:
                fh.write('\n> **2026-10-14 混淆对定向修复**：按 h3 冻结判据删除 %d 点（原因见 `删除清单_混淆对修复_%s.csv`）；'
                         '逐年质检统计/池成员数已按修复后重算；修复前副本备份于 `数据/备份/单类样本_混淆对优化前_2026-10-14/%s/`。\n'
                         % (len(lst), cls, cls))
        emit('%s：完成，删 %d' % (cls, len(lst)))


if __name__ == '__main__':
    main()
