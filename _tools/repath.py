# -*- coding: utf-8 -*-
"""
repath.py — 把代码里的 Z 盘路径批量重指到新位置（先备份，可回滚）
映射规则（按长度降序匹配，避免前缀误伤）：
  项目根两代 → F:/地理所/论文/中国土地覆盖数据_2017-2024
  GUB        → E:/data/非洲城市发展和驱动力分析/数据/...
  SDPT       → E:/地理所/论文/博士论文/数据/数据/...
  CLCD       → E:/2.1_CLCD土地覆盖数据/...
  人口密度   → E:/地理所/论文/中国人口密度2000-2026/...
  F:/r7_stage(重GPKG暂存) → F:/lc_stage
"""
import os, shutil, io, re

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
BACKUP = os.path.join(ROOT, '_复原记录', '路径重指前备份')
NEWROOT = 'F:/地理所/论文/中国土地覆盖数据_2017-2024'

MAP = [
    (r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据', NEWROOT),
    (r'Z:\Mywork\论文\BaiduSyncdisk\中国土地覆盖数据', NEWROOT.replace('/', '\\')),
    (r'Z:/Mywork/论文/中国土地覆盖数据', NEWROOT),
    (r'Z:\Mywork\论文\中国土地覆盖数据', NEWROOT.replace('/', '\\')),
    (r'Z:/Mywork/论文/非洲城市发展和驱动力分析/数据', 'E:/data/非洲城市发展和驱动力分析/数据'),
    (r'Z:\Mywork\论文\非洲城市发展和驱动力分析\数据', 'E:\\data\\非洲城市发展和驱动力分析\\数据'),
    (r'Z:/Mywork/论文/博士论文/数据/数据', 'E:/地理所/论文/博士论文/数据/数据'),
    (r'Z:\Mywork\论文\博士论文\数据\数据', 'E:\\地理所\\论文\\博士论文\\数据\\数据'),
    (r'Z:/Mywork/论文/中国人口密度2000-2026/1.数据', 'E:/2.1_CLCD土地覆盖数据_PARENTHINT'),
    (r'Z:/Mywork/论文/中国人口密度2000-2026', 'E:/地理所/论文/中国人口密度2000-2026'),
    (r'Z:\Mywork\论文\中国人口密度2000-2026', 'E:\\地理所\\论文\\中国人口密度2000-2026'),
    (r'Z:/Mywork/论文/论文idea', 'F:/BaiduSyncdisk/论文idea'),
    (r'F:/r7_stage', 'F:/lc_stage'),
    (r'F:\r7_stage', 'K:\\lc_stage'),
    (r'F:/r1_stage', 'F:/lc_stage_r1'),
    (r'F:\r1_stage', 'K:\\lc_stage_r1'),
]

def main():
    files = []
    for sub in ('代码', '生产_F盘自包含'):
        for dirpath, _d, fs in os.walk(os.path.join(ROOT, sub)):
            for f in fs:
                if f.endswith(('.py', '.bat', '.cmd', '.json')):
                    files.append(os.path.join(dirpath, f))
    changed, leftover = [], []
    for fp in files:
        rel = os.path.relpath(fp, ROOT)
        s0 = io.open(fp, encoding='utf-8').read()
        s = s0
        for old, new in MAP:
            s = s.replace(old, new)
        # CLCD 特例：/1.数据/2.1_CLCD土地覆盖数据 → E:/2.1_CLCD土地覆盖数据
        s = s.replace('E:/2.1_CLCD土地覆盖数据_PARENTHINT/2.1_CLCD土地覆盖数据', 'E:/2.1_CLCD土地覆盖数据')
        if s != s0:
            bkp = os.path.join(BACKUP, rel)
            os.makedirs(os.path.dirname(bkp), exist_ok=True)
            if not os.path.isfile(bkp):
                shutil.copyfile(fp, bkp)
            io.open(fp, 'w', encoding='utf-8', newline='').write(s)
            changed.append(rel)
        if 'Z:' in s:
            leftover.append(rel)
    print(f'扫描 {len(files)} 个文件，改写 {len(changed)} 个；备份于 {BACKUP}')
    for r in sorted(changed): print('  改写:', r)
    print(f'\n仍有 Z: 残留 {len(leftover)} 个（多为注释/说明）:')
    for r in sorted(leftover):
        fp = os.path.join(ROOT, r)
        hits = [ln.strip()[:110] for ln in io.open(fp, encoding='utf-8').read().splitlines() if 'Z:' in ln]
        print(f'  {r}  ({len(hits)} 处)')
        for h in hits[:3]: print('     ', h)

if __name__ == '__main__':
    main()
