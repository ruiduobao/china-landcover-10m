# -*- coding: utf-8 -*-
"""
d1_download_sources.py — 外部样本源下载（Zenodo 直链，断点续传 + 403 重试）
* 用法: python d1_download_sources.py [core|urban|all]
  core = GPW草地点(897MB) + WorldCereal Asia+Global(~25MB) + GALF(304MB)
  urban = GLC_FCS30D urban.cover 2022 (1.3GB)
"""
import os, sys, subprocess, time

DEST = r'数据/外部样本源'
os.makedirs(DEST, exist_ok=True)
UA = 'Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0'
PROXY = 'socks5h://127.0.0.1:7890'
API = 'https://zenodo.org/api/records'

FILES = {
    'core': [
        ('gpw_grassland_harm_point.gpkg',
         f'{API}/15631655/files/gpw_grassland_fscs.vi.vhr.harm_point.samples_20000101_20241231_go_epsg.4326_v2.gpkg/content',
         940_363_776),
        ('WorldCereal_ReferenceData_2018_Asia.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2018_Asia_Version_20260105.zip/content',
         9_794_064),
        ('WorldCereal_ReferenceData_2018_Global.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2018_Global_Version_20251008.zip/content',
         None),
        ('WorldCereal_ReferenceData_2019_Global.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2019_Global_Version_20251008.zip/content',
         None),
        ('WorldCereal_ReferenceData_2020_Global.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2020_Global_Version_20251008.zip/content',
         None),
        ('WorldCereal_ReferenceData_2021_Global.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2021_Global_Version_20251008.zip/content',
         None),
        ('WorldCereal_ReferenceData_2022_Global.zip',
         f'{API}/18769612/files/WorldCereal_ReferenceData_2022_Global_Version_20251008.zip/content',
         None),
        ('GALF_2000.zip', f'{API}/17343790/files/GALF_2000.zip/content', None),
        ('GALF_2010.zip', f'{API}/17343790/files/GALF_2010.zip/content', None),
        ('GALF_2020.zip', f'{API}/17343790/files/GALF_2020.zip/content', None),
    ],
    'urban': [
        ('glc_fcs30d_urban_cover_2022.tif',
         f'{API}/14439377/files/urban.cover_2022_30m.tif/content', 1_310_000_000),
    ],
}

def fetch(fname, url, expect):
    fp = os.path.join(DEST, fname)
    if os.path.exists(fp) and expect and os.path.getsize(fp) >= expect * 0.999:
        print(f'[skip] {fname} 已存在')
        return True
    for attempt in range(6):
        cmd = ['curl', '-L', '-C', '-', '-x', PROXY, '-A', UA,
               '--retry', '10', '--retry-delay', '3', '--retry-all-errors',
               '-sS', '-o', fp, url]
        rc = subprocess.call(cmd)
        if rc == 0 and os.path.exists(fp):
            sz = os.path.getsize(fp)
            print(f'[ok] {fname}: {sz/1e6:.1f} MB')
            return True
        print(f'[retry {attempt+1}] {fname} rc={rc}')
        time.sleep(3 + attempt * 3)
    print(f'[FAIL] {fname}')
    return False

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'core'
    targets = []
    if which in ('core', 'all'):
        targets += FILES['core']
    if which in ('urban', 'all'):
        targets += FILES['urban']
    ok = 0
    for f in targets:
        if fetch(*f):
            ok += 1
    print(f'\n完成 {ok}/{len(targets)}')
