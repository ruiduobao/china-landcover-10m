# -*- coding: utf-8 -*-
"""
d2_glc_fcs10_china.py — 下载 GLC_FCS10（10m 30类 2023）中国经度带 zip，作为第二教师
* 记录: 14729665；中国需 lon 带 E070-E075 … E130-E135 共 7 个 zip（约 25-45GB）
* 用法: python d2_glc_fcs10_china.py   （逐个下载，断点续传，磁盘不足自动停）
"""
import os, sys, requests, subprocess, time

UA = 'Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0'
PROXY = 'socks5h://127.0.0.1:7890'
DEST = r'数据/外部样本源/glc_fcs10_2023'
os.makedirs(DEST, exist_ok=True)
RID = '14729665'
BANDS = ['E070-E075', 'E080-E085', 'E090-E095', 'E100-E105',
         'E110-E115', 'E120-E125', 'E130-E135']

def disk_free_gb(path='Z:'):
    import shutil
    return shutil.disk_usage(path).free / 1e9

def main():
    r = requests.get(f'https://zenodo.org/api/records/{RID}', proxies=PROXY and {'http': PROXY, 'https': PROXY},
                     headers={'User-Agent': UA}, timeout=120)
    files = {f['key']: (f['size'], f['links']['self']) for f in r.json()['files']}
    for band in BANDS:
        key = f'GLC_FCS10maps_2023_{band}.zip'
        if key not in files:
            print('[miss]', key); continue
        size, url = files[key]
        fp = os.path.join(DEST, key)
        if os.path.exists(fp) and os.path.getsize(fp) >= size * 0.999:
            print('[skip]', key); continue
        free = disk_free_gb()
        if free < size / 1e9 + 5:
            print(f'[stop] 磁盘不足({free:.0f}GB)，需要 {size/1e9:.0f}GB'); return
        print(f'[dl] {key} {size/1e9:.1f}GB …', flush=True)
        rc = subprocess.call(['curl', '-sL', '-C', '-', '-x', PROXY, '-A', UA,
                              '--retry', '10', '--retry-delay', '3', '--retry-all-errors',
                              '-o', fp, url])
        print(f'[done rc={rc}] {key} {os.path.getsize(fp)/1e9:.1f}GB', flush=True)
        time.sleep(2)

if __name__ == '__main__':
    main()
