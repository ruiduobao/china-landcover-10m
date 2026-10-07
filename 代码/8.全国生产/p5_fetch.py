# -*- coding: utf-8 -*-
"""
p5_fetch.py — 从 GEE 资产下载已完成的分区影像并本地拼接成 tile
为什么分块：GEE 单次 getDownloadURL 硬上限 48 MB（实测报
`Total request size (...) must be ≤ 50331648`）。10 m 单波段 uint8 下 0.25° ≈ 7.7 MB，安全。
流程：台账查 (tile,year) 的 asset → 逐子块 getDownloadURL → 本地 rasterio 拼接
      → 写 STAGE/<year>/<tile>.tif → 台账标 DONE_FETCHED（幂等：已存在且校准则跳过）
用法：python p5_fetch.py --acct <acct> --tile T0123 --year 2022 [--all-done] [--keep-asset]
"""
import os, sys, io, json, time, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def find_asset(acct, tile, year):
    """提交行带 tile/year/asset，refresh 行带 desc/state——两者必须按 desc 关联，
    否则永远匹配不到“已完成”（2026-09-15 首瓦片下载时踩到）。"""
    fp = os.path.join(C.LEDGER, 'ledger_%s.jsonl' % acct)
    task_state, cands = {}, []
    if os.path.isfile(fp):
        for line in io.open(fp, encoding='utf-8', errors='replace'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task'):
                task_state[r['task']] = r.get('state')
            if r.get('tile') == tile and int(r.get('year', -1)) == int(year) and r.get('asset'):
                if r.get('state') in ('SUCCEEDED', 'SUBMITTED'):
                    cands.append(r)      # 同一瓦片可能被重提交多次 → 收集全部
    for c in cands:                       # 任一次提交的任务跑到 SUCCEEDED 即算完成
        if task_state.get(c.get('task')) == 'SUCCEEDED':
            return dict(c, state='SUCCEEDED')
    return None


def mark(acct, tile, year, state, extra=None):
    fp = os.path.join(C.LEDGER, 'ledger_%s.jsonl' % acct)
    rec = {'acct': acct, 'tile': tile, 'year': int(year), 'state': state,
           'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
    if extra:
        rec.update(extra)
    with io.open(fp, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def boot(acct):
    import ee
    C.apply_account_env(acct)
    for i in range(5):
        try:
            ee.Initialize(project=C.ACCOUNTS[acct]['proj'])
            break
        except Exception as x:
            if i == 4:
                raise
            emit('init retry %d: %s' % (i + 1, str(x)[:70]))
            time.sleep(20)
    return ee


def fetch_tile(ee, acct, tile_row, year, sub=None):
    import requests, rasterio
    from rasterio.merge import merge as rmerge
    sub = sub or C.SUBTILE_DEG
    x0, y0, x1, y1 = tile_row['x0'], tile_row['y0'], tile_row['x1'], tile_row['y1']
    tile = tile_row['tile']
    rec = find_asset(acct, tile, year)
    if not rec:
        emit('台账里找不到 %s %s %d 的资产' % (acct, tile, year))
        return None
    img = ee.Image(rec['asset']).select('class')
    outdir = os.path.join(C.STAGE, str(year), tile)
    os.makedirs(outdir, exist_ok=True)
    nx = int(round((x1 - x0) / sub)); ny = int(round((y1 - y0) / sub))
    parts = []
    for a in range(nx):
        for b in range(ny):
            bx0, by0 = x0 + a * sub, y0 + b * sub
            fp = os.path.join(outdir, 'p%02d%02d.tif' % (a, b))
            parts.append(fp)
            if os.path.isfile(fp) and os.path.getsize(fp) > 200:
                continue
            reg = ee.Geometry.Rectangle([bx0, by0, bx0 + sub, by0 + sub])
            for att in range(3):
                try:
                    url = img.getDownloadURL(dict(region=reg, scale=10, crs='EPSG:4326',
                                                  format='GEO_TIFF'))
                    r = requests.get(url, proxies=C.PROXY, timeout=900)
                    r.raise_for_status()
                    open(fp, 'wb').write(r.content)
                    break
                except Exception as x:
                    if att == 2:
                        emit('  子块 %02d%02d 失败: %s' % (a, b, str(x)[:80]))
                        return None
                    time.sleep(8 + 8 * att)
    srcs = [rasterio.open(p) for p in parts]
    arr, tr = rmerge(srcs)
    prof = srcs[0].profile.copy()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr,
                count=1, dtype=arr.dtype, compress='deflate')
    out = os.path.join(C.STAGE, str(year), '%s_%d.tif' % (tile, year))
    with rasterio.open(out, 'w', **prof) as dst:
        dst.write(arr[0].astype('uint8'), 1)
    for s in srcs:
        s.close()
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    return out, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', required=True)
    ap.add_argument('--tile', default='')
    ap.add_argument('--year', type=int, default=0)
    ap.add_argument('--all-done', action='store_true')
    ap.add_argument('--no-net', action='store_true')
    ap.add_argument('--plan', default='tiles.json')
    a = ap.parse_args()
    C.ensure_dirs()
    tiles = {t['tile']: t for t in
             json.load(open(os.path.join(C.PLAN, a.plan), encoding='utf-8'))['tiles']}
    todo = []
    if a.tile and a.year:
        todo = [(a.tile, a.year)]
    elif a.all_done:
        fp = os.path.join(C.LEDGER, 'ledger_%s.jsonl' % a.acct)
        task_state, cand = {}, {}
        for line in io.open(fp, encoding='utf-8', errors='replace'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task'):
                task_state[r['task']] = r.get('state', '')
            if r.get('tile') and r.get('year') is not None and r.get('asset') and r.get('task'):
                cand.setdefault((r['tile'], int(r['year'])), []).append(r['task'])
        todo = sorted(k for k, tids in cand.items()
                      if any(task_state.get(x) == 'SUCCEEDED' for x in tids))
    if not todo:
        print('没有待下载项（--tile/--year 或 --all-done）'); return 0
    emit('%s 待下载 %d 个分区' % (a.acct, len(todo)))
    ee = None if a.no_net else boot(a.acct)
    ok = 0
    for tile, year in todo:
        t = tiles.get(tile)
        if not t:
            emit('跳过未知分区 %s' % tile); continue
        out = os.path.join(C.STAGE, str(year), '%s_%d.tif' % (tile, year))
        if os.path.isfile(out) and os.path.getsize(out) > 10000:
            emit('%s %d 已存在，跳过' % (tile, year)); ok += 1; continue
        r = fetch_tile(ee, a.acct, t, year)
        if r:
            fp, arr = r
            n_ok = int((arr[0] > 0).sum())
            emit('%s %d → %s（%.1f MB，非零像元 %.2f%%）'
                 % (tile, year, os.path.basename(fp), os.path.getsize(fp) / 1e6,
                    100.0 * n_ok / arr[0].size))
            mark(a.acct, tile, year, 'DONE_FETCHED', {'file': fp, 'nonzero_frac':
                                                      round(n_ok / arr[0].size, 4)})
            ok += 1
    emit('完成 %d/%d' % (ok, len(todo)))
    print('暂存目录:', os.path.join(C.STAGE))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
