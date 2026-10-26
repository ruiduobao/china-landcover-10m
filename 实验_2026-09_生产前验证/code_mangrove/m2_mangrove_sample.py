# -*- coding: utf-8 -*-
"""m2_mangrove_sample.py — 红树林扩充 第2步：0.25° 瓦片内 100 m 像元撒点（逐瓦片幂等）

* 掩膜：WC95 ∪ Giri2000（**逐源 unmask 后 Or**）；采样 scale=100 m，numPixels 给足（≥格点数）避免"先抽点后过滤"的下采样。
* 单瓦片返回 >5000 触发 GEE 上限 → 自动对半拆分递归（最多 3 层）。
* 幂等：cand/parts/p<ii>_t<jj>.parquet；已存在跳过。
* 用法：python m2_mangrove_sample.py --part 0 --nparts 4 --acct x2zdn8
"""
import argparse, json, os, sys, time
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
SCOPE = os.path.join(WORK, 'scope', 'tiles_0p25.json')
OUTD = os.path.join(WORK, 'cand', 'parts')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=4)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--scale', type=int, default=100)
    a = ap.parse_args()
    import ee, v31_common as VC
    VC.ensure_ctx(a.acct)
    sc = json.load(open(SCOPE, encoding='utf-8'))
    tiles = sc['tiles'][a.part::a.nparts]
    os.makedirs(OUTD, exist_ok=True)
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    giri = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS').first().select('1')
    mask = wc.eq(95).unmask(0).Or(giri.eq(1).unmask(0)).toByte().rename('mang')

    def grab(x, y, size, depth=0):
        tile = ee.Geometry.Rectangle([x, y, x + size, y + size])
        n_cells = int(((size * 111000) / a.scale) ** 2 * 1.3) + 100
        fc = mask.selfMask().sample(region=tile, scale=a.scale, numPixels=n_cells,
                                    geometries=True, dropNulls=True, seed=7, tileScale=4)
        try:
            return fc.getInfo()['features']
        except Exception as e:
            if ('5000' in str(e) or 'timed out' in str(e).lower()) and depth < 3:
                h = size / 2
                return (grab(x, y, h, depth + 1) + grab(x + h, y, h, depth + 1)
                        + grab(x, y + h, h, depth + 1) + grab(x + h, y + h, h, depth + 1))
            raise
    n_pts = n_skip = n_ok = 0
    for j, (x, y) in enumerate(tiles):
        fp = os.path.join(OUTD, 'p%02d_t%03d.parquet' % (a.part, j))
        if os.path.exists(fp):
            n_skip += 1; continue
        rows, err = [], ''
        for att in range(3):
            try:
                feats = grab(float(x), float(y), 0.25)
                rows = [dict(lon=f['geometry']['coordinates'][0], lat=f['geometry']['coordinates'][1])
                        for f in feats]
                err = ''; break
            except Exception as e:
                err = str(e)[:110]; time.sleep(6 * (att + 1))
        if err:
            with open(os.path.join(OUTD, '_fail_p%02d.json' % a.part), 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(tile=[x, y], err=err), ensure_ascii=False) + chr(10))
            emit('  tile %s,%s 失败：%s' % (x, y, err))
            continue
        d = pd.DataFrame(rows, columns=['lon', 'lat']).drop_duplicates()
        d['tile'] = '%g_%g' % (x, y)
        d.to_parquet(fp, index=False)
        n_pts += len(d); n_ok += 1
        if (j + 1) % 10 == 0:
            emit('  %d/%d 瓦片（累计 %d 点）' % (j + 1, len(tiles), n_pts))
    emit('完成 part%d：%d 写盘 / %d 跳过 / %d 点' % (a.part, n_ok, n_skip, n_pts))


if __name__ == '__main__':
    main()
