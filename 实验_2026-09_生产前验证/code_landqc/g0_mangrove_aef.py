# -*- coding: utf-8 -*-
"""g0_mangrove_aef.py — 红树林新增点 AEF64 补抓（2023；用户指令"缺失 AEF 下载补全"）

* 输入：红树林/扩充_v1/红树林_新增_10066_2023_WGS84.shp（cid/lon/lat）
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023，逐 band unmask(-9999) 后 cat（生产同源），scale=10
* 输出：data/mangrove_aef/aef_p<part>_c<chunk>.parquet（point_id=cid + A00..A63）
* 分片：--part/--nparts；块 2000（子请求 1000）；幂等
* 用法：python g0_mangrove_aef.py --part 0 --nparts 2 --acct vlgvl6u
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
SHP = (r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
       r'/红树林/扩充_v1/红树林_新增_10066_2023_WGS84.shp')
OUTD = os.path.join(WORK, 'data', 'mangrove_aef')
SENT = -9999.0


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + 'P%d %s' % (os.getpid(), str(m)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--nparts', type=int, default=2)
    ap.add_argument('--acct', required=True)
    ap.add_argument('--step', type=int, default=2000)
    ap.add_argument('--sub', type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(WORK, 'code'))
    import ee
    import v31_common as VC
    VC.ensure_ctx(a.acct)
    r = shapefile.Reader(SHP, encoding='utf-8')
    fl = [x[0] for x in r.fields[1:]]
    recs = list(r.iterRecords())
    pts = [p.points[0] for p in r.iterShapes()]
    r.close()
    d = pd.DataFrame([dict(point_id=str(x[fl.index('cid')]), lon=p[0], lat=p[1])
                      for x, p in zip(recs, pts)])
    mine = d.iloc[a.part::a.nparts].reset_index(drop=True)
    os.makedirs(OUTD, exist_ok=True)
    img = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate('2023-01-01', '2024-01-01').mosaic())
    img = ee.Image.cat([img.select(b).unmask(SENT).rename(b) for b in VC.FEATS]).toFloat()
    emit('本片 %d 点 / %d 块' % (len(mine), (len(mine) + a.step - 1) // a.step))
    for k in range(0, len(mine), a.step):
        fp = os.path.join(OUTD, 'aef_p%02d_c%04d.parquet' % (a.part, k // a.step))
        if os.path.exists(fp):
            continue
        blk = mine.iloc[k:k + a.step]
        rows, failed = [], False
        for s0 in range(0, len(blk), a.sub):
            sub = blk.iloc[s0:s0 + a.sub]
            fcp = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'point_id': p})
                                        for p, x, y in zip(sub.point_id, sub.lon, sub.lat)])
            got, err = None, ''
            for att in range(6):
                try:
                    res = img.sampleRegions(collection=fcp, scale=10, geometries=False,
                                            tileScale=4).getInfo()['features']
                    got = {f['properties']['point_id']: f['properties'] for f in res}; break
                except Exception as e:
                    err = str(e)[:110]
                    emit('  块%d 子%d 重试%d %s' % (k // a.step, s0 // a.sub, att + 1, err))
                    time.sleep(8 * (att + 1))
            if got is None:
                with open(os.path.join(OUTD, '_fail_p%02d.json' % a.part), 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(dict(part=a.part, chunk=k // a.step, err=err), ensure_ascii=False) + '\n')
                failed = True; break
            rows += [dict(point_id=p, **{c: got.get(p, {}).get(c) for c in VC.FEATS}) for p in sub.point_id]
        if failed:
            continue
        o = pd.DataFrame(rows)
        for c in VC.FEATS:
            o[c] = pd.to_numeric(o[c], errors='coerce')
        o['aef_ok'] = (o['A00'] > -9000).astype('int8')
        for c in VC.FEATS:
            o[c] = o[c].where(o.aef_ok == 1, np.nan)
        o.to_parquet(fp, index=False)
        emit('  %d 块完成' % (k // a.step + 1))
    emit('完成 p%02d' % a.part)


if __name__ == '__main__':
    main()
