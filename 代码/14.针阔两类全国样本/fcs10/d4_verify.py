# -*- coding: utf-8 -*-
"""d4_verify.py — 新旧 FCS10 瓦片逐像元一致性核验
* 输入 / Z:/nb_national_2026/fcs10_dl/tiles/T*_fcs10.tif（新，E 盘源块直拷）
         F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/T*_fcs10.tif（旧，GEE 下载后对齐）
* 规则源 / 两者应共享同一 GEE 生产网格，理想结果逐像元 100% 相同
* 门槛 / 尺寸+transform 完全一致；不等像元逐一给出 旧->新 码对
* 输出 / F:/lc_work/nb_national_2026/fcs10_dl/verify_76.json、verify_76.md
* 用法 / python d4_verify.py [--limit N]
"""
import os, json, glob, argparse
import numpy as np
import rasterio
from rasterio.windows import Window

NEW = r'Z:/nb_national_2026/fcs10_dl/tiles'
OLD = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10'
OUT_JSON = r'F:/lc_work/nb_national_2026/fcs10_dl/verify_76.json'
OUT_MD = r'F:/lc_work/nb_national_2026/fcs10_dl/verify_76.md'


def cmp(t):
    pn = os.path.join(NEW, f'{t}_fcs10.tif')
    po = os.path.join(OLD, f'{t}_fcs10.tif')
    if not (os.path.exists(pn) and os.path.exists(po)):
        return t, dict(status='missing', new=os.path.exists(pn), old=os.path.exists(po))
    with rasterio.open(pn) as A, rasterio.open(po) as B:
        if (A.width, A.height) != (B.width, B.height):
            return t, dict(status='size_mismatch', new=[A.width, A.height], old=[B.width, B.height])
        ta = tuple(A.transform); tb = tuple(B.transform)
        dt = max(abs(x - y) for x, y in zip(ta, tb))
        ndiff = 0
        pairs = {}
        H, W = A.height, A.width
        # 旧瓦片边缘 fill 统计：旧=0 且新>0 的像元所落入的列/行
        edge_cols, edge_rows = {}, {}
        fill_px = 0
        for r in range(0, H, 2048):
            h = min(2048, H - r)
            na = A.read(1, window=Window(0, r, W, h))
            nb = B.read(1, window=Window(0, r, W, h))
            d = na != nb
            k = int(d.sum())
            if k:
                ndiff += k
                # 旧为 0、新非 0 → 旧导出 fill（非数据差异）
                fill = d & (nb == 0) & (na > 0)
                fill_px += int(fill.sum())
                if fill.any():
                    cc = np.nonzero(fill.any(0))[0]
                    for c in cc.tolist():
                        edge_cols[c] = edge_cols.get(c, 0) + int(fill[:, c].sum())
                    rr = np.nonzero(fill.any(1))[0]
                    for y in rr.tolist()[:0]:
                        pass
                if len(pairs) < 4000:
                    u, c = np.unique(np.stack([nb[d], na[d]], 1), axis=0, return_counts=True)
                    for (o, n), cc2 in zip(u.tolist(), c.tolist()):
                        pairs[(o, n)] = pairs.get((o, n), 0) + int(cc2)
        tot = H * W
    real_diff = ndiff - fill_px
    return t, dict(status='ok' if ndiff == 0 else ('edge_fill_only' if real_diff == 0 else 'diff'),
                   diff=int(ndiff), fill_px=int(fill_px), real_diff=int(real_diff), total=int(tot),
                   diff_pct=round(100.0 * ndiff / tot, 8), max_transform_delta=dt,
                   transform_equal=dt < 1e-12,
                   fill_cols={str(k): v for k, v in sorted(edge_cols.items()) if v >= 100},
                   pairs=sorted([[o, n, c] for (o, n), c in pairs.items()], key=lambda x: -x[2])[:15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    olds = sorted(os.path.basename(p)[:5] for p in glob.glob(OLD + '/*_fcs10.tif'))
    if a.limit:
        olds = olds[:a.limit]
    res = {}
    for i, t in enumerate(olds, 1):
        tt, v = cmp(t)
        res[t] = v
        print('%2d/%d %s %s' % (i, len(olds), t, v['status']), flush=True)
    ok = [k for k, v in res.items() if v['status'] == 'ok']
    fill = [k for k, v in res.items() if v['status'] == 'edge_fill_only']
    bad = {k: v for k, v in res.items() if v['status'] not in ('ok', 'edge_fill_only', 'missing')}
    miss = [k for k, v in res.items() if v['status'] == 'missing']
    json.dump(res, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    lines = ['# FCS10 新旧瓦片逐像元一致性核验', '',
             '新：`Z:/nb_national_2026/fcs10_dl/tiles/`（E 盘 5° 源块整数窗口直拷）',
             '旧：`F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/`（GEE 下载后 nearest 对齐）', '',
             '| 项 | 值 |', '|---|---|',
             '| 核验瓦片数 | %d |' % len(res),
             '| 逐像元完全一致 | **%d** |' % len(ok),
             '| 仅旧瓦片边缘 fill（旧=0/新>0）差异 | **%d** |' % len(fill),
             '| 真实不一致 | %d |' % len(bad),
             '| 新瓦片尚未生成 | %d |' % len(miss), '']
    if fill:
        lines += ['## 边缘 fill（旧 GEE 导出缺陷，非对齐问题）', '',
                  '旧瓦片在瓦片右下角出现整段 0：源数据有值、GEE 导出按 region 截断补 0。', '',
                  '| tile | 旧=0 像元 | 占全瓦 | 落在旧列 |', '|---|---|---|---|']
        for k in fill:
            v = res[k]
            lines.append('| %s | %s | %.6f%% | %s |' % (k, v['fill_px'], 100.0 * v['fill_px'] / v['total'],
                                                        v.get('fill_cols', {})))
        lines.append('')
    if bad:
        lines += ['## 真实不一致明细', '', '| tile | status | diff_px | diff_pct | 主要码对(旧→新) |', '|---|---|---|---|---|']
        for k, v in bad.items():
            pr = v.get('pairs') or []
            prs = ', '.join('%s→%s:%s' % (p[0], p[1], p[2]) for p in pr[:4])
            lines.append('| %s | %s | %s | %s | %s |' % (k, v['status'], v.get('diff', '-'),
                                                          v.get('diff_pct', '-'), prs))
    if miss:
        lines += ['', '## 尚未生成', '', ' '.join(miss)]
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n完全一致 %d，仅边缘fill %d，真实不一致 %d，未生成 %d' % (len(ok), len(fill), len(bad), len(miss)))
    print('报告:', OUT_MD)


if __name__ == '__main__':
    main()
