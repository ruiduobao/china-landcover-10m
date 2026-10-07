# -*- coding: utf-8 -*-
"""
p9b_cd_analysis.py — 变化检测版 vs 逐年分类版 全面对比（本地）
* 输入: cd/{y1}_{y2}_p*.tif（p9 嵌入 L2 距离图）+ raster/{prefix}_{year}_final.tif（p7 平滑后 8 年）
* 流程:
  1) 距离块 mosaic → 重采样到分类网格
  2) 校准: 分类"未变化"(class(y1)==class(y2)) vs "变化" 像元的 L2 分布 → Youden 阈值
  3) CD 继承版序列: 2017 起步；L2>阈值 → 取新年份分类值，否则继承前一年
  4) 对比: 两版变化总量/逐年转移/不一致率；逐年版跳变被距离图支持的比率
* 输出: {out_dir}/cd_analysis.json + cd_对比.png + CD版 8 年 tif（cd_seq）
"""
import os, sys, glob, json
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

def mosaic_pair(pair, cd_dir):
    fps = sorted(glob.glob(os.path.join(cd_dir, f'{pair}_p*.tif')))
    if not fps:
        return None, None, None
    srcs = [rasterio.open(f) for f in fps]
    if len(srcs) == 1:
        arr = srcs[0].read(1).astype(np.float32)
        tr, crs = srcs[0].transform, srcs[0].crs
    else:
        arr, tr = rio_merge(srcs)
        arr = arr[0].astype(np.float32)
        crs = srcs[0].crs
    for s in srcs:
        s.close()
    return arr, tr, crs

def main():
    out_dir = PC.OUT_DIR
    cd_dir = os.path.join(out_dir, 'cd')
    pairs = [(y, y + 1) for y in PC.YEARS[:-1]] + [(PC.YEARS[0], PC.YEARS[-1])]

    # 1) 分类网格（参考）
    cls = {}
    with rasterio.open(os.path.join(out_dir, 'raster', f'{PC.TASK_PREFIX}_{PC.YEARS[0]}_final.tif')) as s:
        ref_tr, ref_crs, ref_shape = s.transform, s.crs, (s.height, s.width)
        cls[PC.YEARS[0]] = s.read(1)
    for y in PC.YEARS[1:]:
        with rasterio.open(os.path.join(out_dir, 'raster', f'{PC.TASK_PREFIX}_{y}_final.tif')) as s:
            cls[y] = s.read(1)
    H, W = ref_shape
    print(f'分类网格 {H}x{W}, 年份 {len(cls)}', flush=True)

    # 2) 距离图 → 重采样到分类网格
    dists = {}
    for y1, y2 in pairs:
        arr, tr, crs = mosaic_pair(f'{y1}_{y2}', cd_dir)
        if arr is None:
            print(f'[缺] {y1}_{y2}', flush=True)
            continue
        dst = np.full((H, W), np.nan, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs,
                  dst_transform=ref_tr, dst_crs=ref_crs,
                  src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        dists[(y1, y2)] = dst
        print(f'  距离 {y1}_{y2}: 有效 {np.isfinite(dst).mean():.1%} '
              f'中位 {np.nanmedian(dst):.2f}', flush=True)

    # 3) 校准（相邻年对）
    report = {'pairs': {}}
    thresholds = {}
    for (y1, y2), d in dists.items():
        if y2 - y1 != 1:
            continue
        same = cls[y1] == cls[y2]
        valid = np.isfinite(d) & (cls[y1] > 0) & (cls[y2] > 0)
        a = d[valid & same]
        b = d[valid & ~same]
        if len(a) < 100 or len(b) < 10:
            print(f'  {y1}_{y2} 样本不足跳过校准', flush=True)
            continue
        # Youden: max P(d>thr | 变化) + P(d<=thr | 未变化)
        qs = np.percentile(d[valid], [50, 75, 80, 85, 90, 92, 95, 96, 97, 98, 99])
        best_thr, best_j = None, -1
        for thr in qs:
            tpr = (b > thr).mean()
            tnr = (a <= thr).mean()
            if tpr + tnr > best_j:
                best_j, best_thr = tpr + tnr, float(thr)
        thresholds[(y1, y2)] = best_thr
        report['pairs'][f'{y1}_{y2}'] = {
            'n_same': int(len(a)), 'n_diff': int(len(b)),
            'l2_med_same': round(float(np.median(a)), 3),
            'l2_med_diff': round(float(np.median(b)), 3),
            'thr_youden': round(best_thr, 3),
            'tpr': round(float((b > best_thr).mean()), 3),
            'tnr': round(float((a <= best_thr).mean()), 3),
        }
        print(f"  {y1}_{y2}: 未变化中位 {np.median(a):.2f} vs 变化中位 {np.median(b):.2f} "
              f"→ 阈值 {best_thr:.2f} (TPR {(b>best_thr).mean():.2f} TNR {(a<=best_thr).mean():.2f})", flush=True)

    # 4) CD 继承版序列（相邻年对阈值）
    cd = {PC.YEARS[0]: cls[PC.YEARS[0]].copy()}
    n_cd_change = {}
    for y1, y2 in [(y, y + 1) for y in PC.YEARS[:-1]]:
        thr = thresholds.get((y1, y2))
        if thr is None:
            thr = np.nanmedian(list(thresholds.values())) if thresholds else 5.0
        d = dists.get((y1, y2))
        prev = cd[y1]
        cur = prev.copy()
        if d is not None:
            chg = np.isfinite(d) & (d > thr) & (cls[y2] > 0)
            cur[chg] = cls[y2][chg]
            n_cd_change[y2] = int(chg.sum())
        cd[y2] = cur
    # 端点 2017 vs 2024 校验
    dep = dists.get((PC.YEARS[0], PC.YEARS[-1]))
    endpoint_stat = None
    if dep is not None:
        v = np.isfinite(dep) & (cls[PC.YEARS[0]] > 0)
        endpoint_stat = {'l2_med': round(float(np.nanmedian(dep[v])), 3),
                         'p95': round(float(np.nanpercentile(dep[v], 95)), 3)}

    # 5) 对比
    n_cls_change = {}
    disc = 0
    for y1, y2 in [(y, y + 1) for y in PC.YEARS[:-1]]:
        chg_cls = (cls[y1] != cls[y2]) & (cls[y1] > 0) & (cls[y2] > 0)
        chg_cd = (cd[y1] != cd[y2]) & (cd[y1] > 0) & (cd[y2] > 0)
        n_cls_change[y2] = int(chg_cls.sum())
        disc += int((chg_cls ^ chg_cd).sum())
    total_px = H * W * (len(PC.YEARS) - 1)
    summary = {
        'thresholds': {f'{k[0]}_{k[1]}': v for k, v in thresholds.items()},
        'cd_change_px': n_cd_change,
        'cls_change_px': n_cls_change,
        'cd_change_total': int(sum(n_cd_change.values())),
        'cls_change_total': int(sum(n_cls_change.values())),
        'disagree_px_total': disc,
        'disagree_ratio_of_cls_change': round(disc / max(1, sum(n_cls_change.values())), 3),
        'endpoint_2017_2024': endpoint_stat,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    # 6) 输出 CD 版 tif + JSON + PNG
    os.makedirs(os.path.join(out_dir, 'raster_cd'), exist_ok=True)
    for y in PC.YEARS:
        with rasterio.open(os.path.join(out_dir, 'raster', f'{PC.TASK_PREFIX}_{y}_final.tif')) as s:
            meta = s.meta.copy()
        with rasterio.open(os.path.join(out_dir, 'raster_cd', f'CD_{y}.tif'), 'w', **meta) as dst:
            dst.write(cd[y][None])
    json.dump(summary, open(os.path.join(out_dir, 'cd_analysis.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)

    # 变化频率对比 PNG
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
    freq_cls = np.zeros((H, W), np.uint8)
    freq_cd = np.zeros((H, W), np.uint8)
    for y1, y2 in [(y, y + 1) for y in PC.YEARS[:-1]]:
        freq_cls += (cls[y1] != cls[y2]) & (cls[y1] > 0) & (cls[y2] > 0)
        freq_cd += (cd[y1] != cd[y2]) & (cd[y1] > 0) & (cd[y2] > 0)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=100)
    for ax, arr, tt in zip(axes, [freq_cls, freq_cd, freq_cls - freq_cd],
                           ['逐年独立版：变化次数', 'CD 继承版：变化次数', '差值（逐年 - CD）']):
        im = ax.imshow(arr, cmap='hot' if tt != axes[2] else 'coolwarm',
                       vmin=0, vmax=6 if tt != axes[2] else None,
                       extent=[0, W, H, 0])
        ax.set_title(tt, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'变化检测对比：CD 阈值继承版 vs 逐年独立分类版（{PC.YEARS[0]}-{PC.YEARS[-1]}）', fontsize=14)
    fig.tight_layout()
    png = os.path.join(out_dir, 'cd_对比.png')
    fig.savefig(png, bbox_inches='tight')
    plt.close(fig)
    print('输出:', json.dumps({'png': png, 'cd_raster': os.path.join(out_dir, "raster_cd")}), flush=True)

if __name__ == '__main__':
    main()
