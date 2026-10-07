# -*- coding: utf-8 -*-
"""
p9_postfix.py — 成品图后处理（纯本地，不改模型；解决"海绵类"与碎片化）
两类规则，按需启用：
  1) 筛除小斑块（sieving）：把指定类里面积小于阈值的连通斑块，按周围 5×5 多数类重分配。
     依据（技术文档 29 §1.3）：121 落叶灌丛空间纯度仅 51%、邻居 18% 旱地 + 16% 郁闭阔叶林
     → 它在"吸收混合像元"，表现为大量孤立小斑；筛除后面积应回落到与 WorldCover 相称的量级。
  2) 类别归并（merge）：如 92→61（92 纯度仅 20.5%、43% 邻居是 61，本身即伪类）。
判据：只用 p8_audit.py 的前后对比说话——面积是否变合理、命中率是否不退化。
注意：后处理只能修"**过度预测**"（121），不能凭空造出"**漏检**"的类（130/91 必须靠更好的模型）。
用法：
  python p9_postfix.py --in <成品.tif> --out <输出.tif> --sieving 121:8 --merge 92:61
"""
import os, sys, time, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '0.本地流水线'))
import prod_conf as C
from lc_conf import CLASSES

sys.stdout.reconfigure(encoding='utf-8')
MEM_LIMIT_GB = 20.0
CPU_LIMIT = 14


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def guard(tag):
    try:
        import psutil
        r = psutil.Process().memory_info().rss / 1e9
        if r > MEM_LIMIT_GB:
            raise SystemExit('[%s] 内存 %.1fG 超限，中止' % (tag, r))
        emit('  [%s] RSS %.2f GB' % (tag, r))
    except ImportError:
        pass


def sieve(a, cls, min_px, target_whitelist=None):
    """把 cls 类中小于 min_px 的连通斑块，按 5×5 邻域多数类重分配"""
    from scipy import ndimage
    m = (a == cls)
    lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
    if n == 0:
        return a, 0
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < min_px)
    small = small[small != 0]
    if len(small) == 0:
        return a, 0
    kill = np.isin(lab, small)
    emit('  %d 类：%d 个斑块，其中 < %d 像元的有 %d 个（占该类的 %.1f%%）'
         % (cls, n, min_px, len(small), 100.0 * kill.sum() / max(m.sum(), 1)))
    # 邻域多数票（用 5×5 众数近似：对每个候选项，统计邻域内该类的出现次数）
    cnt = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            cnt[(dy, dx)] = np.roll(np.roll(a, dy, 0), dx, 1)
    # 只在候选中挑：非 0、非该小斑块本身
    best = np.zeros_like(a)
    bestn = np.zeros_like(a, dtype=np.int16)
    for (dy, dx), nb in cnt.items():
        ok = kill & (nb > 0) & (nb != cls)
        if target_whitelist:
            ok &= np.isin(nb, target_whitelist)
        # 累计：对每个取值做一次比较累加（取值数少，循环可行）
        for v in np.unique(nb[ok]):
            hit = ok & (nb == v)
            better = hit & (bestn + 1 > bestn)          # 计数
            best = np.where(hit, np.where(bestn == 0, v, best), best)
            bestn = np.where(hit, bestn + 1, bestn)
    fill = kill & (bestn > 0)
    a2 = a.copy()
    a2[fill] = best[fill]
    emit('  重分配 %d 像元（%.1f%% 的该类）' % (int(fill.sum()),
                                            100.0 * fill.sum() / max(m.sum(), 1)))
    return a2, int(fill.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sieving', nargs='*', default=[], help='形如 121:8（类:最小像元数）')
    ap.add_argument('--merge', nargs='*', default=[], help='形如 92:61（源类:目标类）')
    a = ap.parse_args()
    import rasterio
    if a.merge:
        merges = {}
        for s in a.merge:
            k, v = s.split(':')
            merges[int(k)] = int(v)
    t0 = time.time()
    with rasterio.open(a.src) as s:
        prof = s.profile.copy()
        arr = np.asarray(s.read(1))
    emit('读入 %s，%d×%d' % (os.path.basename(a.src), arr.shape[1], arr.shape[0]))
    guard('读入后')
    before = np.bincount(arr.ravel(), minlength=256)

    out = arr
    for spec in a.sieving:
        cls, mp = (int(x) for x in spec.split(':'))
        out, _ = sieve(out, cls, mp)
        guard('筛除 %d 后' % cls)
    if a.merge:
        for k, v in merges.items():
            n = int((out == k).sum())
            out = np.where(out == k, v, out).astype(np.uint8)
            emit('  归并 %d → %d（%d 像元）' % (k, v, n))

    after = np.bincount(out.ravel(), minlength=256)
    prof.update(dtype='uint8', compress='deflate')
    with rasterio.open(a.out, 'w', **prof) as d:
        d.write(out.astype(np.uint8), 1)
        try:
            with rasterio.open(a.src) as s0:
                cm = s0.colormap(1)
            if cm:
                d.write_colormap(1, cm)
        except Exception:
            pass
    emit('写出 %s（%.1f MB）' % (a.out, os.path.getsize(a.out) / 1e6))
    try:
        from rasterio.enums import Resampling
        with rasterio.Env(GDAL_NUM_THREADS=str(CPU_LIMIT), NUM_THREADS=str(CPU_LIMIT)):
            with rasterio.open(a.out, 'r+') as d:
                d.build_overviews([2, 4, 8, 16, 32, 64], Resampling.mode)
        emit('金字塔已重建（mode）')
    except Exception as e:
        emit('金字塔重建失败: %s' % str(e)[:80])

    rows = []
    for c in range(1, 256):
        if before[c] or after[c]:
            rows.append((c, CLASSES[int(c)][1], int(before[c]), int(after[c]),
                         int(after[c]) - int(before[c])))
    rows.sort(key=lambda r: -abs(r[4]))
    emit('\n=== 面积变化最大的 10 类（像元数，100m 格元）===')
    emit('%-4s%-14s%12s%12s%12s' % ('类', '名称', '前', '后', '变化'))
    for c, nm, b, af, d_ in rows[:10]:
        emit('%-4d%-14s%12d%12d%+12d' % (c, nm, b, af, d_))
    json.dump({'src': a.src, 'out': a.out, 'sieving': a.sieving, 'merge': a.merge,
               'changes': [{'class': c, 'name': nm, 'before': b, 'after': af, 'delta': d_}
                           for c, nm, b, af, d_ in rows if d_],
               'elapsed_s': round(time.time() - t0)},
              open(r'F:/lc_work/postfix_report.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('耗时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
