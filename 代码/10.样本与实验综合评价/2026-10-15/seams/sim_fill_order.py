# -*- coding: utf-8 -*-
"""sim_fill_order — 小型可复现模拟：c5「先布放者胜」拼接覆盖是否存在零覆盖 bug

* 输入：无（合成瓦片）；规则源 = 代码/9.第二次分类川粤宁/c5_mosaic_compare.py
    `read_tile_into` L186-232（先布放者胜：`m = (seg == 0) & (blk > 0)`）、
    `build_placements` L167-185（fx=(x0-x_left)/px 后 **round** 取整）、
    `merge_tiles` L236-... （分块逐瓦布放）
* 门槛：S1/S4 必须零零覆盖；S2 必须复现"1 像元零覆盖行"（证明该类 bug 可检出）；
    S3 必须证明源端 nodata 拼接不可修复（只能源端重导）
* 输出：sim_fill_order.json（每案结论 + 零覆盖像元数）
* 用法：python sim_fill_order.py
"""
import os
import sys
import json
import math

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
PXDEG = 8.983152841195215e-05


def read_tile_into_impl(blk_src, p, buf, r0, hh, c0, w):
    """c5.read_tile_into 的等价实现（同一条规则；exact 同格直拷分支）。
    blk_src: p 的完整数组（替代 rasterio 读窗，其余逐行同源）。"""
    ov_r0 = max(r0, p['row0'])
    ov_r1 = min(r0 + hh, p['row0'] + p['height'])
    ov_c0 = max(c0, p['col0'])
    ov_c1 = min(c0 + w, p['col0'] + p['width'])
    if ov_r1 <= ov_r0 or ov_c1 <= ov_c0:
        return 0
    blk = blk_src[ov_r0 - p['row0']:ov_r1 - p['row0'], ov_c0 - p['col0']:ov_c1 - p['col0']]
    seg = buf[ov_r0 - r0:ov_r1 - r0, ov_c0 - c0:ov_c1 - c0]
    m = (seg == 0) & (blk > 0)
    n = int(m.sum())
    np.copyto(seg, blk, where=m)
    return n


def tile(v, h=60, w=60, row0=0, col0=0, val=7, south_nodata=0):
    a = np.full((h, w), val, dtype=np.uint8)
    if south_nodata:
        a[-south_nodata:, :] = 0
    return dict(row0=row0, col0=col0, height=h, width=w), a


def merge(frame_h, frame_w, placements, chunk=1000):
    buf = np.zeros((frame_h, frame_w), dtype=np.uint8)
    stats = []
    for p, a in placements:
        stats.append(read_tile_into_impl(a, p, buf, 0, frame_h, 0, frame_w))
    return buf, stats


def zero_coverage_rows(buf):
    return [int(r) for r in np.where((buf == 0).all(axis=1))[0]]


def main():
    res = {}
    # ---- S1：真实布放（1 像元重叠 + round 取整）→ 必须零零覆盖 ----
    A, a = tile(0, row0=0, col0=0, val=7)            # 北瓦：行 0..59
    B, b = tile(0, row0=59, col0=0, val=3)           # 南瓦：行 59..118（重叠 1 行）
    buf, _ = merge(119, 60, [(A, a), (B, b)])
    res['S1_real_overlap_1px'] = dict(
        zero_rows=zero_coverage_rows(buf), overlap_row_value=int(buf[59, 0]),
        note='1 像元重叠 + round：重叠行由先布放（北瓦 7）胜；无零覆盖行')

    # ---- S2：相位误用 floor（北瓦短 1 像元 / 南瓦起行错 1）→ 必须复现 1 行零覆盖 ----
    A2, a2 = tile(0, h=59, row0=0, col0=0, val=7)    # 北瓦只到行 58（floor 归格少 1 行）
    B2, b2 = tile(0, row0=60, col0=0, val=3)         # 南瓦从行 60 起 → 行 59 谁都不盖
    buf2, _ = merge(120, 60, [(A2, a2), (B2, b2)])
    res['S2_floor_offby1'] = dict(
        zero_rows=zero_coverage_rows(buf2),
        note='若布放用 floor 而非 round（或相位非整）→ 恰好 1 行零覆盖；c5 build_placements 用 round+exact 门 + qa_identity 回归拦截（L167-185/L291-313）')

    # ---- S3：源端南边 nodata（本次实况：测地楔）→ 拼接不可修复 ----
    A3, a3 = tile(0, row0=0, col0=0, val=7, south_nodata=25)   # 北瓦南边 25 行 nodata
    B3, b3 = tile(0, row0=59, col0=0, val=3)
    buf3, _ = merge(119, 60, [(A3, a3), (B3, b3)])
    lost = int((buf3 == 0).sum())
    res['S3_source_nodata_edge'] = dict(
        zero_rows=zero_coverage_rows(buf3), lost_px=lost,
        note='重叠仅 1 行 → 拼接只能补 1 行，源端 25 行 nodata 中 24 行永久为空；证明必须源端重导，不能邻行填')

    # ---- S4：顺序反转 → 覆盖不变，仅重叠行取值来源变化 ----
    buf4, _ = merge(119, 60, [(B, b), (A, a)])
    res['S4_order_reversed'] = dict(
        zero_rows=zero_coverage_rows(buf4), overlap_row_value=int(buf4[59, 0]),
        note='覆盖与顺序无关；仅重叠行归属变化（先布放者胜 → 接缝带类值随布放顺序）')

    res['_summary'] = dict(
        S1_pass=len(res['S1_real_overlap_1px']['zero_rows']) == 0,
        S2_reproduces_1row_bug=res['S2_floor_offby1']['zero_rows'] == [59],
        S3_mosaic_cannot_heal=res['S3_source_nodata_edge']['lost_px'] > 0,
        S4_coverage_order_invariant=len(res['S4_order_reversed']['zero_rows']) == 0)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(HERE, 'sim_fill_order.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('WROTE sim_fill_order.json')


if __name__ == '__main__':
    main()
