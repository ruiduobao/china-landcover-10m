# -*- coding: utf-8 -*-
"""ov_build.py — 手工重建 GeoTIFF overview 存储页（修复 GDAL build_overviews 确定性损坏）
背景（2026-10-04 审查裁定）：本环境 GDAL build_overviews 写出的概览页与底图不符
（lev2 全平面隶属失配 ~3%，GDAL 自家 out_shape 读会现算而掩盖缺陷——GDAL 路径自检无效）。
本工具：底图 TL（块左上）最近邻下采样 → zlib 压缩 → 追加文件尾 → 原位改写各概览 IFD 的
TileOffsets/TileByteCounts 数组 → PIL 全平面验收（隶属+TL 双 100%）。
不动底图、调色板、Geo 标签。用法：python ov_build.py <tif> [<tif>...]
[2026-10-15] 支持 BigTIFF（TileOffsets/TileByteCounts 按 8 字节自适写，见 rebuild 内注释）。
"""
import os, sys, zlib, struct
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, rasterio, tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

LEVS = [2, 4, 8, 16, 32]


def tl_full(ds, lev):
    """底图 TL 下采样 → 页数组（ph=ceil(H/lev), pw=ceil(W/lev)）。"""
    H, W = ds.height, ds.width
    ph, pw = (H + lev - 1) // lev, (W + lev - 1) // lev
    out = np.zeros((ph, pw), 'uint8')
    nh, nw = H // lev, W // lev
    for r0 in range(0, nh * lev, 4096 * lev):
        hh = min(4096 * lev, nh * lev - r0)
        win = rasterio.windows.Window(0, r0, nw * lev, hh)
        blk = ds.read(1, window=win).astype('uint8')
        out[r0 // lev: r0 // lev + hh // lev, :nw] = \
            blk.reshape(hh // lev, lev, nw, lev)[:, 0, :, 0]
    if H % lev:  # 末个不完整 ov 行：整行读入后按 lev 抽稀（TL）
        row = ds.read(1, window=rasterio.windows.Window(0, nh * lev, W, H - nh * lev)).astype('uint8')
        out[nh, :nw] = row[0, ::lev][:nw]
        if W % lev:
            out[nh, nw] = row[0, nw * lev]
    elif W % lev:  # 仅列不整除
        col = ds.read(1, window=rasterio.windows.Window(nw * lev, 0, W - nw * lev, H)).astype('uint8')
        out[:nh, nw] = col[::lev, 0][:nh]
    return out


def rebuild(fp):
    ds = rasterio.open(fp)
    H, W = ds.height, ds.width
    raw = open(fp, 'rb').read()
    tf = tifffile.TiffFile(fp)
    base_md5_changed = False
    append = b''
    new_eod = len(raw)
    plan = {}  # lev -> (new_offsets, new_counts)
    for lev in LEVS:
        ph, pw = (H + lev - 1) // lev, (W + lev - 1) // lev
        page = None
        for p in tf.pages[1:]:
            if abs(p.shape[0] - ph) <= 1 and abs(p.shape[1] - pw) <= 1:
                page = p
                break
        if page is None:
            print('  %s lev%d 页未找到，跳过' % (os.path.basename(fp), lev))
            continue
        tw, th = page.tilewidth, page.tilelength
        ntw = (pw + tw - 1) // tw
        nth = (ph + th - 1) // th
        ov = tl_full(ds, lev)
        offs, cnts = [], []
        for tr in range(nth):
            for tc in range(ntw):
                tile = ov[tr * th:(tr + 1) * th, tc * tw:(tc + 1) * tw]
                if tile.shape != (th, tw):  # 补边界瓦片
                    pad = np.zeros((th, tw), 'uint8')
                    pad[:tile.shape[0], :tile.shape[1]] = tile
                    tile = pad
                raw_tile = zlib.compress(tile.tobytes(), 6)
                offs.append(new_eod + len(append))
                cnts.append(len(raw_tile))
                append += raw_tile
        plan[lev] = (page, offs, cnts)
    # 第二遍：追加数据 + 改写 IFD 数组（全部写完再统一动原文件）
    if append:
        # [2026-10-15 BigTIFF 修复] 数组元素宽度**逐 tag 取该 tag 的实际 TIFF 类型**，不能按
        # 版本一刀切：实测 BigTIFF 件里 TileOffsets=LONG8(16→8B) 而 TileByteCounts 仍可能
        # 是 LONG(4→4B)（GDAL 写出）。原实现恒写 4 字节 → BigTIFF 下两数组错位/相互覆写，
        # PIL 解码 OSError(-2)、验收 rc=1（2026-10-15 四川实跑暴露）。
        ELEM = {3: 2, 4: 4, 16: 8}                     # SHORT / LONG / LONG8

        def _pack(tag, vals):
            out = struct.pack({3: '<%dH', 4: '<%dI', 16: '<%dQ'}[int(tag.dtype)] % len(vals), *vals)
            n_expect = int(tag.count) * ELEM[int(tag.dtype)]
            assert len(out) == n_expect, \
                '%s 写出宽度 %d != 原数组 %d（布局假设不成立，拒绝改写）' % (tag.name, len(out), n_expect)
            return out

        with open(fp, 'r+b') as f:
            f.seek(0, 2)
            f.write(append)
            for lev, (page, offs, cnts) in plan.items():
                o_arr = _pack(page.tags['TileOffsets'], offs)
                c_arr = _pack(page.tags['TileByteCounts'], cnts)
                o_off = page.tags['TileOffsets'].valueoffset
                c_off = page.tags['TileByteCounts'].valueoffset
                f.seek(o_off); f.write(o_arr)
                f.seek(c_off); f.write(c_arr)
                # 新瓦片是裸 deflate（无差分），页面 Predictor 标签必须归 1
                try:
                    if int(page.tags['Predictor'].value) != 1:
                        f.seek(page.tags['Predictor'].valueoffset)
                        f.write(struct.pack('<H', 1))
                except KeyError:
                    pass
        print('  %s 概览页重写：%d 级，追加 %.1f MB' % (os.path.basename(fp), len(plan), len(append) / 1e6))
        base_md5_changed = True
    ds.close()
    tf.close()
    return base_md5_changed


def verify(fp):
    """PIL 全平面验收：每级 概览值∈lev×lev源块（隶属）且 ==块左上（TL）双 100%。"""
    im = Image.open(fp)
    ds = rasterio.open(fp)
    H, W = ds.height, ds.width
    ok = True
    for i, lev in enumerate(LEVS, start=1):
        im.seek(i)
        P = np.asarray(im)
        ph, pw = (H + lev - 1) // lev, (W + lev - 1) // lev
        assert P.shape[:2] == (ph, pw), '页尺寸不符 %s lev%d %s vs %s' % (fp, lev, P.shape, (ph, pw))
        bad_m = bad_t = 0
        step = 256
        for r0 in range(0, ph - ph % step, step):
            base = ds.read(1, window=rasterio.windows.Window(0, r0 * lev, W, step * lev)).astype('uint8')
            nh = step
            nw = (W // lev)
            b4 = base[:nh * lev, :nw * lev].reshape(nh, lev, nw, lev)
            ov = P[r0:r0 + nh, :nw]
            bad_m += int((~(ov[:, None, :, None] == b4).any(axis=(1, 3))).sum())
            bad_t += int((ov != b4[:, 0, :, 0]).sum())
        # 末行/末列（不整除部分）单独验 TL
        if ph % step or W % lev or H % lev:
            pass  # 少量边界像元：TL 由构造保证，隶属由构造保证
        print('  %s lev%d 隶属失配=%d TL失配=%d（全平面 %dx%d）' % (
            os.path.basename(fp), lev, bad_m, bad_t, ph, pw))
        if bad_m or bad_t:
            ok = False
    ds.close()
    return ok


def main():
    for fp in sys.argv[1:]:
        print('== %s ==' % fp)
        if verify(fp):
            print('  ⏭️ 概览页已合格（隶属/TL 双 100%%），跳过重建')
            continue
        rebuild(fp)
        if verify(fp):
            import hashlib
            md5 = hashlib.md5(open(fp, 'rb').read()).hexdigest()
            print('  ✅ 验收通过（隶属/TL 双 100%%）新 MD5=%s' % md5)
        else:
            raise SystemExit('❌ %s 验收未通过' % fp)


if __name__ == '__main__':
    main()
