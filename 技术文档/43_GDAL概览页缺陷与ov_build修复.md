# 43. GDAL build_overviews 概览页确定性损坏与 ov_build.py 修复（2026-10-04）

> 状态：**★当前有效**。本发现源于 2023 五省先行批生产的批 1 独立审查（`F:/lc_work/prod5p_2023/review/review_batch1.md`），
> 适用于本机全部 rasterio/GDAL `build_overviews` 产物（含 `试点成果_2023瓦片/`，待回溯）。
> 结论采信链条：独立审查员两轮 PIL 帧解码 + 生产方手动 zlib 解码互相印证。

## 一、缺陷描述

本环境（rasterio 1.4.3 / GDAL 3.x，Windows）中，`rasterio` 的 `DatasetWriter.build_overviews(levels, Resampling.nearest)`
**写出的概览存储页内容与底图确定性不符**：

- T2713（22265×22265，tiled 512，deflate）lev2 概览页 484 瓦片中 384 个受损；
- 典型证据：概览值=1 而对应 2×2 源块全为 3——**任何最近邻现算都不可能产生源块外的值**；
- 坏页 vs 底图 TL（块左上）参照差 14.09%（手动 zlib 解压存储字节裁决）；
- 与调用方式无关：'w' 模式内建与 close 后 'r+' 重建产出**逐字节相同**的坏页（MD5 一致）；
- 与 predictor 无关：predictor=2 与 predictor=1 底图同样受损。

## 二、为什么常规自检发现不了（重要教训）

`ds.read(indexes=1, out_shape=(H//2, W//2), resampling=nearest)` 在读概览级时 **GDAL 在现算下采样、并不读存储页**
（实证：lev4/lev8 现算结果与存储页一致率仅 84–96%）。因此"GDAL 路径自检失配=0"是**无效自证**；
而 PIL（`Image.open(fp).seek(级)`）忠实解码存储字节。**验收概览页必须用 PIL 帧直读。**

## 三、修复工具：`F:/lc_work/prod5p_2023/pilot/ov_build.py`

原理：不再让 GDAL 生成概览内容——底图 **TL（块左上）最近邻**下采样自算 5 级（2/4/8/16/32）→
`zlib` 手动压缩 → 追加文件尾 → **原位改写**各概览 IFD 的 `TileOffsets`/`TileByteCounts` 数组
（概览页 Predictor 归 1；底图字节、调色板、Geo 标签、nodata 全部不动）→ **PIL 全平面验收**：
每级「概览值 ∈ lev×lev 源块（隶属）」与「== 块左上（TL）」双 100%，任一不过即 SystemExit。

用法：`python ov_build.py <tif> [<tif>...]`；验收通过打印新 MD5。
批 1 首验：T2713=`7d887a74ceaa0d1206acc0b1321f3037`、T2102=`fe9c4f3c6bf65f972c7334eaa138b3cd`（5 级双 100%）。

## 四、纪律（新增到交付规范）

1. **一切交付 GeoTIFF 的概览页必须由 ov_build.py 生成并经其 PIL 验收**；禁止直接使用 `build_overviews` 产物。
2. **禁止用 GDAL `out_shape` 读数做概览页自检**（现算路径，无效自证）。
3. 尺寸不整除（如 22265 = 43×512+281）的边界页由构造保证 TL/隶属一致（工具内已处理，验收仍全平面扫描）。

## 五、遗留与影响面

- `试点成果_2023瓦片/` 4 个成品（t_colorize 生成，同款 build_overviews）概览页**疑似同病**（底图无恙），交付前用 ov_build.py 回溯；60m 污染目录除外。
- 批 2/3 全部瓦片由批量组在 colorize 后强制走 ov_build.py（已注入，2026-10-04）。
- 审查员建议：批 2 各瓦片交付前按 r8 同款方法做独立概览抽验（已纳入批次审查清单）。
