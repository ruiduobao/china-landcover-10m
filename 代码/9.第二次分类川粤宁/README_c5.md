# 代码/9.第二次分类川粤宁/ — c5：第二次成品拼接 + 与第一次栅格级对比

`c5_mosaic_compare.py` 是第二次分类（川 24/粤 10/宁 6 = 40 瓦）的**成品拼接与栅格级对比工具**：
拼接口径与第一次 `F:/lc_work/prod5p_2023/mosaic/{mosaic.py,clip_prov.py,p5common.py}` 逐条一致（10m 原生
方度格、全局同锚零重采样同格直拷、重叠带"先布放者胜"、DataV 省界 GCJ-02 原样 + `buffer -0.005°`）。
**不做点留出评估**（那是 c4 的职责）；只做栅格像元口径的拼接 / 面积 / 转移矩阵 / 一致率。

## 命令

```bash
# ① 拼接（第二次栅格就绪后）：40 瓦 → 总幅 + 逐省省口径成品（四川/广东/宁夏）
python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py mosaic --prov all
# ② 栅格级对比：共同域逐类像元/面积 + 24×24 转移矩阵 + 采样一致率
python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py compare --prov all
# ③ 逐省面积/覆盖（双口径）+ 与第一次面积表/plan/DataV 对账
python 代码/9.第二次分类川粤宁/c5_mosaic_compare.py qa --prov all
# 辅助：--dry-frame 只算框架；--no-clip 只出总幅；--prov-mosaics 另出省瓦片口径；
#       --sample-only 采样快路径；--ovr {none,scaffold,ovbuild}（ovbuild=脚手架+ov_build 重建+PIL 验收）
```

输出只落 **`F:/lc_work/prod2nd_2023/mosaic2/`**（tif + .clr + _plan/_qa.json + preview）与
**`F:/lc_work/prod2nd_2023/reports/mosaic2/`**（`trans_<省>_24x24.csv`、`compare_<省>_class.csv`、
`compare_<省>_run1_run2.json/.md`、`compare_all.md`、`qa_<省>_<label>.md/.csv/.json`）。
第一次参照（`D:/生产_2023五省/成品/`、`F:/lc_work/prod5p_2023/`）**一律只读**。

## 冒烟自检留痕（2026-10-14，全部用第一次既有产物只读跑通；证据在 `mosaic2/smoke/`、`reports/mosaic2/smoke/`）

| 命令 | 结果 |
|---|---|
| `mosaic --prov all --tiles-dir <第一次瓦片> --dry-frame` | 40/40 瓦校验+布放，相位 0（全局格）→ 框架 222640×222640 |
| `mosaic --prov 川 --tiles T1304,T1404 --out-dir …/mosaic2/smoke` | 合并 9.90 亿 px/0.6 min；**回读一致性双瓦 PASS**；裁剪 川 35,897 km²（部分覆盖警告正确触发） |
| `compare --prov 川 --second-file <smoke川> --first-file <同一文件>` | 身份自检：ΣΔ=0，一致率 **1.00000**，转移矩阵纯对角 |
| `compare --prov 川 --second-file <smoke川> --first-file D:/…/四川_2023_10m.tif` | 跨幅冒烟：共同域 4.05 亿 px，全通过一致率 **1.00000**（采样 n=202,058 亦 1.0）——本工具拼/裁的局部幅与第一次交付**逐像元一致** |
| `compare … --sample-only` | 采样一致率 1.0（n=97,078） |
| `qa --prov 宁 --src-dir D:/…/分省 --label first_smoke` | 与第一次面积表**逐类 0 差**（50,984.9 km²，15 类；DataV −1.90% 属文档化口径带） |
| `mosaic --tiles T1304 --ovr ovbuild` | 脚手架→ov_build 重建→PIL 验收：合幅/裁剪件 5 级**隶属/TL 双 100%** |
| 默认路径（第二次产物未就绪） | `compare/qa` 逐省明确报"第二次成品缺失（先跑 mosaic）"，不写空结果 |

## 已知口径坑（引用结果前先读）

1. **第一次裁剪的真实口径是 GCJ-02 多边形原样套 WGS84 栅格**（clip_prov.py 未做 gcj2wgs，评估归省同）；
   本脚本默认 `--gcj raw` 复刻该口径以保可比，`--gcj gcj2wgs` 会改变足迹、不可与第一次逐像元比。
2. **相邻瓦片重叠 1 px**（各瓦片 GEE 网格实际跨度 >2°）：重叠带规则=先布放者胜。**回读一致性校验必须
   按"重放先布放者胜"重建期望值**——按单瓦窗口比对会在重叠带假 FAIL（第一次 mosaic.py 靠随机窗概率侥幸通过）。
3. **布放取整用 round**：瓦片相对框架的残差 ~1e-7 px，`floor+eps` 会让整幅错 1 px（dry-frame 冒烟实测）。
4. **compare 的列映射符号**：A 列 c → B 列 `floor(c − c_off + 0.5)`（c_off=(B.left−A.left)/px）。身份测试
   （Δ=0）查不出符号错——跨幅冒烟才暴露；现与第一次交付 1.00000 吻合。
5. **命名**：第一次交付用"交付短名"（`四川_2023_10m.tif`/`宁夏_…`），不是 DataV 全名（四川省/宁夏回族
   自治区）。本脚本输出与参照探测均按交付短名（`DELIV_NAME`）。
6. **面积公式**：与第一次 mosaic/clip/qa_build 同式 `p5common.cell_m2`（块心纬度加权）。`prod2_driver.qa`
   的 `111.32/110.57` 近似式是**瓦片 QA 口径**，两者不可混用（本工具与交付面积表对账实测 0 差）。
7. **`--sample-only` 未必更快**：200k 点 geo 直查实测 ~100 s，全通过扫描 4e8 px 仅 84 s——快路径只适合
   "不想扫全图"的场合；面积/转移矩阵以全通过为准。
8. **概览纪律（doc43）**：默认不写内建概览；交付件必须 `python F:/lc_work/prod2nd_2023/run/ov_build.py <tif>`
   （`--ovr ovbuild` 可自动串起来）。本环境 rasterio 读窗的 numpy DeprecationWarning 无害，已单条静音。
9. 用 `--tiles` 子集拼接时逐省给"[警告] 部分覆盖"，面积/构成不可当交付值——第二次产物就绪后
   去掉 `--tiles` 全量重跑即可（`mosaic2/smoke/` 下的样件届时不要混入正式对比）。
10. **`--ovr ovbuild` 在 BigTIFF 幅面上曾失效（2026-10-15 实跑暴露，已修）**：省口径成品多为
   BigTIFF（如川 124446×91908 ⇒ 11.4 Gpx ⇒ `BIGTIFF='IF_SAFER'` 立 BigTIFF），而
   `ov_build.py` 原实现把 TileOffsets/TileByteCounts 恒按 4 字节写回——BigTIFF 下
   TileOffsets=LONG8(8B)、TileByteCounts 可能是 LONG(4B)，两数组错位/相互覆写 ⇒ PIL 解码
   `OSError(-2)`、ov_build 验收 rc=1，且 `run_ovbuild` 的 SystemExit 会**中断整条 mosaic 命令**
   （已产出的省口径件不受影响，未裁剪的省不再继续）。修复：`ov_build.py` 按 tag 实际 TIFF
   类型取宽度（工作区 run/ 与 `ov_build_副本.py` 同步，MD5 `69ab0f3a…`，回归证据
   `reports/mosaic2/rerun/regress.json`）。**大幅/交付件的概览仍按第一次惯例走外置 `.ovr`**
   （`F:/lc_work/prod5p_2023/delivery_pkg/ov_ext_big.py`，流式 + 读回隶属/TL 全平面验收），
   `--ovr ovbuild` 仅适合小件（单瓦级）；总幅（222640²）概览重建为小时级，默认不开。
11. **`mosaic --prov all` 等价于"不过滤省"（2026-10-15 v2 补瓦实测）**：`cmd_mosaic` 里
   `--prov all` 解析结果是 `provs=None`（与 `compare/qa` 走 `resolve_provs` 返回 川/粤/宁 列表
   **不同**）。想并入 plan 主省之外的第 41 瓦（T2202，主省=闽）而仍只裁川/粤/宁，必须写
   `--prov 川,粤,宁 --tiles <全部瓦>`；若写 `--prov all --tiles <全部瓦>`，裁剪段会把 闽 也裁出
   一份单瓦条带（本次已移出正式命名空间）。另一坑：**裁剪/重拼（`w` 重建主图）会静默删掉同名
   `.ovr`**（GDAL 清理旧外置概览）——重裁后必须重生成 `.ovr` 并重验收。
