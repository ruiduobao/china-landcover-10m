# 代码/9.第二次分类川粤宁/ — c6：第二次交付打包（树对齐第一次五省批）

`c6_deliver.py` 把第二次分类（川 24/粤 10/宁 6 = 40 瓦 · 2023）的 c5 省口径成品组装成交付包：
交付树/口径/纪律与第一次 `F:/lc_work/prod5p_2023/delivery_pkg/` 逐条对齐（pipeline_after_mosaic +
make_readme + qa_build + ov_ext_big 的适配版），不重写算法；主 tif 单页 + 外置 .ovr（doc43）。

## 命令

```bash
# 正式打包（第二次产物就绪后；默认 --dest F:/lc_work/prod2nd_2023/delivery）
python 代码/9.第二次分类川粤宁/c6_deliver.py
python 代码/9.第二次分类川粤宁/c6_deliver.py --prov 川,粤,宁 --ovr auto --stats auto

# 冒烟自检（用第一次交付件只读跑通，产物落 <dest>/_smoke/）
python 代码/9.第二次分类川粤宁/c6_deliver.py --smoke --prov 宁 \
  --src-dir D:/生产_2023五省/成品/分省 --ref-area-dir D:/生产_2023五省/成品/面积表
#   --c5-reports-dir .../reports/mosaic2/smoke --c5-label first_smoke --stats reuse  # 验证 c5 面积复用分支
#   --ovr build                                                                      # 验证 ov_ext 重建+全瓦验收
```

产物：`交付包_README.md`、`MANIFEST_MD5.txt`、`台账/生产台账摘要.md`、`成品/{分省,预览,面积表,色卡,QA,评估}`。
关键参数：`--stats {auto,reuse,own}` 面积统计来源（auto=优先复用 c5 面积 QA，缺则 10m 原生格重扫）；
`--ovr {auto,copy,build,none}`（auto=源有 .ovr 则复制+逐级随机瓦片读回抽检，无则 ov_ext 流式重建+全瓦验收）；
`--ref-area-dir` 可选逐类对账（冒烟证据）；幂等可重跑（清 `成品/台账/README/MANIFEST` 重建）。

## 冒烟自检留痕（2026-10-14，输入=第一次宁夏成品，只读；产物 `delivery/_smoke/`）

| 命令/分支 | 结果 |
|---|---|
| 默认（stats auto + ovr auto，附 `--c5-reports-dir …/mosaic2/smoke --c5-label first_smoke`） | **19 件**一套齐；tif MD5 与源一致（`6de1b175e012…`）；.ovr 复制+逐级随机瓦片 zlib 读回抽检 PASS（22.9 MB）；面积复用 c5 QA（`stats_source=c5_qa_json`）且 vs 第一次面积表 **逐类 0 差**（ΣΔ=+0 px） |
| `--stats own`（未命中 c5 QA 时） | 10m 原生格重扫，与第一次面积表仍 **逐类 0 差**（0.2 min） |
| `--ovr build` | ov_ext 流式写出 + 全瓦读回验收：5 级 TL/隶属失配全 0，GDAL 挂载 [2,4,8,16,32]；重建 .ovr 与源件**字节同 MD5**（`7cc5dad7bea8…`） |
| 正式模式（第二次产物未就绪） | rc=1 并逐省报"成品缺失（待生产）"，不写空包 |
| 台账摘要 | 读 `ledger/eecu_ledger_2nd.csv`（505 行/12 账号，上传期账）ΣECU·h=0.00；c3 生产台账运行时自动并入 |
| 完整性 | `MANIFEST_MD5.txt` 19/19 `md5sum -c` OK；预览 PNG 1952×2400（长边≤2400） |

> 上表为 2026-10-14 首跑留痕（当时台账 505 行/上传期账）。`delivery/_smoke/` 现行内容为
> **2026-10-15 重跑件（22 件）**，见下节。

## 正式运行留痕（2026-10-15，第二次 41 瓦产物就绪后收官）

命令（默认参数即正式口径：`prov=all` 川/粤/宁、`stats auto` 复用 c5 面积 QA、`ovr auto` 复制+抽检）：

```bash
python 代码/9.第二次分类川粤宁/c6_deliver.py    # → F:/lc_work/prod2nd_2023/delivery/
```

| 项 | 结果 |
|---|---|
| 交付包规模 | **45 件 / 1.0 GB**（不含 `_smoke/` 冒烟件；MANIFEST 44 件 + 自身）；`md5sum -c` **44/44 OK** |
| 复制保真 | 三份 tif/ovr/clr 与 `mosaic2/` 源件**逐字节 MD5 一致**（川 `c2b0ffbd748f…`、粤 `aad6c2e93f9e…`、宁 `2d1ebc0a48c2…`） |
| 门禁 | 3/3 通过：uint8/单波段/nodata=0/EPSG:4326/10m 原生方度格/256 调色板/**单 IFD 页** |
| .ovr | 复制路线 + 逐级随机瓦片 zlib 读回抽检：5 级 ×2 瓦 **bad=0**，GDAL mount [2,4,8,16,32] |
| 面积/覆盖 | 复用 c5 QA（`stats_source=c5_qa_json`）；**双口径**分报；vs DataV −0.76/−1.43/−1.90%（buffer 口径内） |
| 评估 | 11 件副本（3×eval2 json/md + 3×实跑小结 + 三省汇总 + 拼接对比报告_c5）+ `compare_run1_run2.md` 合并 3 省 |
| 台账摘要 | `eecu_ledger_2nd.csv` 827 行 → **ΣECU·h = 860.08 / 950（90.5%）**；覆盖面 41/41 |
| 预览 | 四川 2400×1772、广东 2400×1686、宁夏 1952×2400（长边 ≤2400） |

正式运行暴露并修复的三处（最小修复；修复后重跑冒烟 22 件复验通过）：

1. **台账重复计**：`ledger/*.csv` glob 混入 `board_2nd.csv`（逐瓦状态板），其 eecu 列与
   `eecu_ledger` 的 `p5t_*` 行同源 → Σ 虚增（860.08 → 1040.22）。修复：汇总排除 `board*`，
   board 仅按路径备注引用。
2. **正式包 MANIFEST 污染**：`list_files` 递归把 `<dest>/_smoke/`（22 件冒烟件）与运行日志
   写进正式清单。修复：`list_files` 排除 `_` 前缀目录/文件。
3. **计数/文案 40→41**：plan 已含补瓦 T2202（闽）= 41 瓦，脚本内 40 的硬文案与覆盖面"期望"值
   全部改为按 plan 动态；README 增补 41 瓦/T2202 说明与「核心评估结论」块（照抄三省汇总 §4.1）。
4. **新增收集**：`{省}_实跑小结.md` + `三省_实跑汇总.md` + `拼接对比实跑报告_c5.md` 收入
   `成品/评估/`（README 核心结论可追溯）；冒烟自检同步验证该分支。

## 硬门槛（写在代码里）

1. 源成品门禁：uint8/单波段/nodata=0/EPSG:4326/10m 原生方度格/内嵌 256 调色板/**主 tif 单 IFD 页**
   （含内建概览页直接中止——c5 须用默认 `--ovr none`，交付级 .ovr 由 c6 外置生成）。
2. 复制件 MD5 必须与源一致才放行；类别全链 nearest；统计=块心纬度加权 `PC.cell_m2`（与 c5 同式）。
3. 覆盖率双口径分报（`coverage_px_pct` / `coverage_km2_pct`）；QA/面积统计来源显式标注（c5 复用 or 重扫）。

## 与第一次交付树对照（缺口声明）

- 本册范围不含**总幅**（`mosaic2/2nd_2023_瓦片合幅_10m.tif` 留在工作区，未入交付包；如需另加）；
- 第一次的 `QGIS.qgz` / 色卡 PNG·HTML·实景图 / `fj_check`（闽无窗专项）/ `分省评估汇总` 未复刻：
  评估汇总角色由 `成品/评估/compare_run1_run2.md`（含 c5 栅格对比附览）承担；其余按需再补。
- 其余逐条对齐：分省 tif(+.ovr/.clr)、预览、面积表、色卡、QA、评估、README、MANIFEST_MD5、台账摘要。
