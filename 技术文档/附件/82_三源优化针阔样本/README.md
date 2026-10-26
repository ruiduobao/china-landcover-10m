# 附件：82 三源外部数据优化针阔样本 · 证据清单

> 对应正文：[`../../82_三源外部数据优化针阔样本结果.md`](../../82_三源外部数据优化针阔样本结果.md)
> 归档时间：2026-10-24。原始工作区：`F:/lc_work/nb_national_2026/opt/`（本目录为轻量证据副本，**不含 parquet 与栅格**）。

## 一、文件

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `opt_v1_stats.json` | `opt/opt_v1_stats.json` | **融合样本总账**：248,828 点、A/B/C/D 分级、tier×leaf、12 种 reason、265 瓦/34 省 |
| `opt_v1_by_prov.csv` / `by_tile.csv` / `by_form.csv` | 同左 | 分省/分瓦/分群系的 tier 构成 |
| `veg_qa.json` | `opt/veg_qa.json` | **植被图诊断**：V2=0.5753、判别效度（青藏 +37.9pp … 华南 −3.4pp）、南方马尾松/杉木图斑数字、4 个附加判别段 |
| `veg_qa_by_prov.csv` / `veg_qa_by_type.csv` | 同左 | 分省（35 行）/ 逐植被类型（551 行） |
| `pft_qa.json` | `opt/pft_qa.json` | **PFT 诊断**：`decade_test`（2013 年针叶图斑内 65.8%）、`south_pine_fir`（马尾松 66.7→67.3→53.2%，广东 −26.4pp）、`V3`=0.6476、`stability_filter` |
| `pft_yearly_curve.csv` | 同左 | 33 年 × 8 类 PFT 面积曲线 |
| `r1_30m_qa.json` | `opt/r1_30m_qa.json` | **30m 诊断**：R2=0.5768、`generation_test`（同源五项证据）、`three_way`、`south` |
| `r1_code_map.csv` | 同左 | **863 个码**的映射与歧义裁决（含 `建议处置`） |
| `r1_by_tile.csv` | 同左 | 313 瓦的 30m 像元统计（含 cos²φ 加权 km²） |
| `r4_audit.json` | `opt/r4_audit.json` | **影像核查（判据已证无效）**：G1–G5 分组、阈值敏感性、判别力 AUC 0.38–0.56、三根因 |
| `code/*.py`（6 个） | `opt/code/` | q1/q3/r1/r2/r4/r5 全部脚本，可复现 |

**未归档**：`needle_broad_opt_v1.parquet`（248,828 行）、`needle_broad_veg30_v1.parquet`（138,175 行）、`r1_points.parquet`、`veg30/`（30m 栅格 1.18 GB）—— 可用脚本重建。

## 二、复核四条最快路径

1. **核规模与分级**：`opt_v1_stats.json` → `n_total=248828`、`by_tier={A:130721,B:30077,C:51907,D:36123}`、`tiles=265`。
2. **核"30m 与植被图同源"**：`r1_30m_qa.json` → `generation_test` 的五项（码重合 0.9572 / 词表 0.7387 / 嵌套 0.823 / 点级类型组 0.9089 / 现代引种树种均 0）；再用 `r1_30m_qa.json` 的 `vs_vegmap` 看一致率 0.9837。
3. **核 FCS10 南方失效**：`veg_qa.json` → 判别效度各生态区差值（华南 −3.4pp）与"非森林区 26.0% > 针叶图斑内 19.3%"。
4. **核年代边界**：`pft_qa.json` → `decade_test.by_region`（青藏 89.8 … 华南 36.6）与 `south_pine_fir`（广东 60.2→33.8，转换图斑 17.6%）。

## 三、必须知道的口径警告

1. **"三源"实为两源**：30m 是 100 万植被图的栅格化细分版（0.9837）。任何报告**不得**把"30m + 植被图都说是针叶"当作两张独立票。
2. **V4 判据无效，不得引用其分组针叶率（71.9/67.6/44.4/62.5%）作为真值**——对照组 59.1% 与之无法区分，北方对照组 0%。`r4_audit.json` 里这些数字只作留档。
3. **PFT 只到 2013**，且该数据自身有质量缺陷：全国森林覆盖 2018 见顶（239 万 km²）后 2023 回落 10.9%，与九清持续增长矛盾 → **所有"2023 vs 2013"对比只作参考，2013 及以前段才是可靠证据**。
4. **DNF（落叶针叶）的 PFT 变化产品事件像元为 0**，其余 7 类非零 → 针叶变化事件实际只由 ENF_boreal + ENF_temperate 代表。
5. **30m 栅格面积不要用 web-mercator 名义值**：1 有效像元 = 900 m² 名义，真实面积需按 cos²φ 折算（R1 已分报两口径）。

## 四、质量裁决（R6，2026-10-25，**最重要**）

| 本目录文件 | 原始路径 | 内容 |
|---|---|---|
| `r6_retrain.json` | `opt/r6_retrain.json` | 四臂全指标（5 折 sd）、跨臂互评、McNemar、配对检验、诊断 |
| `r6_verdict.md` | `opt/r6_verdict.md` | 裁决正文 |
| `code/r6_retrain.py` / `r6_aef_fetch.py` | `opt/code/` | 重训与 AEF 取样脚本 |

**结论：数量与分布提升，质量显著下降。**

| 臂 | n | macro-F1 |
|---|---:|---:|
| **ARM_V2** | 109,260 | **0.8701 ± 0.0019** |
| ARM_A | 130,721 | 0.7876 ± 0.0042 |
| ARM_MIX | 200,000 | 0.7675 ± 0.0028 |
| ARM_AB | 160,798 | 0.7569 ± 0.0063 |

Δ(AB−V2) = **−0.1131**，配对 **t=−34.17，p=4.38e-6**；第三方 CGLS 裁判 kappa **0.4524 vs 0.2131**。

**→ 下游继续用 ARM_V2 训练；`opt_v1` 只作分歧地图与人工判读优先队列，不作训练数据。**

## 五、复核指引（新增 3 条）

5. **核质量结论**：`r6_retrain.json` → `paired_tests_AB_vs_V2.macro_f1`（mean_diff −0.1131 / t −34.17 / p 4.38e-6）；`arms.{V2,AB}.folds[*].macro_f1` 自行算均值 sd。
6. **核第三方裁判**：`opt/r6_arbiter2.json`（本目录未归档，见原工作区）→ AB 主场块上两模型 kappa。
7. **核"新点更差"**：`r6_retrain.json` → `diagnostics` 里的按来源分层 OA（v2 来源 0.8455 vs veg30 0.7263）。
