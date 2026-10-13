# grp1_crop 样本质检报告（草本旱地 class24=1 / 灌溉耕地 class24=3）

- 日期：2026-10-13
- 判据版本：`criteria_version = v1.1-2026-10-13T10:00`（预注册，未改动；判据说明见 `reports/land_qc_2026-10-13/预注册判据.md`）
- 组/类：grp1_crop = class24 1 草本旱地（class_new 10）/ 3 灌溉耕地（class_new 12）
- 输入点表：`F:/lc_work/v31_exp/data/land_qc/lists/grp1_crop.parquet`（334,142 点）
- 证据：`feats/crop/land_p*_c*.parquet`（84 块、27 波段，v7_land_fetch.py）
- 产物：
  - `F:/lc_work/v31_exp/data/land_qc/out/land_qc_crop.parquet`（334,142 行逐点三态）
  - `F:/lc_work/v31_exp/data/land_qc/out/stats_crop.json`
  - `F:/lc_work/v31_exp/data/land_qc/qc/quarantine_crop.csv`（6,513 行）
  - `F:/lc_work/v31_exp/reports/land_qc_2026-10-13/spot_crop.json`（v9 独立抽查）

## 1 覆盖与抓取过程

| 项 | 值 |
|---|---|
| 期望点数 | 334,142 |
| 实得（块内/去重） | 334,142 / 334,142 |
| 缺口 | **0** |
| 块数 | 84（6 分片 × 14 块；`ceil(55691/4000)=14`） |
| `_fail_*.json` | 无（失败块记录 0） |
| 合并后无证据点 | 0（v8 报"无证据 0"） |
| point_id 重复 | 无（去重数＝块内数） |

各分片（账户映射按冻结的 `landqc_run.sh` 账户表，重启命令不变）：

| 分片 | 账户 | 点数 | 块 | 完成时刻 |
|---|---|---:|---:|---|
| p0 | ccstqqb | 55,691 | 14/14 | 11:13:38 |
| p1 | ch2fg2t9 | 55,691 | 14/14 | 11:07:47 |
| p2 | chengruiduobao | 55,690 | 14/14 | 11:08:03 |
| p3 | hrbm3pdw | 55,690 | 14/14 | 11:11:03 |
| p4 | ief3nj | 55,690 | 14/14 | 11:11:45 |
| p5 | n08zt4 | 55,690 | 14/14 | 11:11:12 |

过程时间线：
- 09:58:54 `bash landqc_run.sh crop` 启动 6 分片（旧版 4000 点单请求）。
- 10:04 旧版 p0/p2/p3 块 0 各出现 1 次 "Computation timed out" 内部重试（p1/p4/p5 无超时记录）。
- 10:17:31–33 主代理上线 v7 子请求改造（每块内部再切 2000 点子请求，`--sub 2000` 默认生效）并重启全部 6 分片；重启时已有 15 块 / 60,000 点，按幂等规则沿用。重启会截断旧日志，故旧版重试记录以轮询现场观察为准。
- 改造后全程无任何超时、无失败块、无人工补跑；**未 kill 任何 python 进程**。
- 09:58:54 → 11:13:38 全部完成，总墙钟 74 分 44 秒。
- 11:14:32 运行 `python v8_land_qc.py --group crop`；11:15 运行 v9 抽查。

## 2 逐类三态（抄 `out/stats_crop.json`）

| class24 | 名称 | n | keep | quarantine | flag | keep% | quar% | flag% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 草本旱地 | 208,274 | 129,671 | 5,178 | 73,425 | 62.26 | 2.49 | 35.25 |
| 3 | 灌溉耕地 | 125,868 | 104,393 | 1,335 | 20,140 | 82.94 | 1.06 | 16.00 |
| **合计** | | **334,142** | **234,064** | **6,513** | **93,565** | 70.05 | 1.95 | 28.00 |

类码映射：class24 1 → class_new 10；class24 3 → class_new 12。

样本来源（src × class24）：
- class1：`v2_glc_fcs30d2020_7yr_stable` 202,059 + `glc_fcs10_2023_band130` 2,519 + `glc_fcs10_2023` 1,182 + 东北作物专源（maize HLJ 346 / JL 364 / LN 422；soybean HLJ 514 / JL 509 / LN 359）。
- class3：`v2_glc_fcs30d2020_7yr_stable` 121,879 + `glc_fcs10_2023_band130` 1,935 + `glc_fcs10_2023` 711 + rice（HLJ 712 / JL 492 / LN 139）。

## 3 quarantine 理由分布（`qc/quarantine_crop.csv` 按 reason × class24）

| reason | class1 | class3 | 合计 |
|---|---:|---:|---:|
| NOT_CROPLAND | 5,178 | 0 | 5,178 |
| NOT_IRRIGATED_CROPLAND | 0 | 1,335 | 1,335 |
| **合计** | **5,178** | **1,335** | **6,513** |

本组只有 2 种 reason（top5 即全部）。矛盾分支（互斥，无重叠）：

- class1 NOT_CROPLAND（5,178；共同前提 crops<0.10，crops 中位 0.065）：
  `trees>=0.50 且 built<0.50` 2,727（52.7%，trees 中位 0.574）；
  `built>=0.50 且 trees<0.50` 2,413（46.6%，built 中位 0.591）；
  仅 `water>=0.50` 38（0.7%）；`bare>=0.60` 0。
  地理：lon p10–p90 103.9–117.5°E、lat 中位 28.1°N、dem 中位 292 m —— 南方丘陵/华东平原交错带，疑似"林地/建设用地吞并的耕地边界点"。
- class3 NOT_IRRIGATED_CROPLAND（1,335；共同前提 crops<0.10，crops 中位 0.072）：
  `built>=0.40`（不含水/树）1,051（78.7%，built 中位 0.556）；
  仅 `water>=0.50` 143（10.7%）；仅 `trees>=0.50` 141（10.6%）。
  地理：lon p10–p90 105.0–120.1°E、lat 中位 29.8°N、dem 中位 130 m —— 长江中下游平原城镇边缘。

来源上 6,505/6,513（99.9%）来自主源 `v2_glc_fcs30d2020_7yr_stable`（该源占本组 96.96%，其自身 quar 率 2.01%，无来源特异性异常）；fcs10 系仅 8 点。

## 4 flag 机制（本组 flag 93,565，占 28.00%）

- class1 flag 73,425：**90.8% 落在 `crops∈[0.10,0.30)` 的中间带**（keep 需 ≥0.30、quar 需 <0.10 且带矛盾证据）；9.2% 为 `crops<0.10` 但无 trees/built/water/bare 矛盾证据；`crops>=0.30 却不满足 water/built/trees 护栏` 的点为 0。flag crops 四分位 0.141/0.200/0.252。
- class3 flag 20,140：**95.2% 落在 `crops∈[0.10,0.25)` 的中间带**；4.8% 为 `crops<0.10` 无矛盾证据；护栏违例 0。flag crops 四分位 0.151/0.194/0.225。
- 即 flag 几乎完全是 300 m 耕地概率的"中间带"点（keep/quar 阈值之间的灰色区），不是数据缺失或渠道矛盾。

## 5 独立抽查（v9_spot_check，备用账号 vidalmvtpronet18）

```
python v9_spot_check.py --group crop --n 40 --acct vidalmvtpronet18
```

结果：`MISMATCH BANDS: NONE（全部 0.0）`
- 路 A（v7 同构重建）vs 产物：12 个关键波段 max|Δ| 全部 0.0；
- 路 B（独立最小构建）vs 路 A：12 个关键波段 max|Δ| 全部 0.0。
- JSON：`reports/land_qc_2026-10-13/spot_crop.json`（`mismatch_bands = []`）。

## 6 异常观察与备注

1. **quar 率低但非零**：class1 2.49%、class3 1.06%，远低于 flag 率；"硬矛盾"仅 6,513 点（1.95%），分支集中在 trees/built 两类（class1 内占 99.3%、class3 内 built 占 78.7%），全组无 bare 分支点。
2. **flag 率 class1（35.25%）明显高于 class3（16.00%）**：原因是 class1 keep 阈值更高（crops≥0.30 vs 0.25），中间带更宽；两类的 flag 都由概率中间带主导，保留标低置信是设计内行为，不必收紧判据。
3. **DW 概率列 0% 缺测**：`dw_built_150 / dw_water_150 / dw_bare_150` 无 NaN（负数已按 v1.1 设计归一为 0）——本组证据链完整。
4. **S2 NDVI 缺测 0.56%**：`s2_ndvi_xia` NaN 1,882 点（class1 0.62%、class3 0.47%），全部被 `s2_ndvi_dong` 回退覆盖（两者同时 NaN = 0，派生 `ndvi` NaN 率 0.00%）。
5. **GSW 点级覆盖极低，但对本组判决无影响**：`occurrence` 有效值仅 6,040 点（1.81%），`max_extent=1` 仅 6,051 点（1.81%）；class1/3 判据不使用 GSW 波段，故零影响（与 water 组的 flag 机制不同）。
6. **邻域裁剪证据覆盖低**：`m07_150>0` 仅 4.9%、`m01_150>0` 仅 3.3%（判据未用该列，仅存档）。
7. **restricted-mode 警告**：6 个分片日志均出现 "exceeded its noncommercial compute quota ... restricted mode" 警告 13–25 次/分片，但**全部 84 块照常完成、零失败**，本轮无操作影响；下一组跑批前建议仍按惯例做账号项目级健康检查。
8. 无 point_id 重复、无 has_ev=0 点；v7/v8/v9 脚本与判据数值均未作任何修改。

## 7 给主代理的结论（一行）

**grp1_crop 覆盖 334,142/334,142（84 块、缺口 0、零失败块、6 分片全 14/14 完成）；三态 keep 234,064（70.05%）/ quarantine 6,513（1.95%）/ flag 93,565（28.00%）；
剔除由"草本旱地被林地 2,727 + 建设用地 2,413 + 水面 38"与"灌溉耕地被建设用地 1,051 + 水面 143 + 林地 141"构成，flag 主体是 300 m crop 概率中间带（class1 90.8% 在 0.10–0.30、class3 95.2% 在 0.10–0.25）；
v9 抽查（vidalmvtpronet18，n=40）MISMATCH NONE；建议 quarantine 6,513 点从训练视图剔除、flag 93,565 点保留并标 tier=weak，本组无需人工重点复核的异常簇。**
