# F-VIEW G2（独立新框）实验结果 — 2026-10-15

> 规范：[技术文档 70](技术文档/70_样本与分类改进实施方案及20分钟小区域实验规范.md) §4.3 F-VIEW / §6 / §7
> 工作区：`F:/lc_work/improve_2023_pilot_20261015/terrain/`；结果目录 `terrain/out/fview_g2/`
> 运行：`control/run_experiment.py --spec control/specs/FVIEW_G2_strict_common.json`（父级硬 1200s，只杀自身子树）
> 状态：**DONE**（`control/run_results/FVIEW_G2_strict_common.json`，budget_ok=true）

## 1. 实验单元

| 项 | 值 |
|---|---|
| 区域 | G2 = T1606 小框 106–107°E / 31–32°N（F1/F2 以外的新框） |
| 训练窗 | 瓦片 ±2° = 103–109°E / 28–34°N |
| 参考 | `eval_2023_w2` ∩ 四川，n=194（dev 119 / conf 75；8 块：dev 5 / conf 3）；非FCS10 143、FCS10 51 |
| 臂 | `baseline_common` vs `strict_common`（共同支持 count；**非森林同点**） |
| 种子 | 7 / 11 / 23（配对） |
| RF | 100 树 / minLeaf=2 / maxNodes=5000 / AEF 64 维 / n_jobs=10（本地 sklearn） |
| 主门 | conf ∩ 非FCS10，目标类 4 F1（n<30 支持不足）；组级非劣 Δmacro9 点估 ≥ −0.01 且 CI 下界 ≥ −0.02 |

## 2. 计时（实测，阶段分列）

| 阶段 | 秒 | 说明 |
|---|---|---|
| prepare | 7.125 | 重建两臂 + ID/块独立性 + 臂自检（独立计时，不并入窗口） |
| train | 5.36 | **6 fits**，逐 fit 0.57–0.61s，`n=17697 shape=(17697,64) 类=17` |
| infer | 1.765 | 逐点预测（6 次 predict） |
| eval | 4.829 | 分层指标 + 配对块 bootstrap |
| **deadline_window_s** | **11.954** | train→eval 窗口，`limit_s=1200` → budget_ok=true |

## 3. 臂完整性与独立性（`t4_fview_preflight_G2v2.json`）

- **v1 缺陷（已修）**：`t4_fview_prep.py` 组装 strict 臂非森林行时写 `X[pool]`（pool 为窗内相对下标、X 为全局池矩阵）→ 取错行，违反"非森林两臂完全同选择"。v1 npz 标 `INVALID_STRICT_ARM_KEPT_AS_HISTORY`，保留不覆盖、不用于训练。
- **v2 修复 + 强化**：`X[iw][pool]`；并新增硬断言——每类 count 两臂相同；非森林**同点号且 X 数值逐位一致**；森林全部来自 strict 视图；特征全 finite；索引按来源唯一；索引落在训练窗内。三 seed 全 PASS。
- **strict 候选与池同过滤规则**（不绕过过滤新增漏点）：窗内 ∩ 排除本框 eval 块（67 点）∩ 排除已看 conf 块（474 点）∩ 有 DEM 支持（剔 882 点）→ 可用 c4 1721→**1303**、c6 2269→**1729**、c7 3。
- **strict 只做质量成员筛选 + 年份对齐**：`forest_samples_v1_strict.parquet`，**只取 aef_year==2023**（剔非 2023 共 3313，保留年份集合 {2023}，已 assert），非八年并集。
- **参考独立性**：conf 3 块 ∩ 已看 69 块 = ∅；conf 75 点 row_id/位置键 ∩ 已看（orchard SC1/SC2/GD/NX + terrain F1/F2，809 点）= **0** → `UNSEEN_CONFIRMED`。
- **共同支持 count**：`n_train_common = 17697`（三 seed 相同），森林降幅 c4 2329→1303、c6 2996→1729、c7 12→3（c5 2279 不变，非森林不变）。
  - 与 v1 注册件 18588 / audit 记录 18556 的差异：v1 对 strict 候选**未**施加与池相同的过滤（只排 eval 块），本轮加排已看 conf 块 + DEM 支持域 → strict 可用森林下降、共同支持 count 随之下降。属预注册的过滤口径修正，非静默改动。

## 4. 结果（开发口径）

主层 = conf ∩ 非FCS10（n=56，**3 块 <4 → bootstrap CI 不可用**，仅框内描述）

| 指标 | baseline_common | strict_common | Δ |
|---|---|---|---|
| OA24 | 0.7262 | 0.5833 | **−0.1429** |
| macro9 | 0.9048 | 0.7381 | **−0.1667** |
| 类 4 F1（n=15） | 0.8928 | 0.7407 | **−0.1521** |
| 类 6 F1（n=21） | 0.8108 | 0.6583 | −0.1525 |

其它层（3 seed 均值 Δ）：

| 层 | n | 块 | ΔOA24 | Δmacro9 | Δc4F1 | Δc6F1 |
|---|---|---|---|---|---|---|
| conf | 75 | 3 | −0.1067 | −0.1245 | −0.1584 | −0.1588 |
| dev | 119 | 5 | −0.0785 | −0.0925 | −0.3038 | −0.0702 |
| 非FCS10_all | 143 | 8 | −0.1212 | −0.1445 | −0.2180 | −0.1002 |
| FCS10 | 51 | 8 | 0.0 | +0.0065 | —（n=0） | —（n=0） |
| 全点 | 194 | 8 | −0.0893 | −0.1048 | −0.2209 | −0.1031 |

dev 层有 5 块，配对块 bootstrap 95%CI 均不含 0（如 Δmacro9 −8.4 [−11.81, −4.5]、ΔOA24 −7.56 [−13.59, −0.9]）→ 负向在可算 CI 的层上稳健。

## 5. 结论

- **采纳门：REJECT**。目标类 4 参考 n=15 <30 → `INSUFFICIENT_SUPPORT`（不作类级达标声明）；同时组级非劣门不过（主层 Δmacro9 = −0.1667 < −0.01，全点 −0.1048），方向与目标类一致向下。
- 主层新退化类判定：参考 n 均 <30，无法判类级退化（0 个"n≥30 且 ΔF1≤−0.05"）。
- **机制旁证**（`diag_strict_train_src_G2.json`，仅组分统计、非新臂）：窗口内 strict 森林训练点 8428 中 **8220（97.5%）来自 `v2_glc_fcs30d2020_7yr_stable`**，仅 233 来自 `glc_fcs10_2023`。即 strict 视图不是"同标签更干净"，而是**标签来源置换**（FCS30D 2020 七年稳定主导），与参考（FCS10 系教师标签）惯例不一致——方向与负结果自洽。
- **限制**：本对只改变森林行身份（count 相同），无法在同一单元内分解"质量 vs 来源置换 vs 点数下降"；参考为多轮使用的教师标签开发集，非独立人工真值；本地 sklearn RF ≠ GEE smileRF。结论限于本地筛选，不外推省域。

## 6. 文件

| 文件（`terrain/out/fview_g2/`） | 内容 |
|---|---|
| `t4_fview_registration_G2v2.json` | 预注册（含 v1→v2 bug 修复声明、输入 MD5、counts、逐 seed 自检） |
| `t4_fview_preflight_G2v2.json` | 过滤镜像/年份/计数口径/断言汇总 |
| `t4_fview_idcheck_G2v2.json` | 块级 + 点级（row_id/位置键）独立性证据 |
| `t4_fview_arms_G2_seed{7,11,23}.npz` | 两臂训练矩阵 + `bc_idx/sc_idx/sc_src` 行号留痕 + 评价点 |
| `train_manifest.json` | 6 fits 逐次 fit 秒数/shape/类数/trainIDs sha16 |
| `pred_G2_<arm>_seed<s>.csv` | 原始逐点预测（row_id/lon/lat/true/pred/is_fcs10/split/block） |
| `preds_G2.npz` | 预测与评价元数据（供 eval 精确复用） |
| `t5_metrics_G2.json` / `t5_report_G2.md` / `t5_tables_G2.csv` | 指标、报告、逐层逐类完整表 |
| `diag_strict_train_src_G2.json` | strict 训练点来源×类组分（机制旁证） |

代码归档：`代码/11.改进实施与小区域实验/terrain/{t4b_fview_prep_g2.py, t5_fview_g2.py, FVIEW_G2_strict_common.json}`。
运行日志：`F:/lc_work/improve_2023_pilot_20261015/control/logs/FVIEW_G2_strict_common/`。
