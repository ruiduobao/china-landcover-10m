# T-SC 四川森林坡向受控分类小实验（地形组）

> 依据：`技术文档/69`（§5.3/§8.2）与 `技术文档/70` v1.0（20 分钟小区域实验规范）。
> 工作区：`F:/lc_work/improve_2023_pilot_20261015/terrain/`（仅新增；原池/生产/交付只读）。
> 状态：**已完成**——主对 12 fits（2 框 × 2 臂 × 3 seed）在父进程硬截止下运行完毕；`t3_aggregate.py`
> 仅重汇总（未训练）给出 doc70 主门表。结论见 [`RESULTS_TSC.md`](RESULTS_TSC.md)：
> **REJECT（依目标未改善，非依已证退化）**。主门 conf∩非FCS10：F1 n=121/5块 ΔOA24 −0.80pp（CI 含 0）、
> Δmacro9 −2.98pp（点估低于组级非劣门）；F2 n=63/3块 点估微正但 CI 不可用。
> 机制改写：地形重采样**改变了有预测力的标签结构、未证实纠错**；真实生态/教师偏差仍不可分。
> F-VIEW 独立对已写好但未跑（`--arms all`）。

## 目标
检验"同每类 count 下，按地形分层限制选点"（T-STRATA）相对生产式随机抽样（baseline）是否改善
四川常绿针叶(6)/常绿阔叶(4) 的坡向条件错误与 F1。**不以地图 OR 下降当精度提升**；无独立针/阔
真值时只作开发诊断。

## 已选框（w2∩川，均在单一 60m DEM 瓦内）
| 框 | 瓦片 | bbox(°E,°N) | 评价n | 森林 | ENF/EBF/DBF | dev/conf | fcs10/非fcs10 | 0.25°块 |
|---|---|---|---|---|---|---|---|---|
| F1 | T1607 | 105.5,32.0,106.5,33.0 | 418 | 189 | 90/66/33 | 266/152 | 86/332 | 15 |
| F2 | T1706 | 107.0,31.0,108.0,32.0 | 206 | 120 | 32/66/22 | 121/85 | 48/158 | 8 |

> F2 的 conf 85 点、F1 的 15 块 <30 块 → 依 doc70 §6.3，CI 仅作该框描述，不作省域泛化。

## 训练候选与防泄漏（基线/候选同规则）
- 训练窗 = 瓦片 ±2°（doc70 §2.1）；**但 DEM 仅川 24 瓦** → 实际候选 = 窗内相邻 DEM 瓦并集。
  F1 候选 55460 点、F2 38677 点；基线各取 ≈20k（cap 绑定）。
- 评价点按 0.25° 空间块**按块哈希排序取前 60% 为 dev**，其余 conf；同块不拆。
- **训练候选落在任一评价块内者全部剔除**（F1 剔 1588、F2 剔 938），基线/候选同规则。

## 臂（doc70 §4.3）
- **主对**：`baseline`（类内均匀随机，每类 count）vs `T-STRATA`（同每类 count；类内按
  aspect3×elevband 分层，各层分配 ∝ 框内地形面积；层内随机；不足按盈余层补）。各 ≈20k。
- **独立对（--arms all，首轮不跑）**：`baseline_common` vs `F-VIEW`（森林每类降到 forest strict
  公共上限、非森林不变；总 N≤20k）。

## 固定配置
- 种子 7/11/23（配对）；RF=sklearn `n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
  n_jobs=6`；特征 A00..A63；每模型 ≤20k；deadline 1200s（自检等效 control/deadline.py）。
- 指标：OA24、macro9、macroF1、逐类 F1（4/5/6/7）、OR(预测/训练标签)、阳/阴坡条件错误、
  0.5° 块 bootstrap OR 区间、多种子均值/极差；全点 + fcs10/非fcs10 + dev/conf 分层。

## 文件
| 文件 | 用途 |
|---|---|
| `t1_prep.py` | 范围/臂/参考/防泄漏准备（无 RF，可复跑） |
| `t2_rf_eval.py` | RF 训练+预测+评价（`--dry-run` 只跑评价管路；`--arms main/all`） |
| `out/t1_registration.json` | 冻结注册（输入身份、范围、臂、块、参考、缺口） |
| `out/t1_frames.csv` | 每框参考支持与计数 |
| `out/t1_arms_<F>_seed<seed>.npz` | 各臂选中点（供 t2 直接训练） |
| `out/t2_metrics_<F>.json` / `t2_report_<F>.md` | 结果（运行后） |
| `out/t2_pred_<F>_<arm>_seed<seed>.csv` | 逐点预测（**非伪连续图**；连续 GeoTIFF 由 geometry 组） |

## 运行（slot 释放后）
```bash
python t2_rf_eval.py --frame F1 --arms main      # 主对，3 seed，≈12 fits
python t2_rf_eval.py --frame F2 --arms main
```

## 参考与缺口（诚实性）
- 主参考 = w2∩川 五窗开发点（FCS10 系教师标签，多轮开发使用）→ **非独立真值**；全点 OA 受
  fcs10 层机械主导，须分层报。
- **无独立针/阔真值**：doc54 A/A2 仅 48 个 w2 森林视觉点且 conf 以低为主、针/阔 AUC 0.724
  → 仅开发诊断。物候标签只分常绿/落叶，不能判针/阔。`records_川.npz` 无 AEF，不可当真值。
- 本地 sklearn RF ≠ GEE smileRF；20k 抽样为统计复现，非 EE `randomColumn` 逐点复现。
