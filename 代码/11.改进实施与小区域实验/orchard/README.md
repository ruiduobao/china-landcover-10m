# 代码/11.改进实施与小区域实验/orchard — 园地先验 + 森林筛选·小区域受控分类实验

> 依据：`技术文档/70_样本与分类改进实施方案及20分钟小区域实验规范.md`（v1.0 冻结）
> 工作区：`F:/lc_work/improve_2023_pilot_20261015/orchard/`
> 结果：`<WS>/results/summary.md`（另有 summary.json / paired_by_seed.csv / unit_runs.csv）

## 脚本

| 脚本 | 作用 | 产物 |
|---|---|---|
| `e0_eval_inventory.py` | 五窗 eval 全类计数 + 候选小框枚举（只读、零 RF、不看预测） | `<WS>/explore/e0_eval_class_counts.csv`、`e0_candidate_boxes.csv` |
| `e1_prep_regions.py` | 区域定义、0.25° 空间块 dev/conf 拆分、训练窗剔块、冻结注册 | `<WS>/prereg/regions_frozen.json`、`<WS>/cache/<rid>_{train_window,ref}.parquet` |
| `e2_run_unit.py` | 单元运行器：region×arm×3 种子（baseline / O-CAP）训练→预测→落盘 | `<WS>/results/<rid>_<arm>_{pred.csv,json}` |
| `e3_driver.py` | 限时驱动（每单元子进程硬超时 1200s）+ 配对汇总（块 bootstrap、McNemar） | `<WS>/results/summary.json`、`paired_by_seed.csv`、`unit_runs.csv` |
| `e4_report.py` | 结果表渲染 | `<WS>/results/summary.md` |
| `e5_audit_evidence.py` | 审计证据（训练规模/计时/特征/泄漏/准备期来源） | `<WS>/audit/inbox/orchard_audit_evidence.{md,json}` |
| `qsrc_orchard.py` | 园地质量来源 helper（**仅元数据**：grade/subtype/srcset 分布 + 4 窗可用计数 + 规则建议） | `<WS>/orchard/qsrc/*` |
| `e6_prepare_qarm.py` | 高证据质量臂**注册准备（不跑）**；含无可信本地源时的 BLOCKED 评估 | `<WS>/prereg/qarm_registration.json` |

## 首轮结果（O-CAP vs baseline，4 区×2 臂×3 seed，24 fits）

- 主门（非FCS10 确认层 macro9）：SC1 +0.62 / SC2 +0.71 / GD +0.82 / NX −0.26 pp → SC1/SC2/GD **PROMISING**（组级非劣、点估略正但 CI 含 0），NX **REJECT**。
- **目标类 F1 一律未改善**（类 4 ΔF1 各区为负）→ target_status = NOT_IMPROVED / INSUFFICIENT_SUPPORT。**PROMISING ≠ 已提高**。
- **回归风险**：O-CAP 使园地预测近乎消失（SC1/SC2/GD 全 3 seed 归零；NX 19→1）；确认层园地真值 n≤1，UA/PA 不可评。
- SC1+SC2 联合（描述性，预测后补登记）：n=154，Δm9 +0.65pp，类 4 n=51 ΔF1 −0.0223。
- 结论：cap 臂**未证明精度提升**；下一步应查标签质量，非再扫 cap 阈值。

## 质量臂（H2）状态：BLOCKED

`qarm_registration.json`：川/粤无"可信本地园地源"（grade 高 tier 全为海南橡胶，不可迁移；茶/苹果/柑橘真园率均低且未独立核验）→ 判 **BLOCKED**，未运行。如需推进须新独立证据（重下专题栅格/独立判读点）。

## 运行

```bash
cd F:/lc_work/improve_2023_pilot_20261015/orchard/explore
PYTHONDONTWRITEBYTECODE=1 python e1_prep_regions.py          # 冻结注册（零 RF）
PYTHONDONTWRITEBYTECODE=1 python e3_driver.py                # 8 单元（4 区×2 臂×3 seed）+ 汇总
PYTHONDONTWRITEBYTECODE=1 python e4_report.py                # 渲染 summary.md
```

## 口径（写死在注册 JSON）

- **池**：`train2023_v2.parquet`（cl 已是 v31 24 码，**不再 30→24 remap**）。旧池才需 `to_v31`。
- **臂**：baseline = 训练窗内均匀随机前 20,000（复现生产 `randomColumn` 机制，非逐点复制）；
  O-CAP = 同一 permutation，类 2 园地上限 200（=1%×20k），非园地按同窗同 seed 随机序补足，固定总 N。
- **空间留出（本轮设计）**：训练窗 = 小框所属 2°瓦 ±2°；剔除本区全部 0.25° dev+conf 块；
  位置键零重叠自检通过。**与生产 classify（用全窗点）不同，不可与生产精度混比**。
- **评价**：同点；全点 / FCS10 / 非 FCS10 三层完整报；主门 = 非 FCS10 确认层 macro9 非劣
  （点估≥−1pp 且块 bootstrap 95% 下界≥−2pp）；目标类 F1 同层；class8 真值记错不删。
- **本地 sklearn RF ≠ GEE smileRF**（本地筛选器，迁移须实测）。
