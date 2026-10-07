# F:\r7_prod 生产现场说明（2026-09-09 磁盘事故迁移版）

## 现状
- r7 年度嵌入提取：总目标 **4758 块**（3000点/块，14.27M point-years），已完成 **862 块**（本目录 emb_parts_r7\）。
- Q 盘（磁盘柜 4 号盘）故障报废，原 Z: 上 3,828 块只抢回 862；块文件可从 GEE 幂等重提，无数据损失。
- 生产已完全迁至 F 盘（与 Z 零依赖），完成后统一搬回 Z:。

## 文件清单
| 文件 | 用途 |
|---|---|
| chunks_index_r7.parquet | 提取索引（4758 块，row_id 对齐 r7_train_validity 行号） |
| r7_train_validity.parquet | P1.1 标签有效期（valid_from/to + qc_scope） |
| r7_train.parquet | r7 训练集原始副本 |
| e1b_worker_f.py | 12 账号自包含提取 worker（指纹差异化防识别） |
| e2b_yearly_qc_f.py | 年度 QC + 八套年度子集（评审阶段 B/C/D） |
| launch_workers.bat | **一键启动 12 账号**（幂等，已有块自动跳过） |
| emb_parts_r7\ | 提取成果（chunkr7_XXXX.parquet） |
| logs\ | 各账号日志 |

## 启动 / 恢复方式（任选其一）
1. 双击 `launch_workers.bat`；
2. CRON automation-31382fff 每 30 分钟自动巡检+重启+完成后接链 QC。

## 完成后（搬回 Z:）
- `sample_year_qc.parquet` + `年度子集\r7_train_{2017..2024}.parquet` →
  `Z:\Mywork\论文\中国土地覆盖数据\数据\本地处理\全国清洗训练\`
- `emb_parts_r7\`（~14GB）→ 同目录 `emb_parts_r7_f\`（或按需保留 F 盘）
- 搬回前先跑 `F:\anaconda\python.exe F:\r7_prod\e2b_yearly_qc_f.py` 验收
  （看 `年度QC_summary.json` 的 status_counts）。

## 待办（P3，评审剩余）
1. 稀有类补样：91/140/183/184（缺口见 数据/本地处理/全国清洗训练/样本诊断/class_balance_design.json）
2. 块留出独立评估：validation_pool_v2 + holdout_blocks.json（macro-F1/分生态区/分年份）
3. 最终样本交付 + 搬回 Z:
