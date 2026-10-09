# MANIFEST — code_sep/（2026-10-09 三类可分性实验，doc 54）

本目录是 `F:\lc_work\v31_exp\code\` 中本轮实验脚本的存档副本（工作副本仍在 v31_exp）。

| 脚本 | 职责 |
|---|---|
| `n11_exp_features.py` | 统一特征取样：AEF64 + S1 冬夏 + S2 四季(SCL 掩膜) + DEM/坡度；`--set A/B/C/A2/C2/C3/C4` |
| `n12_merge_judges.py` | 合并子代理判读 CSV → `exp*_judged.csv`（按 point_id 去重、覆盖自检、并入特征） |
| `n12_probe_lib.py` | 可分性探针公共库（5 折分层 CV、RF300 balanced_subsample、AUC/最佳 F1） |
| `n13_expA_probe.py` | 实验 A 第一轮探针（目视林型 T1/T2 + 物候交叉核验 A3 + 物候探针 A2） |
| `n14_expB_probe.py` | 实验 B 探针（湿润区真灌丛率 + 灌丛二分可分性） |
| `n15_expC_probe.py` | 实验 C 第一轮探针（湿地三关键对 + 水体/水田附加对） |
| `n16_expA2_points.py` | 实验 A 第二轮点表（FCS10 森林层分层抽样 188 点） |
| `n17_expA2_probe.py` | 实验 A 第二轮探针（合并两轮：T2a/T2b 针阔 + T1p 物候常绿/落叶 + 终判） |
| `n18_expC2_points.py` | 实验 C 第二批点表（三江 FCS10 湿地/裸地/森林层补样 108 点） |
| `n19_expC2_probe.py` | 实验 C 终版探针（C+C2+C3(+C4) 合并，三关键对 + conf 子集复测） |
| `n20_expC3_points.py` | 实验 C 第三批点表（WorldCover 树层补样 40 点） |
| `n21_expC4_points.py` | 实验 C 第四批点表（本项目 2023 黑龙江交付成品森林像元候选 40 点） |
| `m5k_zoom.py` | 判读材料：140 m 窗 ×3 放大 + 4 点联系表 + sheet_map.json |
| `m5l_box_crop.py` | 判读材料：10 m 青框邻域 120 px 窗 ×8 + 4 点拼图（box4_sheets；注意按数值排序，修复过 sheet_100 字典序 bug） |

**判据预注册**：n13/n14/n15/n17/n19 的文件头写死了判据（先写死再跑）；n17/n13 中的"物候标签/交叉核验"为补充分析，均单独标注。

**运行顺序**（可复现）：
```bash
python m5j_m3b_imagery.py fetch --in data/m3/expA_forest_points.csv --outdir data/m3/expA_imagery --workers 6
python m5k_zoom.py --outdir data/m3/expA_imagery
python m5l_box_crop.py --outdir data/m3/expA_imagery
# 子代理按 judge_rubric_A.md 判读 → data/m3/expA_judge/judge_A_*.csv
python n11_exp_features.py --set A
python n12_merge_judges.py --set A
python n13_expA_probe.py
```
