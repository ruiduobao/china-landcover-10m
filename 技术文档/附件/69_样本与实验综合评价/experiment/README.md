# quality_review_2026-10-15 / experiment — 实验效果证据审计（子目录索引）

> 审计日 2026-10-15 ｜ 范围：第一/第二次分类（川/粤/宁 2023）效果证据核查 ｜ 零 GEE、只读既有交付、串行轻量复算。
> 主报告：**[`实验效果证据审计_2026-10-15.md`](实验效果证据审计_2026-10-15.md)**（结论、精确数字、两处勘误、P0–P2）。
> 实验设计（建议判据，未执行）：**[`q7_分阶段实验设计与预算依据.md`](q7_分阶段实验设计与预算依据.md)**。
> 主报告速查件：**[`三省_面积与配对速查.md`](三省_面积与配对速查.md)**、[`审计速查.json`](审计速查.json)。

## 脚本（按序执行；全部只读输入、写点只在本目录）

| 脚本 | 作用 | 主要产物 |
|---|---|---|
| `q1_extract_metrics.py` | 三省 24 类/macro9 OA 精确值 + 逐类 F1/面积机械提取 | `q1_三省指标精确表.{csv,md}`、`q1_逐类F1对照{,_fcs10系,_非fcs10系}.csv`、`q1_面积结构对照.csv`、`q1_口径说明.json` |
| `q2_windows_check.py` | 五窗/泄漏三键/类 8/来源分层独立复验（import c4 同源函数） | `q2_windows_leakage.json`、`q2_province_truth_classes.csv`、`q2_windows_bounds.csv` |
| `q3_transitions.py` | 全通过矩阵复算：全域/核心区/S 区一致率、不涉 S 变化、转移对 | `q3_栅格口径核对.json`、`q3_最大转移对.csv`、`q3_转移汇总.csv` |
| `q4_pool_delta.py` | 新旧池逐类点数/来源构成 + **同点位置键重叠**（核"森林=同源筛选"之说） | `q4_池类目对照.csv`、`q4_池来源构成.json`、`q4_池点位重叠.csv` |
| `q5_tile_qa_compare.py` | 22 瓦 QA 类构成 1:1 对照 + 川四瓦对账 | `q5_瓦片QA类构成对照.csv`、`q5_瓦片QA对照核对.json` |
| `q6_ning_plus6.py` | 宁 n 1945→1951 固定共同有效集分解（本地取 1964 点，非重扫） | `q6_宁_plus6分解.json`、`q6_宁_共点预测缓存.csv` |
| `q8_converge.py` | 装配速查件（无新计算） | `三省_面积与配对速查.md`、`审计速查.json` |

复现（串行，约分钟级；建议加 `PYTHONDONTWRITEBYTECODE=1` 以免 import c4 时在被引目录落 `__pycache__`）：

```bash
cd F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/experiment
PYTHONDONTWRITEBYTECODE=1 python q1_extract_metrics.py && PYTHONDONTWRITEBYTECODE=1 python q2_windows_check.py && \
PYTHONDONTWRITEBYTECODE=1 python q3_transitions.py && PYTHONDONTWRITEBYTECODE=1 python q4_pool_delta.py && \
PYTHONDONTWRITEBYTECODE=1 python q5_tile_qa_compare.py && PYTHONDONTWRITEBYTECODE=1 python q6_ning_plus6.py && \
PYTHONDONTWRITEBYTECODE=1 python q8_converge.py
```

## 三条必须随数字携带的口径

1. **任务定义**：24 类（含混交 8）与 23 类（第二次缺 8）是两个任务；重定义口径不得与原 OA 直接比（主报告 §6.1）。
2. **分层**：全点 OA 受 fcs10 灌丛同源点机械主导，M3 判读前不得作发布指标；只认非fcs10 净窗口 + 明确标注。
3. **形态与窗口**：第一次=交付合幅、第二次=未裁剪瓦片（非fcs10 层形态效应实测 0.0000pp）；五窗为固定地理盒（川=w2、粤=w3、宁=w4+w2），非 25km 块、非省域随机。

## 纪律声明

未修改训练池/交付件/生产配置；未发 GEE 任务；未取消他人任务；不更新技术文档索引（doc68 两处勘误文本在本目录主报告 §6，供 doc69 采用）。
