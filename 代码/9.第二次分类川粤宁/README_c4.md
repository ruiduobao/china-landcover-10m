# 代码/9.第二次分类川粤宁/ — c4：第二次分类（川/粤/宁 40 瓦）分省评估

`c4_eval_second.py` 是第二次分类的**分省评估工具**：评估口径与第一次
`F:/lc_work/prod5p_2023/evalprep/eval_by_prov.py` **逐条一致**（同一留出池、同一归省、同一
miss 口径、同一来源分层、同一指标函数 `v31_common.metrics`），只换两处输入：分类图与训练池排除清单。

## 命令

```bash
# ① 本地采样冒烟自检（现在就能跑；不需要第二次栅格）= 随机点走 sample_local vs 独立仿射读数
python 代码/9.第二次分类川粤宁/c4_eval_second.py --self-test --self-test-n 400 \
  --local F:/lc_work/prod5p_2023/rasters_batch/T1902_10m.tif [--self-test-ref <参考tif,...>]
# ② 正式评估（等 F:/lc_work/prod2nd_2023/rasters_batch/*_10m.tif 就绪；缺哪幅报哪幅）
python 代码/9.第二次分类川粤宁/c4_eval_second.py --prov all --year 2023 --compare
# ③ 只用已有结果出对比表（不重算）
python 代码/9.第二次分类川粤宁/c4_eval_second.py --compare-only --prov 粤
```

`--prov` 收 川/粤/宁(简称或全名，逗号可多个)/all；`--local` 逗号分隔多幅或留空按 plan 自动组盘。
输出（只落此处，不写 prod5p/v31_exp）：`F:/lc_work/prod2nd_2023/reports/eval2/`
`eval2_<省简称>_2023.json/.md`、`compare_run1_run2_<省简称>.md`、`selftest_<省简称>.json`。

## 与第一次的口径对照

| 项 | 第一次（eval_by_prov.py） | 第二次（本脚本） |
|---|---|---|
| 留出点 | `v31_exp/data/yearly/eval_2023_w1..w5.parquet` | 同（默认 w1,w2,w3,w4,w5） |
| 归省/省界 | DataV 省界（GCJ-02）经 p5common | 同（省界集合仍取五省，矩阵与边界标记不漂移） |
| 分类图 | GEE 资产或 `D:/…五省合幅_2023_10m.tif` | `prod2nd_2023/rasters_batch/{tile}_10m.tif`（`--local` 可换） |
| 训练池排除 | `train2023_clean(row_id)` | `train2023_v2`(point_id 键 ∪ (lon,lat) 6 位位置键) **∪** 旧池 `train2023_clean(row_id)`（双保险） |
| 错位清单 | 同文件 | 同 |
| miss 口径 | 类码 0/缺失/>250 计 miss 不计指标 | 同 |
| 分层 | 全点 / fcs10 / 非fcs10 **强制分报** | 同 |
| 指标 | `v31_common.metrics`（24 类 + `to_macro` 9 大类） | 同函数（不重写） |

## 已知口径坑（冒烟实测，引用结果前先读）

1. **`train2023_v2.point_id 不唯一**：1,764,020 行里 121,992 个 id 各对应两个坐标不同（相距可
   达数十度）的点，且这些 id 全在旧池 row_id 空间里 → 单靠 point_id 键会**假命中多扣点**。
   默认 `--pool2-id-key both`（两键都排），报告会显式给出 point_id 独有点数；要对位置键口径，
   加 `--pool2-id-key pos` 重跑。留出点与两池的交叉实测均为 **0**（含 0 显式打印）。
2. **瓦片口径 ≠ 合幅口径**：用同一批 `rasters_batch` 瓦片与交付 `五省合幅_2023_10m.tif` 跑同一点集，
   广东有 **117/2912 点取值不同**（类码 9↔10/1↔9/3↔1，远离瓦片边界，非拼接伪影），OA 差约 0.4 pp；
   宁另有 6 点覆盖差（交付合幅按省界 buffer −0.005° 裁剪）。**Δ 与"瓦片/合幅一致性"必须并列解读**；
   要严格同形对比，等第二次分省或合幅裁剪产物就绪后 `--local` 换成同一形态重跑。
3. 本脚本用 `sys.dont_write_bytecode=True`，只读加载 p5common / v31_common（不写 __pycache__）；
   GEE 用量记账落 `prod2nd_2023/logs/eecu_tooling_2nd.csv`（**不用** `PC.eecu_log`，那会写 prod5p）。
4. `<30 点只报合并级；<2 pp 的单次差异不可信`（AGENTS 硬约束 5/12）——fcs10 系 macroF1 被小 n 类拉低，
   引用以 OA + 逐类表为主。

## 冒烟自检留痕（2026-10-14，用第一次栅格，只读）

| 命令 | 结果 |
|---|---|
| `--self-test --self-test-n 400 --local T1902_10m.tif` | sample_local vs 独立仿射读数 **400/400 一致（1.0000）**；类码全落 24 类表内 |
| `--prov 粤 --local T1902,T1903,T2002,T2003` | n=2911，全点 OA **0.7681** / 非fcs10 0.8240 / fcs10 0.6617（与 evalprep/README 记载一致） |
| `--prov 粤 --local 五省合幅_2023_10m.tif --out-suffix _smoke_mosaic` | 全点 OA **0.7640** / fcs10 0.6497 / 非fcs10 0.8240 —— **与第一次结果 json 逐位一致**（口径保真证据） |
| `--prov 川` / `--prov 宁`（瓦片口径） | 川 n=1406 OA 0.8556；宁 n=1951 OA 0.8088；n_pool 与依赖瓦片清单与第一次结果吻合 |

> 上述产物是**演示/自检件**（用第一次栅格跑出来的），第二次生产栅格就绪后按命令 ② 重跑即为正式结果。
