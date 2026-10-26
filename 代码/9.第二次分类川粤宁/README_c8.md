# 代码/9.第二次分类川粤宁/ — c8：生产收官摘要生成器

`c8_batch_summary.py` 把第二次分类（川 24/粤 10/宁 6 = 40 瓦 · 2023）驱动台账汇总成一份可引用的
`batch_summary_2nd.md`（版式对齐第一次 `F:/lc_work/prod5p_2023/reports/batch_summary.md`），并把
双 schema 台账规范化成统一 9 列 CSV。**只读输入、只写 `<reports>/`，无 GEE 调用、串行、零依赖**。

## 命令

```bash
python 代码/9.第二次分类川粤宁/c8_batch_summary.py                 # 正式：读 prod2nd_2023 台账
python 代码/9.第二次分类川粤宁/c8_batch_summary.py --reports-dir D:/x # 换输出目录（唯一写点）
python 代码/9.第二次分类川粤宁/c8_batch_summary.py --smoke          # 冒烟：第一次五省批（只读）→ <reports>/_smoke/
python 代码/9.第二次分类川粤宁/c8_batch_summary.py --stdout        # 摘要同时打到 stdout
```

产物：`batch_summary_2nd.md`（总览 / 逐省小计 / 逐瓦台账 / 赛跑异常 / EECU 校准 / 口径核对 / 产物）
+ `eecu_ledger_2nd_normalized.csv`（time,phase,acct,tile,task_id,desc,state,eecu_h,note；幂等覆盖）。

## 口径要点（写死在代码里）

1. **双 schema 台账按列数分支**：`eecu_ledger_2nd.csv` 首行是 5 列表头 `time,acct,task_id,desc,eecu_h`
   （c1 上传段），生产段是 prod2_driver 追加的 8 列且**不再补表头** → 5 列=upload（空列补空）、
   8 列=prod；用 DictReader 按 8 列名读会把 task_id 当 tile。
2. **逐瓦数字不得手写**：状态/账号/EECU 来自 `board_2nd.csv`，QA 列来自 `qa/{tile}_qa.json`；
   省/zone/land_frac 来自 `plan/tiles_secondrun.csv`（board 无这些列），zone 判定 = land_frac<1。
3. **成本带只算终态瓦终值**（QA_OK/DONE/BLOCKED/FAILED）；在跑瓦 board.eecu 是**进行中累计读数**
   （表中标 `†`），混入会把均值系统性压低。判据带沿用第一次裁决：内陆 ≤24 EECU·h。
4. **覆盖率双口径分报**（coverage_px_pct / coverage_km2_pct）+ 海面被分类% + 孤类数（doc 43 纪律）。
5. 进行中/收官由 `state.json` 的 `finished/why` 判定，正文显著标注**数据时点 = board/台账/state 的
   mtime 最大值**（生产进行时即为快照摘要，可反复重跑）。

## 冒烟自检留痕（2026-10-14，输入=第一次五省批，只读；产物 `reports/_smoke/`）

| 项 | 结果 |
|---|---|
| 命令 | `python c8_batch_summary.py --smoke` |
| 输出 | `reports/_smoke/batch_summary_2nd.md` + `eecu_ledger_2nd_normalized.csv`（754 prod 行 + 表头） |
| 与第一次发布报告对齐 | 74/74 QA_OK；总 1240.47/1600（发布 1,240.5；批 1 的 87.9 另计）；内陆 n=49 均值 18.5、沿海 n=24 均值 14.0；超 24 瓦 0 —— **逐项一致** |
| 双 schema 自检 | 摘要头部印固定 5 列/8 列样例的规范化结果（upload 补空 / prod 满 9 列） |
| 异常段 | 多路瓦 T1709/T1610（含收编路 w）；赛跑 467 行/去重 19 瓦；熔断 1 次（原 19.4 带）；保真复跑 2 次 |
| 口径核对段 | plan 76 vs board 74（差 T2713/T2102 试产瓦，⚠ 提示但非缺陷）；状态/QA/表头/字段全 PASS |