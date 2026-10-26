# 园地实验·审计证据（训练规模/计时/特征/泄漏/准备期来源）

> 生成 2026-10-15 14:18:03 ｜ 冻结注册 md5 `0abb0e78cfaa064a1f643875f6b43489`

## 1 训练规模与特征（真实值）

- 特征 = `A00..A63`（64 维，缓存实测全 finite）
- RF = {"n_estimators": 100, "min_samples_leaf": 2, "max_leaf_nodes": 5000, "n_jobs": 10, "note": "sklearn RF ≠ GEE smileRF（本地筛选器）"}
- 每模型训练上限 n_train_limit = 20000；实测每 seed n_train 见下表
- 单元数 8（4 区 × 2 臂），拟合次数 24（× 3 seed）

| 区 | 臂 | seed | n_train | 类2 | 训练类数 | fit_s | predict_s | 单元 wall_s |
|---|---|---|---|---|---|---|---|---|
| GD | OCAP | 7 | 20000 | 200 | 17 | 0.6 | 0.0 | 3.2 |
| GD | OCAP | 11 | 20000 | 200 | 17 | 0.7 | 0.0 | 3.2 |
| GD | OCAP | 23 | 20000 | 200 | 17 | 0.8 | 0.0 | 3.2 |
| GD | baseline | 7 | 20000 | 2254 | 17 | 0.7 | 0.0 | 3.1 |
| GD | baseline | 11 | 20000 | 2243 | 17 | 0.7 | 0.0 | 3.1 |
| GD | baseline | 23 | 20000 | 2305 | 17 | 0.7 | 0.0 | 3.1 |
| NX | OCAP | 7 | 20000 | 200 | 16 | 0.6 | 0.0 | 2.8 |
| NX | OCAP | 11 | 20000 | 200 | 16 | 0.6 | 0.0 | 2.8 |
| NX | OCAP | 23 | 20000 | 200 | 16 | 0.6 | 0.0 | 2.8 |
| NX | baseline | 7 | 20000 | 368 | 16 | 0.7 | 0.0 | 2.9 |
| NX | baseline | 11 | 20000 | 353 | 16 | 0.7 | 0.0 | 2.9 |
| NX | baseline | 23 | 20000 | 370 | 16 | 0.6 | 0.0 | 2.9 |
| SC1 | OCAP | 7 | 20000 | 200 | 18 | 0.8 | 0.0 | 3.2 |
| SC1 | OCAP | 11 | 20000 | 200 | 18 | 0.7 | 0.0 | 3.2 |
| SC1 | OCAP | 23 | 20000 | 200 | 17 | 0.7 | 0.0 | 3.2 |
| SC1 | baseline | 7 | 20000 | 2470 | 18 | 0.8 | 0.0 | 3.4 |
| SC1 | baseline | 11 | 20000 | 2533 | 18 | 0.8 | 0.0 | 3.4 |
| SC1 | baseline | 23 | 20000 | 2511 | 17 | 0.8 | 0.0 | 3.4 |
| SC2 | OCAP | 7 | 20000 | 200 | 15 | 0.7 | 0.0 | 3.1 |
| SC2 | OCAP | 11 | 20000 | 200 | 14 | 0.7 | 0.0 | 3.1 |
| SC2 | OCAP | 23 | 20000 | 200 | 14 | 0.7 | 0.0 | 3.1 |
| SC2 | baseline | 7 | 20000 | 3082 | 15 | 0.7 | 0.0 | 2.9 |
| SC2 | baseline | 11 | 20000 | 3186 | 14 | 0.7 | 0.0 | 2.9 |
| SC2 | baseline | 23 | 20000 | 3180 | 14 | 0.7 | 0.0 | 2.9 |

> 计时汇总：fit 0.60–0.80s（均值 0.70s），predict ≤0.0s，单元 wall ≤3.4s（截止 1200s）。
> 3 seed 约 5s 的原因：20,000×64、100 树、min_leaf=2、max_leaf_nodes=5000（树小）、n_jobs=10。

## 2 泄漏自检（训练窗 vs 参考点位置键）

| 区 | 窗内点数 | 参考点数 | 位置键重叠 | 剔块数 | 窗原始→剔后 |
|---|---|---|---|---|---|
| SC1 | 72723 | 355 | **0** | 13 | 73880→72723 |
| SC2 | 73317 | 199 | **0** | 10 | 74083→73317 |
| GD | 66805 | 599 | **0** | 16 | 69042→66805 |
| NX | 38991 | 410 | **0** | 14 | 40429→38991 |

> 全部区位置键重叠 = 0；训练窗已剔除本区全部 0.25° dev+conf 块（doc70 §6.1 空间留出）。

## 3 准备期来源（无预训/无泄漏）

- **note**：prepare(e1) 只写 cache/ 与 prereg/；不训练、不加载任何既有模型；e2 每 seed 用 RandomForestClassifier(random_state=seed) 从零拟合（无预训/无 warm_start）。
- **no_pretrained_model**：True
- **no_warm_start**：True
- **model_artifacts_written**：[]
- **caches**：['F:\\lc_work\\improve_2023_pilot_20261015\\orchard\\cache\\SC1_train_window.parquet', 'F:\\lc_work\\improve_2023_pilot_20261015\\orchard\\cache\\SC2_train_window.parquet', 'F:\\lc_work\\improve_2023_pilot_20261015\\orchard\\cache\\GD_train_window.parquet', 'F:\\lc_work\\improve_2023_pilot_20261015\\orchard\\cache\\NX_train_window.parquet']

## 4 代码哈希

- `e0_eval_inventory.py` md5 `750cb4b44e53574be3b03acb26849e62`
- `e1_prep_regions.py` md5 `28e8a055aafd11f6ff7a49ce0feef14c`
- `e2_run_unit.py` md5 `4f944222bc2e6effc5f11d4188d5b602`
- `e3_driver.py` md5 `0c71e37a60210bed9b8b22651eaed466`
- `e4_report.py` md5 `3d071b9cea0c06797d501853cc69a33a`
