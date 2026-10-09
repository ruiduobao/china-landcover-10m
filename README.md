# China 10 m Annual Land Cover (2017–2024)

> **中文**：完整中文版请见 [`README.zh-CN.md`](README.zh-CN.md)。

Annual 10 m land-cover maps of China (2017–2024), built on **Google AlphaEarth Foundations** yearly satellite embeddings (64-D, 10 m) with **server-side Random Forest on Google Earth Engine**. Main product: **24 classes** (plus a 9-macro-class version), benchmarked against GLC_FCS10 (ESSD 2025, reported OA 83.16%).

All production configuration is **frozen (v3.1-R1)**; a **2023 five-province pilot** (Sichuan / Heilongjiang / Guangdong / Ningxia / Fujian, 76 tiles) has been **delivered**. Large data (sample library, rasters) are **not** included in this repository.

## Companion skill: sample verification

[![rs-sample-labeling](https://img.shields.io/badge/skill-rs--sample--labeling-blue)](AI判定遥感样本点SKILL/README.md)

[`AI判定遥感样本点SKILL/`](AI判定遥感样本点SKILL/) — **rs-sample-labeling**: verify remote-sensing sample points with high-resolution imagery (Esri/Google) plus Sentinel-2 time series, and label detailed classes (paddy rice, maize, soybean, wheat, cotton, tea, greenhouse, …) with blind review and QC reports.

```bash
npx rs-sample-labeling install
```

Full Chinese illustrated manual: [`AI判定遥感样本点SKILL/README.zh-CN.md`](AI判定遥感样本点SKILL/README.zh-CN.md).

---

## Status (2026-10-08)

| Milestone | Status |
|---|---|
| Production config | **Frozen v3.1-R1**: 24 classes + local single RF (±2° neighborhood borrowing, 100 trees / minLeaf2 / maxNodes5000 / 20k points) + year-matched samples |
| 2023 five-province pilot | ✅ **76/76 tiles delivered** (Sichuan/Heilongjiang/Guangdong/Ningxia/Fujian, 10 m, palette/pyramids/area tables/QA; 1,328.4/1,600 EECU·h) |
| Cross-product comparison (E14) | ✅ Pixel-wise vs GLC_FCS10 / GLC_FCS30D / ESRI LULC / ESA WorldCover (group agreement 46–85%) |
| Independent validation | M3 design done (3,210 general + 863 shrub-arbitration + 403 Sanjiang-source points); **M3-B first-pass AI review done** (arm-A accuracy 0.107 → preregistered rule points to "filter the FCS10 shrub layer", pending human confirmation) |
| National rollout | 313 tiles × 8 years pending (≈43k EECU·h scale) |

## Method sketch

```
AlphaEarth Foundations annual embeddings (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, 64-D, 10 m)
        │  per-tile ±2° neighborhood borrowing (~20k pts/tile, year-matched subsets)
        ▼
GEE server-side smileRandomForest (100 trees / minLeaf=2 / maxNodes=5000 / seed=7)
        ▼
per-tile 10 m classification → uint8 + palette + pyramids (nearest) + nodata=0
        ▼
Deliverables: GeoTIFF + area tables (native 10 m grid, local stats) + overview pages (PIL full-plane check)
```

Evaluation discipline: always use the 25 km block holdout `validation_pool_v2`, stratified by label source (FCS10-family vs non-FCS10).

## Repository layout

| Path | Contents |
|---|---|
| `代码/0.本地流水线/` | **Config source** `lc_conf.py` (class codes / paths); every script reads it as the single source |
| `代码/4.全国清洗训练/` | Cleaning–training chain `e0–e10` (sample QA gate `e10_sample_qa.py`, spatial holdout `e6_spatial_eval.py`) |
| `代码/8.全国生产/` | **National production chain** `p0–p10` (deploy / tile / submit / watch / fetch / mosaic) + account worker template |
| `代码/6.下一代补样/`、`代码/7.vfinal/` | Rare-class top-up chain; v-final spec, separability diagnostics, freeze |
| `技术文档/` | Technical docs in Chinese (index: `技术文档/00_文档索引.md`; `39` handover overview, `38` production basis, `44` cross-product comparison, `46` D0 materials, `47` M3-B AI review) |
| `实验_2026-09_生产前验证/` | Archived reports / code / metrics of pre-production experiments |
| `AGENTS.md` / `handoff.md` / `VERSION.md` | Hard constraints & gate commands for take-over agents / session handover / version history |
| `_tools/`、`_复原记录/` | One-off diagnostics; recovery records after the 2026-09-11 incident |
| `AI判定遥感样本点SKILL/` | **rs-sample-labeling skill** (English entry `SKILL.md`, Chinese manual `README.zh-CN.md`) |

> ⚠️ **Data not in repo**: sample library (2,252,207 pts), embedding year-subsets, 10 m rasters, pilot GeoTIFFs and other large data are excluded (size). Code + docs are sufficient to reproduce the method; contact via issues for data.

## Quick start (needs a Google Earth Engine account)

```bash
# account liveness + project health (zero EECU; run before any account; s0 scripts live in the experiment workspace)
python <workspace>/s0_sweep.py --force && python <workspace>/s0b_health.py

# environment self-check (accounts / proxy / disk / deps)
python 代码/8.全国生产/p0_doctor.py

# sample QA gate (required after any sample change)
python 代码/4.全国清洗训练/e10_sample_qa.py

# national production chain (example: deploy → tile → workers → submit → watch → fetch → mosaic)
python 代码/8.全国生产/p1_deploy_model.py --year 2023
python 代码/8.全国生产/p2_plan_tiles.py
python 代码/8.全国生产/p3_gen_workers.py
python 代码/8.全国生产/fleet/pw_<acct>.py --submit --max 1
python 代码/8.全国生产/p4_watch.py
python 代码/8.全国生产/p5_fetch.py --acct <acct> --tile T0123 --year 2023
python 代码/8.全国生产/p6_mosaic.py --year 2023
```

## Key facts (must-read for reproduction)

1. **Any derivation / overview / statistics of class rasters must explicitly use nearest-neighbor**, or compute on the native 10 m grid (other resampling invents classes).
2. **Always evaluate with the 25 km block holdout `validation_pool_v2`**, stratified by source (FCS10-family vs non-FCS10) — full-point OA is mechanically dominated by eval-point provenance.
3. **Download large rasters with the `getPixels` concurrencer**, aligned per-tile from the GEE export grid (`projection()` → affine); do not use chunked `getDownloadURL`.
4. **Method discipline**: pre-register criteria before running; single-seed noise reaches ±1.5 pp on small samples; single-run gains <2 pp are not trusted.
5. **Yearly training-pool source decisions are pre-registered per year**; pool changes go through server-side `src` filtering and retraining — the sample library itself is read-only.

## Main technical docs (in Chinese)

- [`技术文档/39_实验全览与交接说明.md`](技术文档/39_实验全览与交接说明.md) — experiment ledger, assets, known issues, reproduction guide (read first)
- [`技术文档/38_R1评审修正与2017-2024生产方案.md`](技术文档/38_R1评审修正与2017-2024生产方案.md) — current execution basis
- [`技术文档/44_五省2023与主流土地覆盖产品对比.md`](技术文档/44_五省2023与主流土地覆盖产品对比.md) — cross-product agreement panorama
- [`技术文档/46_D0材料生成与体检报告.md`](技术文档/46_D0材料生成与体检报告.md) · [`技术文档/47_M3B灌丛仲裁AI预判读报告.md`](技术文档/47_M3B灌丛仲裁AI预判读报告.md) — latest experiments
- [`技术文档/00_文档索引.md`](技术文档/00_文档索引.md) — full doc index with validity labels

## Acknowledgments

- Data basis: Google AlphaEarth Foundations (Earth Engine annual embeddings); comparison products GLC_FCS10/FCS30D, ESRI LULC, ESA WorldCover.
- This is an **ongoing study**: every accuracy number states its protocol (dev set / teacher generalization / independent human validation); latest conclusions live in `技术文档/` and `VERSION.md`.
- Citation info and license TBD; contact via issues for data or collaboration.
