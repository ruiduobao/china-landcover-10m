# China 10 m Annual Land Cover (2017–2024)

> **中文**：完整中文版请见 [`README.zh-CN.md`](README.zh-CN.md)。

Annual 10 m land-cover maps of China (2017–2024), built on **Google AlphaEarth Foundations** yearly satellite embeddings (64-D, 10 m) with **server-side Random Forest on Google Earth Engine**. Main product: **24 classes** (plus a 9-macro-class version), benchmarked against GLC_FCS10 (ESSD 2025, reported OA 83.16%).

All production configuration is **frozen (v3.1-R1)**; a **2023 five-province pilot** (Sichuan / Heilongjiang / Guangdong / Ningxia / Fujian, 76 tiles) has been **delivered**, followed by a **second batch of 41 tiles** (Sichuan 24 / Guangdong 10 / Ningxia 6 / Fujian 1) closed out on **2026-10-15**. Large data (sample library, rasters) are **not** included in this repository.

> `[2026-10-18 correction, previous value: "a 2023 five-province pilot (Sichuan / Heilongjiang / Guangdong / Ningxia / Fujian, 76 tiles) has been delivered." — the second batch was not mentioned]`
> The **second batch was absent from this README**. It is 41 tiles, 860.06/950.00 EECU·h, 41/41 QA_OK, closed 2026-10-15. See `文档差异对照表.md` §1 row 3.

## Companion skill: sample verification

[![rs-sample-labeling](https://img.shields.io/badge/skill-rs--sample--labeling-blue)](AI判定遥感样本点SKILL/README.md)

[`AI判定遥感样本点SKILL/`](AI判定遥感样本点SKILL/) — **rs-sample-labeling**: verify remote-sensing sample points with high-resolution imagery (Esri/Google) plus Sentinel-2 time series, and label detailed classes (paddy rice, maize, soybean, wheat, cotton, tea, greenhouse, …) with blind review and QC reports.

```bash
npx rs-sample-labeling install
```

Full Chinese illustrated manual: [`AI判定遥感样本点SKILL/README.zh-CN.md`](AI判定遥感样本点SKILL/README.zh-CN.md).

---

## Status (2026-10-18)

> `[2026-10-18 correction, previous heading: "## Status (2026-10-08)"]` — the date was stale by 10 days and three deliverables/conclusions were missing.

| Milestone | Status |
|---|---|
| Production config | **Frozen v3.1-R1**: 24 classes + local single RF (±2° neighborhood borrowing, 100 trees / minLeaf2 / maxNodes5000 / 20k points) + year-matched samples |
| 2023 five-province pilot | ✅ **76/76 tiles delivered** (Sichuan/Heilongjiang/Guangdong/Ningxia/Fujian, 10 m, palette/pyramids/area tables/QA; 1,328.4/1,600 EECU·h) |
| **Second batch (2023 new-sample v2)** `[2026-10-18: new row — was missing]` | ✅ **41/41 tiles delivered** (Sichuan 24 / Guangdong 10 / Ningxia 6 / **Fujian 1 = T2202, eastern-Guangdong strip top-up**), closed **2026-10-15**; **860.06 / 950.00 EECU·h (90.5%)**; 41/41 QA_OK; fidelity 41/41 = 100% (3,000 pts/tile). Sample pool `train2023_v2` = 1,764,020 pts / 23 classes; delivery pack 45 items / 1.0 GB, `md5sum -c` 44/44 OK |
| Cross-product comparison (E14) | ✅ Pixel-wise vs GLC_FCS10 / GLC_FCS30D / ESRI LULC / ESA WorldCover (group agreement 46–85%) |
| Independent validation | M3 material ready = **4,476 points** (3,210 general + 863 shrub-arbitration + 403 Sanjiang-source). **M3-B (863 pts) has TWO passes of *AI* review — 0.107 → 0.058, both ≤0.40, pointing to "filter the FCS10 shrub layer" — but this is AI, NOT human review and CANNOT be used as OA. M3-A/C/D (3,613 pts) have had NO review at all.** `[2026-10-18 correction, previous value: "M3-B first-pass AI review done (arm-A accuracy 0.107 ...)"]` |
| **Needleleaf/broadleaf 2-class sample v1** `[2026-10-18: new row — was missing]` | 2026-10-17: main sample 55,712 pts (needle 23,700 / broad 30,184 / mixed 1,828; 76 tiles, 23 provinces); 2-class macro-F1 0.7547 (old labels) → **0.8539** (FCS10-referenced). **Pre-registered gates G1–G10: G2 / G3 / G10 = FAIL** — see Limitations |
| National rollout | 313 tiles × 8 years: **scheduling COMPLETE, awaiting go-ahead** — `plan/fleet_summary.json` → 313 tiles / 11 accounts / **2,504 exports** (232×5 + 224×6). Budget ≈43k EECU·h **is a unit-price scale estimate, not measured**. `[2026-10-18 correction, previous value: "313 tiles × 8 years pending (≈43k EECU·h scale)"]` |

## Limitations (must be stated in any publication) `[2026-10-18: new section]`

**1. Full-point OA is NOT publishable.** Independent **human** validation (M3) is **not done**. M3-B's 0.107 → 0.058 are **AI review passes on 863 points**, not human labels; M3-A/C/D (**3,613 points**) have no review at all. Therefore only the following may be published: (a) dev/holdout diagnostics; (b) stratified reference-product agreement (E14); (c) M3-B results **explicitly labelled as AI pre-review**.

**2. Three pre-registered gates FAIL** (needleleaf/broadleaf v1, national scope; `技术文档/77` lines 1050–1059):

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| **G2** aspect unbiasedness | OR ∈ [0.8, 1.25] | **2.05–2.19** (sample 2.183 / population 1.908 / MH 1.903) | **FAIL** — down 56–61% from old labels (4.918); *mitigated, not resolved*; residual OR≈2.0 not attributed |
| **G3** national coverage | ≥90% of China land 2° tiles | **248/313 = 79.2%** (literal) ／ **≈99.8%** (China-terrestrial-forest basis) | **FAIL** on the literal reading; **nature = sampling-frame defect, not progress** |
| **G10** southern expression | southern provinces must not disable needleleaf | **model layer returns zero needleleaf** in Guangdong/Fujian/Zhejiang (recall≈0); true needleleaf in the south is only 1.66–1.78% | **FAIL** (handed to release strategy) |

**3. QA_OK / 100% fidelity ≠ no gaps.** 41/41 `QA_OK` + 3,000-point 100% fidelity **cannot** prove the tile is gap-free: the former is self-referential (expectation rebuilt from the same tiles) and the latter's sampling space does not intersect the defect space. Measured: **all 41 tiles show a wedge of nodata along the southern edge** (~42 px at 30N), hypothesis = `ee.Geometry.Rectangle` geodesic southern-edge bulge (**not settled**). Cause level = "already missing in the source tile" — **do not assert AEF data absence**.

**4. Shrub D2 policy is UNDECIDED.** Production currently keeps the shrub layer (R0). Full-filter branch (server-side `src` filter of `glc_fcs10_2023_shrub`, 521,400 pts → retrain + re-produce 76 tiles ≈1,400 EECU·h) is prepared but **requires explicit human sign-off**. Shrub accuracy/area must not be published as settled before that.

**5. Budget figures are scale estimates.** "≈43k EECU·h" = 16.2 EECU·h/tile × 313 × 8. Only the two pilot batches are measured (1,328.4/1,600 and 860.06/950).

> Traceability for every number above: `文档差异对照表.md` (same directory).

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
