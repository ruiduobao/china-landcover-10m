---
name: rs-sample-labeling
description: Verify remote-sensing sample points with high-resolution imagery (Esri/Google) plus Sentinel-2 time series, and label detailed land-cover classes (paddy rice, maize, soybean, wheat, cotton, tea, greenhouse, and more), with blind review, double-pass audit, and QC reports. Use when a user hands you a point table and asks to "check whether the labels are right / relabel / review interpretations". Not for automated classification modeling or map production.
---

# Remote-Sensing Sample Verification & Land-Cover Labeling

Given a point table (`point_id, lon, lat`), produce **per-point interpretations** — correct / wrong / unreadable, actual class, confidence, evidence — plus reviewable QC statistics and GIS deliverables.

**Method in one sentence**: high-resolution imagery judges structure and boundaries, Sentinel-2 time series judges dates and phenology, auxiliary indicators (canopy height/tree cover/flooding) are supporting evidence only; blind review prevents anchoring, double-pass and audit prove reliability.

> **中文用户**：完整中文图文手册见 [`README.zh-CN.md`](README.zh-CN.md)。

## Four hard rules

1. **Blind review**: review batches (contact sheets, lists) contain **only `point_id`** — no prior labels, strata, or arm info. Label first, then compare. Showing the answer inflates agreement (measured: 12% to 100% on the same points).
2. **Evidence triple**: every conclusion must carry **confidence + evidence (imagery source with date/season + auxiliary indicator values)**; either missing means incomplete.
3. **"Unreadable" is its own column**: never merged into "wrong", never in the confirmation-rate denominator; clouds/shadows/unusable imagery need a stated reason.
4. **No date, no crop**: crop-level conclusions (paddy/maize/soybean/wheat/...) require **dated time-series** support; basemaps only judge structure.

## Standard pipeline

```bash
S=<skill>/scripts
python $S/prep_points.py     --points points.csv --outdir run1 --prior-col prior_class --strata-col region
python $S/fetch_chips.py     --points run1/points_clean.csv --outdir run1          # z18, ~0.5 m structure
python $S/s2_timeseries.py   --points run1/points_clean.csv --outdir run1                              --year 2023 --start 05-01 --end 09-30                # phenology (required for crops)
python $S/build_sheets.py    --chips run1/chips --outdir run1                      # 2x2 blind contact sheets
#   optional: merge phenology hints into the form (cross-check during interpretation)
python $S/prep_points.py     --points run1/points_clean.csv --outdir run1 --metrics run1/s2_metrics.csv
#   -> interpret: read each sheet, fill run1/form_filled.csv (columns below)
python $S/qc_protocol.py     sample --labels run1/form_filled.csv --outdir run1/qc --strata-col region
#   -> take qc/*_blind.csv lists through fetch, sheet, interpret again; fill a second-pass form
python $S/qc_protocol.py     review --labels run1/form_filled.csv --recheck run1/second_pass.csv --outdir run1/qc
python $S/merge_stats.py     --base run1/points_clean.csv --labels run1/form_filled.csv run1/second_pass.csv                              --outdir run1 --prior-col prior_class --strata-col region --threshold 0.40
python $S/points_to_shp.py   --csv run1/merged_labels.csv --outdir run1/shp --gcj02
```

Interpretation form columns (`prep_points.py` skeleton is the standard):
`point_id, verdict (correct/wrong/unreadable), actual_class, confidence (high/medium/low), evidence (imagery features + date + auxiliary indicators)`

## Key technical points

- **Chips carry their own reticle**: the crosshair leaves the target pixel visible; the cyan box marks the target pixel size (default 10 m; use `--box-m 30` for Landsat scenes). Always judge the **dominant land cover inside the box**; note boundary mixing in the evidence column.
- **Contact sheets are 2x2 (4 points, 512 px native)**: a side longer than ~1568 px gets downscaled by vision models and loses detail; 3x3 is only for rough triage.
- **Source choice**: Esri as primary (fast, stable), Google satellite as backup (when Esri tiles are missing/cloudy — i.e. to get a different date); 8-10 workers is the throughput sweet spot.
- **Blurry/stale/cloudy basemaps**: fall back to Sentinel-2 growing-season composites (dated, shows phenology); still unusable -> record "unreadable (reason)".
- **Crop discrimination**: check the regional crop-calendar window first, then combine "high-resolution structure + time-series phenology + management features (flooding/mulch/row spacing/stubble height)". Per-crop criteria: paddy flooding grids, maize wide rows + tall stubble, soybean early canopy closure, cotton mulched strips, contoured tea hedges, lined greenhouse arches. (Measured Chinese criteria live in `docs/`: start from [`README.zh-CN.md`](README.zh-CN.md).)
- **Auxiliary indicators**: `ch_mean` (canopy height) 1.5-5 m hints shrub/young forest, >5 m trees; `ndvi_amp` >0.4 hints annual crops; flooding signal hints paddy/wetland. **Imagery wins on conflict** — say so in the evidence column.

## QC and reporting language

- Double-pass self-agreement >=0.90, "wrong"-audit accuracy >=0.90 (`qc_protocol.py` defaults: 10% random plus all low-confidence/unreadable; per-stratum audit of N). Below threshold -> expand review.
- `merge_stats.py` reports: confirmation rate (total/grouped), prior-by-interpretation confusion, sensitivity (misjudgment rate needed to overturn), double-pass agreement.
- Wording discipline: "confirmation rate / agreement" is an **interpretation result, not accuracy**; AI interpretations must say "AI pre-interpretation, human-reviewed or not"; points sampled from a training pool rate **label quality**, not product accuracy.
- For decision tasks, **pre-register criteria** (thresholds, sample size, seed) before interpreting; do not change them afterwards.

## References (read as needed)

- `docs/` — full Chinese documentation: illustrated manual, field interpretation atlas (crop decision trees, regional crop calendars), imagery/tile handbook (tile math, sources, pitfalls), interpretation protocol (QC, blind review, reporting language). Start from [`README.zh-CN.md`](README.zh-CN.md).
- Chinese `references/` were moved here as `docs/` (same content); if a copy named `references/` exists in an old install, treat it as identical.

## Install (npx / from source)

```bash
# recommended: one command
npx rs-sample-labeling install

# or from this repo
git clone https://github.com/ruiduobao/china-landcover-10m
cd china-landcover-10m/AI判定遥感样本点SKILL
node scripts/install.cjs install
```

## Boundaries

- Point-level interpretation and verification only; no classification modeling, no map production.
- AI interpretation cannot replace a formal two-person blind review; published/delivered conclusions must state their review status.
- Imagery rights: Esri/Google are commercial basemaps — interpretation use only; follow each source's terms when publishing results.
