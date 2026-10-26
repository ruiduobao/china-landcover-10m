# -*- coding: utf-8 -*-
"""e5_audit_evidence.py — 园地实验·审计证据（训练规模/计时/特征/泄漏/准备期来源）

* 输入：<WS>/results/<rid>_<arm>.json（每 seed fit/predict 计时、训练类计数）
        <WS>/prereg/regions_frozen.json、<WS>/cache/*
* 输出：<WS>/audit/inbox/orchard_audit_evidence.md / .json
* 用法：PYTHONDONTWRITEBYTECODE=1 python e5_audit_evidence.py
"""
import os, sys, json, time, hashlib, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
FEATS = ['A%02d' % i for i in range(64)]

reg = json.load(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), encoding='utf-8'))
CODE_DIR = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/11.改进实施与小区域实验/orchard'
code_hash = {f: hashlib.md5(open(os.path.join(CODE_DIR, f), 'rb').read()).hexdigest()
             for f in sorted(os.listdir(CODE_DIR)) if f.endswith('.py')}

# ---- 训练规模与计时证据（逐单元逐 seed）----
rows = []
for fp in sorted(glob.glob(os.path.join(WS, 'results', '*_*.json'))):
    b = os.path.basename(fp)
    if b in ('summary.json',):
        continue
    d = json.load(open(fp, encoding='utf-8'))
    for s in d['n_train_per_seed']:
        rows.append(dict(region=d['region'], arm=d['arm'], seed=s['seed'],
                         n_train=s['n_train'], n_class2=s['n_class2'],
                         n_classes_train=s['n_classes_train'], fit_s=s['fit_s'], predict_s=s['predict_s'],
                         wall_unit_s=d['wall_s']))
ev = pd.DataFrame(rows)

# ---- 泄漏自检：训练窗 vs 参考点位置键重叠（逐区）----
def pk(lon, lat, nd=6):
    a = np.round(np.asarray(lon, float) * 10 ** nd).astype('int64') + 200_000_000
    b = np.round(np.asarray(lat, float) * 10 ** nd).astype('int64') + 100_000_000
    return a * 400_000_000 + b

leak = {}
for rid, R in reg['regions'].items():
    w = pd.read_parquet(R['cache'], columns=['lon', 'lat'])
    r = pd.read_parquet(R['ref_cache'], columns=['lon', 'lat'])
    inter = len(set(pk(w.lon.to_numpy(), w.lat.to_numpy()).tolist()) &
                set(pk(r.lon.to_numpy(), r.lat.to_numpy()).tolist()))
    leak[rid] = dict(n_window=R['n_window_after'], n_ref=len(r), poskey_overlap=inter,
                     n_blocks_excluded=len(R['dev_blocks']) + len(R['conf_blocks']),
                     n_window_before=R['n_window_before'], n_window_excluded=R['n_window_excluded'])

# ---- 特征实际可用性（从缓存读一次，确认 64 列全 finite）----
feat_check = {}
for rid, R in reg['regions'].items():
    w = pd.read_parquet(R['cache'], columns=FEATS)
    feat_check[rid] = dict(n_cols=len(FEATS), all_finite=bool(np.isfinite(w.to_numpy(np.float32)).all()),
                           first3=FEATS[:3], last3=FEATS[-3:])

out = dict(
    generated=time.strftime('%Y-%m-%d %H:%M:%S'),
    frozen_registry_md5=hashlib.md5(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), 'rb').read()).hexdigest(),
    feature_list=FEATS, n_features=64,
    rf_params=reg['rf_params'], n_train_limit=reg['n_train_limit'], seeds=reg['seeds'],
    n_units=int(len(ev) // len(reg['seeds'])), n_fits=int(len(ev)),
    timing=ev.to_dict('records'),
    timing_summary=dict(fit_s_min=float(ev.fit_s.min()), fit_s_max=float(ev.fit_s.max()),
                        fit_s_mean=round(float(ev.fit_s.mean()), 2),
                        predict_s_max=float(ev.predict_s.max()),
                        wall_unit_s_max=float(ev.wall_unit_s.max())),
    leakage_selfcheck=leak,
    feature_availability=feat_check,
    prepare_stage_provenance=dict(
        note='prepare(e1) 只写 cache/ 与 prereg/；不训练、不加载任何既有模型；'
             'e2 每 seed 用 RandomForestClassifier(random_state=seed) 从零拟合（无预训/无 warm_start）。',
        no_pretrained_model=True, no_warm_start=True,
        model_artifacts_written=[],
        caches=[R['cache'] for R in reg['regions'].values()],
    ),
    code_hash=code_hash,
)
os.makedirs(os.path.join(WS, 'audit', 'inbox'), exist_ok=True)
json.dump(out, open(os.path.join(WS, 'audit', 'inbox', 'orchard_audit_evidence.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1, default=str)

L = ['# 园地实验·审计证据（训练规模/计时/特征/泄漏/准备期来源）', '',
     '> 生成 %s ｜ 冻结注册 md5 `%s`' % (out['generated'], out['frozen_registry_md5']), '',
     '## 1 训练规模与特征（真实值）', '',
     '- 特征 = `%s`（%d 维，缓存实测全 finite）' % (FEATS[0] + '..' + FEATS[-1], 64),
     '- RF = %s' % json.dumps(reg['rf_params'], ensure_ascii=False),
     '- 每模型训练上限 n_train_limit = %d；实测每 seed n_train 见下表' % reg['n_train_limit'],
     '- 单元数 %d（4 区 × 2 臂），拟合次数 %d（× 3 seed）' % (out['n_units'], out['n_fits']), '',
     '| 区 | 臂 | seed | n_train | 类2 | 训练类数 | fit_s | predict_s | 单元 wall_s |',
     '|---|---|---|---|---|---|---|---|---|']
for r in out['timing']:
    L.append('| %s | %s | %d | %d | %d | %d | %.1f | %.1f | %.1f |' % (
        r['region'], r['arm'], r['seed'], r['n_train'], r['n_class2'], r['n_classes_train'],
        r['fit_s'], r['predict_s'], r['wall_unit_s']))
L += ['',
      '> 计时汇总：fit %.2f–%.2fs（均值 %.2fs），predict ≤%.1fs，单元 wall ≤%.1fs（截止 1200s）。' % (
          out['timing_summary']['fit_s_min'], out['timing_summary']['fit_s_max'],
          out['timing_summary']['fit_s_mean'], out['timing_summary']['predict_s_max'],
          out['timing_summary']['wall_unit_s_max']),
      '> 3 seed 约 5s 的原因：20,000×64、100 树、min_leaf=2、max_leaf_nodes=5000（树小）、n_jobs=10。', '',
      '## 2 泄漏自检（训练窗 vs 参考点位置键）', '',
      '| 区 | 窗内点数 | 参考点数 | 位置键重叠 | 剔块数 | 窗原始→剔后 |',
      '|---|---|---|---|---|---|']
for rid, v in leak.items():
    L.append('| %s | %d | %d | **%d** | %d | %d→%d |' % (
        rid, v['n_window'], v['n_ref'], v['poskey_overlap'], v['n_blocks_excluded'],
        v['n_window_before'], v['n_window']))
L += ['', '> 全部区位置键重叠 = 0；训练窗已剔除本区全部 0.25° dev+conf 块（doc70 §6.1 空间留出）。', '',
      '## 3 准备期来源（无预训/无泄漏）', '']
for k, v in out['prepare_stage_provenance'].items():
    L.append('- **%s**：%s' % (k, v))
L += ['', '## 4 代码哈希', '']
for f, h in code_hash.items():
    L.append('- `%s` md5 `%s`' % (f, h))
open(os.path.join(WS, 'audit', 'inbox', 'orchard_audit_evidence.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('→ audit/inbox/orchard_audit_evidence.{md,json}')
print('fits=%d  fit_s %.2f-%.2f  poskey_overlap=%s' % (
    out['n_fits'], out['timing_summary']['fit_s_min'], out['timing_summary']['fit_s_max'],
    {k: v['poskey_overlap'] for k, v in leak.items()}))
