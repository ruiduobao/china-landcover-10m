# -*- coding: utf-8 -*-
"""c7_local_ab.py — 第二次训练池(v2) vs 第一次训练池(v3.1)：川/粤/宁留出窗 · 纯本地子采样 A/B 预览

* 输入：
  - 旧池 F:/lc_work/prod5p_2023/data/train2023_clean.parquet（2,060,577 点；class_new 为旧 30 码，
    经 v31_map 映射为 24 类）
  - 新池 F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（1,764,020 点；cl 已是 v31 24 码，缺 8 类）
  - 留出窗 F:/lc_work/v31_exp/data/yearly/eval_2023_w{1..5}.parquet（row_id/lon/lat/class_new/src/A00..A63）
  - 省界 DataV GCJ-02（经 p5common 只读加载）；排除键 / 归省 / 分层口径逐条 import 复用 c4_eval_second.py
* 规则源：AGENTS 硬约束 4/5/12（留出池 / 按来源分层 / 噪声纪律）；doc38 <2pp 需多种子；
  c4_eval_second.py（双键排除、归省、strata_block）
* 门槛（**预注册，写死再跑**，原文固化于输出 json 的 prereg 字段）：
  H0 对照假设：新池在 川/粤/宁 留出窗上的 24 类 OA 与 9 大类 OA **不低于**旧池。
  判读（Δ = 新 − 旧，pp 单位）：Δ ≤ −1.0pp = 回归 ｜ −1.0pp < Δ ≤ +1.0pp = 持平 ｜ Δ > +1.0pp = 提升；
  按省分别报 + 三省合计报。
  训练规格：sklearn RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
  max_leaf_nodes=5000, random_state=7, n_jobs=15)；特征 = A00..A63；两臂**同一子采样规模**
  （每类 ≤30000 点、类内随机、种子同模型种子；不足的类全取）。
  **本实验是"子采样 A/B"：数值不得与全量生产/发布数字直接比，只比方向。**
  评估口径：留出点逐 rid / 位置键排除两池 + 错位清单 → DataV 归省（川/粤/宁）→ 预测 →
  24 类 OA / 9 大类 OA（macro9.json）/ macroF1 / 按 src 分层（fcs10 系 vs 非 fcs10，强制分报）/
  逐类 F1（n<30 只作合并级参考）。
  可比性处理：两臂评估时均剔除真值 = 8（针阔混交林，新池无 8 类）的留出点并显式报剔除数；
  旧池另附"含 8 类全点"数字仅供参考（两臂都附，明确不可比）。
* 输出：F:/lc_work/prod2nd_2023/reports/ab_local/ab_report.md + ab_metrics.json
        （--save-models 时另落 models/<arm>_seed<seed>.joblib；本脚本只写 ab_local，不碰
        prod5p_2023 / v31_exp / 数据 目录）
* 用法：
    python 代码/9.第二次分类川粤宁/c7_local_ab.py                  # 正式：seed 7
    python 代码/9.第二次分类川粤宁/c7_local_ab.py --seeds 7,11,23  # 多种子复核（串行）
    python 代码/9.第二次分类川粤宁/c7_local_ab.py --prov 粤        # 只评粤（两臂训练仍全跑）
    python 代码/9.第二次分类川粤宁/c7_local_ab.py --smoke          # 管路冒烟（输出带 _smoke 后缀，非结论）
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')      # 15 个 joblib 进程 × 1 BLAS 线程，控制在 ≤15 核
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, argparse

sys.dont_write_bytecode = True       # 只读纪律：不往 p5common / v31_common / 本目录写 __pycache__
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                              # 同目录复用 c4_eval_second
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')  # p5common：省界/别名/分层口径（只读）
sys.path.insert(0, r'F:/lc_work/v31_exp/code')        # v31_common：metrics/to_v31/to_macro（只读）

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import c4_eval_second as C4
import p5common as PC
import v31_common as VC

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
ROOT1 = r'F:/lc_work/prod5p_2023'                      # 第一次（只读）
ROOT2 = r'F:/lc_work/prod2nd_2023'                     # 第二次
POOL1 = os.path.join(ROOT1, 'data', 'train2023_clean.parquet')   # 旧池（旧 30 码 class_new）
POOL2 = os.path.join(ROOT2, 'data', 'train2023_v2.parquet')      # 新池（v31 24 码 cl）
EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'           # 留出窗（只读）
OUTDIR = os.path.join(ROOT2, 'reports', 'ab_local')
MODELDIR = os.path.join(OUTDIR, 'models')
FEATS = ['A%02d' % i for i in range(64)]
PROV2 = [('川', '四川省'), ('粤', '广东省'), ('宁', '宁夏回族自治区')]   # 第二次 40 瓦三（同一批对照口径）
LAB_COL = {'old': 'class_new', 'new': 'cl'}
GATE_PP = 1.0                      # 预注册阈值：±1.0pp
CLASS8 = 8                         # 针阔混交林（新池缺席类）
PREREG = {
    'hypothesis': '新池(train2023_v2, 23 类)在川/粤/宁留出窗上的 24 类 OA 与 9 大类 OA 不低于旧池(train2023_clean, 24 类)',
    'gate_pp': GATE_PP,
    'verdict_rule': 'Δ = 新 − 旧（pp）：Δ ≤ −1.0pp=回归；−1.0pp < Δ ≤ +1.0pp=持平；Δ > +1.0pp=提升。按省 + 三省合计',
    'train_spec': 'sklearn RandomForestClassifier(n_estimators=100, min_samples_leaf=2, '
                  'max_leaf_nodes=5000, random_state=<seed>, n_jobs=15)；特征 A00..A63',
    'subsample': '两臂同一子采样规模：每类 ≤30000 点、类内随机（种子=模型种子）；不足的类全取。'
                 '**子采样 A/B：只比方向，不与全量生产/发布数字比分值**',
    'eval': '留出点逐 rid/位置键排除两池+错位清单 → DataV 归省（川/粤/宁）→ 同一留出集评估：'
            '24 类 OA / 9 大类 OA / macroF1 / 按 src 分层（fcs10 系 vs 非 fcs10）/ 逐类 F1（n<30 合并级）',
    'class8_handling': '两臂均剔除真值=8（针阔混交林，新池无 8 类）的留出点并显式报剔除数；'
                       '另附"含 8 类全点"两臂数字，明确不可比、仅供参考',
    'noise_discipline': '单次种子噪声 ±1.5pp（doc38/AGENTS 硬约束 12）；<2pp 的 Δ 属噪声带内，需多种子复核',
    'gee_calls': 0,
}


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description='第二次新池 vs 第一次旧池 · 纯本地子采样 A/B 预览')
    ap.add_argument('--prov', default='all', help='川/粤/宁 或 all 或逗号分隔（只影响评估范围，两臂训练恒全跑）')
    ap.add_argument('--seeds', default='7', help='逗号分隔；首个为主种子（默认 7）。建议多种子 7,11,23')
    ap.add_argument('--cap', type=int, default=30000, help='每类子采样上限（默认 30000；0=不设限＝全池，仅作补充稳健性检查）')
    ap.add_argument('--trees', type=int, default=100, help='n_estimators（预注册 100）')
    ap.add_argument('--windows', default='w1,w2,w3,w4,w5')
    ap.add_argument('--year', type=int, default=2023)
    ap.add_argument('--out-suffix', default='', help='输出文件名后缀（默认空；补充跑用，如 _fullpool）')
    ap.add_argument('--smoke', action='store_true', help='管路冒烟：trees=5、cap=3000，输出带 _smoke 后缀')
    ap.add_argument('--save-models', action='store_true', help='joblib 落 models/（默认不落）')
    a = ap.parse_args()
    if a.smoke:
        a.trees, a.cap = 5, 3000      # 冒烟：小树小采样（--seeds 仍按命令行，便于验证多种子报告路径）
    a.seed_list = [int(s) for s in a.seeds.split(',') if s.strip()]
    assert a.seed_list, '--seeds 为空'
    a.suffix = '_smoke' if a.smoke else a.out_suffix
    return a


# ---------------- 数据装载 ----------------
def load_holdout(year, windows):
    """五窗留出表（含 A00..A63 特征）合并。"""
    frames = []
    for w in [s.strip() for s in windows.split(',') if s.strip()]:
        fp = os.path.join(EVAL_DIR, 'eval_%d_%s.parquet' % (year, w))
        if not os.path.exists(fp):
            emit('缺表跳过 %s' % fp)
            continue
        d = pd.read_parquet(fp, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + FEATS)
        d['win'] = w
        frames.append(d)
    assert frames, '无可用留出表'
    return pd.concat(frames, ignore_index=True)


def load_pool(arm):
    """整池读入内存：返回 (X float32[n,64], lab int64[n])。旧池 30 码经 to_v31 → 24 类。"""
    fp = POOL1 if arm == 'old' else POOL2
    col = LAB_COL[arm]
    t0 = time.time()
    lab = pq.read_table(fp, columns=[col])[col].to_numpy()
    lab = np.asarray(lab, dtype=np.int64)
    if arm == 'old':
        lab = VC.to_v31(lab)
    t = pq.read_table(fp, columns=FEATS)
    X = np.empty((t.num_rows, len(FEATS)), dtype=np.float32)
    for i in range(len(FEATS)):
        X[:, i] = t.column(i).to_numpy()
    del t
    bad = int((~np.isfinite(X)).sum())
    emit('池装载 %s：n=%d 类数=%d 非有限特征=%d（%.1f 分钟）' % (
        arm, len(lab), len(np.unique(lab)), bad, (time.time() - t0) / 60))
    assert bad == 0, '%s 池存在非有限特征 %d 个' % (arm, bad)
    return X, lab


def subsample_idx(lab, cap, seed):
    """类分层子采样：每类 ≤cap，类内随机（种子=模型种子）；不足的类全取。cap<=0 = 全池不裁。"""
    if cap is None or cap <= 0:
        return np.arange(len(lab))
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        if len(idx) > cap:
            idx = idx[rng.permutation(len(idx))[:cap]]
        keep.append(idx)
    out = np.concatenate(keep)
    out.sort()
    return out


def train_rf(X, y, seed, trees):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, max_leaf_nodes=5000,
                                random_state=seed, n_jobs=15)
    t0 = time.time()
    rf.fit(X, y)
    return rf, time.time() - t0


# ---------------- 判读 ----------------
def judge_pct(d_pp):
    """预注册判读：Δ(pp) → 提升/持平/回归。"""
    if d_pp <= -GATE_PP:
        return '回归'
    if d_pp <= GATE_PP:
        return '持平'
    return '提升'


def delta_pack(bold, bnew):
    """两臂块 → Δ（比例 + pp）与判读。块缺席（如分层 n<10）→ None/'—'。"""
    out = {}
    for k in ('OA', 'macroF1', 'm9'):
        av, bv = bold.get(k), bnew.get(k)
        if av is None or bv is None:
            out[k] = None
            out[k + '_pp'] = None
        else:
            d = bv - av
            out[k] = round(d, 4)
            out[k + '_pp'] = round(d * 100, 2)
    out['judge_24'] = judge_pct(out['OA_pp']) if out['OA_pp'] is not None else '—'
    out['judge_9'] = judge_pct(out['m9_pp']) if out['m9_pp'] is not None else '—'
    return out


def _f(x, nd=4):
    return '—' if x is None else ('%.*f' % (nd, x))


def _dp(v, nd=1):
    """pp 值格式化（None → '—'）。"""
    return '—' if v is None else ('%+.*fpp' % (nd, v))


def per_class_lines(bold, bnew, names, title):
    """逐类 F1 对照 md 表（24 类口径，按类码联合）。"""
    per1 = (bold.get('all', {}) or {}).get('per', {}) or {}
    per2 = (bnew.get('all', {}) or {}).get('per', {}) or {}
    ks = sorted(set(int(k) for k in per1) | set(int(k) for k in per2))
    L = ['### %s' % title, '',
         '| 类码 | 名称 | n(真值) | 旧池 F1 | 新池 F1 | ΔF1 | 备注 |',
         '|---|---|---|---|---|---|---|']
    for c in ks:
        a, b = per1.get(str(c)) or per1.get(c), per2.get(str(c)) or per2.get(c)
        n = None
        for d in (a, b):
            if d and d.get('n', 0) > 0:
                n = d['n']
        note = []
        if n is None:
            note.append('真值=8 已按口径剔除（仅预测侧）' if c == CLASS8 else '本省无该类真值（仅预测侧）')
        elif n < 30:
            note.append('⚠n<30 仅合并级')
        if a is None:
            note.append('旧池缺席（未预测亦无真值）')
        if b is None:
            note.append('新池缺席')
        fa = a['f1'] if a and (a.get('n', 0) > 0) else None
        fb = b['f1'] if b and (b.get('n', 0) > 0) else None
        dstr = '—' if (fa is None or fb is None) else '%+.3f' % (fb - fa)
        L.append('| %d | %s | %s | %s | %s | %s | %s |' % (
            c, names.get(str(c), '?'), n if n is not None else '—',
            _f(fa, 3), _f(fb, 3), dstr, '；'.join(note)))
    return L


# ---------------- 主流程 ----------------
def main():
    a = parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.save_models:
        os.makedirs(MODELDIR, exist_ok=True)
    names = VC.V31_NAMES()
    tags = [t for t, _ in PROV2] if a.prov == 'all' else \
        [PC.PROV_SHORT.get(PC.PROV_ALIAS.get(s.strip(), s.strip()), s.strip())
         for s in a.prov.split(',') if s.strip()]
    prov_sel = [(t, dict(PROV2)[t]) for t in tags]
    emit('c7 本地 A/B：省=%s ｜ seeds=%s ｜ trees=%d ｜ cap=%d ｜ 输出后缀=%s' % (
        ','.join(tags), a.seed_list, a.trees, a.cap, a.suffix or '(无)'))

    # 1) 留出点 + 排除（口径逐条复用 c4）
    ev_raw = load_holdout(a.year, a.windows)
    n0 = len(ev_raw)
    keep, xd = C4.build_exclusions(ev_raw)
    emit('留出点 %d → 排除：∩旧池 rid=%d ｜ ∩新池 point_id 键=%d ｜ ∩新池 6 位位置键=%d ｜ '
         '∩错位=%d → 保留 %d' % (n0, xd['excl_v1_rid'], xd['excl_v2_pid'],
                                 xd['excl_v2_pos'], xd['excl_misaligned'], xd['n_keep']))
    ev = ev_raw[keep].reset_index(drop=True)
    del ev_raw

    # 2) 归省（DataV GCJ-02，复用 c4.assign_prov；五省集合保持与第一次矩阵口径一致）
    prov_full, bdist = C4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['prov_full'] = prov_full
    ev['near_boundary'] = bdist < 0.005
    mat = ev.groupby(['win', 'prov_full']).size().unstack(fill_value=0)
    emit('窗×省 交叠矩阵：\n%s' % mat.to_string())

    t24_all = VC.to_v31(ev.class_new.to_numpy().astype(np.int64))
    isf_all = PC.is_fcs10(ev.src).to_numpy()
    Xe = ev[FEATS].to_numpy(np.float32)
    assert int((~np.isfinite(Xe)).sum()) == 0, '留出点存在非有限特征'

    # 3) 逐省子集：真值 8 类剔除（两臂同集）
    prov_idx = {}
    for tag, full in prov_sel:
        ip = np.flatnonzero(ev.prov_full.to_numpy() == full)
        if len(ip) == 0:
            emit('%s 无留出点——跳过' % tag)
            continue
        m8 = t24_all[ip] != CLASS8
        prov_idx[tag] = dict(full=full, ip=ip, ie=ip[m8], n8=int((~m8).sum()))
    assert prov_idx, '所选省份均无留出点'

    # 4) 两池装载（一次）+ 逐种子训练评估
    X1, y1 = load_pool('old')
    X2, y2 = load_pool('new')
    cnt1 = {int(c): int(v) for c, v in zip(*np.unique(y1, return_counts=True))}
    cnt2 = {int(c): int(v) for c, v in zip(*np.unique(y2, return_counts=True))}

    runs = []
    for seed in a.seed_list:
        run = dict(seed=seed, trees=a.trees, cap=a.cap)
        preds = {}
        for arm, X, y in (('old', X1, y1), ('new', X2, y2)):
            ii = subsample_idx(y, a.cap, seed)
            run.setdefault('n_train', {})[arm] = int(len(ii))
            run.setdefault('n_train_classes', {})[arm] = int(len(np.unique(y[ii])))
            rf, sec = train_rf(X[ii], y[ii], seed, a.trees)
            run.setdefault('fit_seconds', {})[arm] = round(sec, 1)
            t0 = time.time()
            preds[arm] = rf.predict(Xe)
            emit('seed %d ｜ %s 臂：训练 %d 点（%s 类）耗时 %.1fs ｜ 预测 %d 点 %.1fs' % (
                seed, arm, len(ii), len(np.unique(y[ii])), sec, len(Xe), time.time() - t0))
            if a.save_models:
                import joblib
                joblib.dump(rf, os.path.join(MODELDIR, '%s_seed%d%s.joblib' % (arm, seed, a.suffix)))
        # 逐省 + 合计
        provs_out = {}
        all_ie = []
        for tag, d in prov_idx.items():
            ip, ie = d['ip'], d['ie']
            all_ie.append(ie)
            blk = {}
            for arm in ('old', 'new'):
                blk[arm] = C4.strata_block(t24_all[ie], preds[arm][ie], isf_all[ie])
                # 含 8 类全点（仅供参考，不可比）
                bf = C4.strata_block(t24_all[ip], preds[arm][ip], isf_all[ip])
                blk[arm + '_full8'] = {'OA': bf['all']['OA'], 'm9': bf['all']['m9'],
                                       'n': bf['all']['n']}
            o = dict(n_eval=int(len(ie)), n_truth8=d['n8'],
                     n_fcs10=int(isf_all[ie].sum()), n_nonfcs10=int((~isf_all[ie]).sum()),
                     n_pred8={arm: int((preds[arm][ie] == CLASS8).sum()) for arm in ('old', 'new')},
                     old=blk['old'], new=blk['new'],
                     old_full8=blk['old_full8'], new_full8=blk['new_full8'])
            o['delta'] = delta_pack(blk['old']['all'], blk['new']['all'])
            o['delta_fcs10'] = delta_pack(blk['old'].get('fcs10', {}), blk['new'].get('fcs10', {}))
            o['delta_nonfcs10'] = delta_pack(blk['old'].get('nonfcs10', {}), blk['new'].get('nonfcs10', {}))
            provs_out[tag] = o
        ic = np.concatenate(all_ie)
        comb = dict(n_eval=int(len(ic)), n_fcs10=int(isf_all[ic].sum()),
                    n_nonfcs10=int((~isf_all[ic]).sum()))
        blk = {}
        for arm in ('old', 'new'):
            blk[arm] = C4.strata_block(t24_all[ic], preds[arm][ic], isf_all[ic])
        comb.update(old=blk['old'], new=blk['new'])
        comb['delta'] = delta_pack(blk['old']['all'], blk['new']['all'])
        comb['delta_fcs10'] = delta_pack(blk['old'].get('fcs10', {}), blk['new'].get('fcs10', {}))
        comb['delta_nonfcs10'] = delta_pack(blk['old'].get('nonfcs10', {}), blk['new'].get('nonfcs10', {}))
        run['provs'] = provs_out
        run['combined'] = comb
        runs.append(run)
        del preds

    # 5) 输出 json
    cap_txt = ('每类≤%d、种子=模型种子' % a.cap) if a.cap > 0 else '全池不裁（补充稳健性口径，非预注册主口径）'
    scale_note = ('子采样 A/B（每类≤%d 点）：数值不得与全量生产/发布数字相比，只比方向' % a.cap) \
        if a.cap > 0 else '全池口径（无子采样），但仍是本地 sklearn RF，不得与 GEE 生产数字直接比'
    out = dict(
        experiment='c7_local_ab', gee_calls=0,
        title='第二次新池(train2023_v2, 23 类) vs 第一次旧池(train2023_clean, 24 类) · 川/粤/宁留出窗 · 纯本地 A/B 预览',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        script='代码/9.第二次分类川粤宁/c7_local_ab.py',
        prereg=PREREG,
        pools={
            'old': dict(path=POOL1, n=len(y1), n_classes=len(np.unique(y1)),
                        label='class_new(旧30码)→to_v31 24类', counts=cnt1,
                        subsample_rule=cap_txt),
            'new': dict(path=POOL2, n=len(y2), n_classes=len(np.unique(y2)),
                        label='cl(v31 24码，缺 8 类)', counts=cnt2,
                        subsample_rule=cap_txt),
        },
        holdout=dict(n_raw=n0, excl_detail=xd,
                     n_eval_total=int(sum(len(d['ie']) for d in prov_idx.values())),
                     per_prov={t: dict(n_all_prov=int(len(d['ip'])), n_eval=int(len(d['ie'])),
                                       n_truth8=d['n8']) for t, d in prov_idx.items()},
                     windows=a.windows, year=a.year),
        runs=runs, primary_seed=a.seed_list[0], all_seeds=a.seed_list,
        limitations=[
            scale_note,
            '留出窗为开发集口径；单种子噪声 ±1.5pp，<2pp 的 Δ 属噪声带内',
            '真值 class_new 经 to_v31 映射，评估标签体系与两池训练标签同源（FCS10 系惯例风险仍在）',
        ],
    )
    fpj = os.path.join(OUTDIR, 'ab_metrics%s.json' % a.suffix)
    with open(fpj, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    emit('→ %s' % fpj)

    # 6) 输出 md
    fp_md = os.path.join(OUTDIR, 'ab_report%s.md' % a.suffix)
    with open(fp_md, 'w', encoding='utf-8') as f:
        f.write(build_report(a, out, names, cnt1, cnt2, prov_idx))
    emit('→ %s' % fp_md)


def build_report(a, out, names, cnt1, cnt2, prov_idx):
    scale_line = ('> **这是"子采样 A/B"**（每类 ≤%d 点）：数值不得与全量生产/发布数字直接比较，**只比方向**。' % a.cap) \
        if a.cap > 0 else '> **全池口径**（无子采样，补充稳健性检查）：仍是本地 sklearn RF，不得与 GEE 生产数字直接比。'
    L = ['# 第二次新池 vs 第一次旧池 · 纯本地%s A/B 预览（川/粤/宁 · %d 留出窗）' % (
             '子采样' if a.cap > 0 else '全池', out['holdout']['year']),
         '',
         '> 生成 %s ｜ 脚本 `%s` ｜ **零 GEE 调用（gee_calls=0）**，纯本地 CPU' % (
             out['generated'], out['script']),
         scale_line,
         '> 预注册判据：Δ = 新 − 旧；Δ ≤ −1.0pp=**回归**；−1.0 < Δ ≤ +1.0pp=**持平**；Δ > +1.0pp=**提升**。',
         '> 单次种子噪声 ±1.5pp（doc38/AGENTS 硬约束 12）——<2pp 的 Δ 属噪声带内，须多种子复核。',
         '']
    # §0 预注册
    L += ['## 0 预注册判据（写死再跑）', '']
    for k, v in PREREG.items():
        L.append('- **%s**：%s' % (k, v))
    L += ['']
    # §1 数据与排除
    h = out['holdout']
    xd = h['excl_detail']
    L += ['## 1 数据与留出点口径', '',
          '| 步骤 | 点数 |', '|---|---|',
          '| 五窗留出原始 | %d |' % h['n_raw'],
          '| 排除 ∩旧池(train2023_clean) rid | %d |' % xd['excl_v1_rid'],
          '| 排除 ∩新池(train2023_v2) point_id 键 | %d |' % xd['excl_v2_pid'],
          '| 排除 ∩新池(train2023_v2) 6 位位置键 | %d |' % xd['excl_v2_pos'],
          '| 排除 ∩错位清单 | %d |' % xd['excl_misaligned'],
          '| 保留 | %d |' % xd['n_keep'], '',
          '| 省 | 省留出点 | 真值=8（剔除） | 计入评估 | fcs10 系 | 非 fcs10 |',
          '|---|---|---|---|---|---|']
    for tag, full in PROV2:
        if tag not in prov_idx:
            continue
        d = out['runs'][0]['provs'][tag]
        L.append('| %s | %d | %d | %d | %d | %d |' % (
            tag, len(prov_idx[tag]['ip']), d['n_truth8'], d['n_eval'], d['n_fcs10'], d['n_nonfcs10']))
    comb0 = out['runs'][0]['combined']
    L += ['| **合计** |  |  | %d | %d | %d |' % (comb0['n_eval'], comb0['n_fcs10'], comb0['n_nonfcs10']),
          '',
          '> 真值 = 8（针阔混交林）点为"新池无 8 类"的不可比点，两臂评估一律剔除；见 §3 全点参考。',
          '> 排除键口径（point_id 重号假命中风险）与 c4_eval_second.py 完全一致（both 双键）；'
          '位置键独有命中 %d，point_id 独有命中 %d。' % (xd['excl_v2_pos_only'], xd['excl_v2_pid_only']),
          '']
    # §1.2 两池构成
    L += ['### 1.2 两池构成（24 类口径点量）', '',
          '| 类码 | 名称 | 旧池 n | 新池 n |', '|---|---|---|---|']
    for c in range(1, 25):
        L.append('| %d | %s | %s | %s |' % (c, names.get(str(c), '?'),
                                            f"{cnt1.get(c, 0):,}", f"{cnt2.get(c, 0):,}"))
    L += ['| | **合计** | %s | %s |' % (f"{sum(cnt1.values()):,}", f"{sum(cnt2.values()):,}"), '',
          '> %s；实际参与训练量见各 run 的 n_train（旧池 %s / 新池 %s）。' % (
              ('每类子采样上限 %d' % a.cap) if a.cap > 0 else '全池不裁（补充口径）',
              ', '.join(str(r['n_train']['old']) for r in out['runs']),
              ', '.join(str(r['n_train']['new']) for r in out['runs'])), '']
    # §2 主结果（主种子）
    r0 = out['runs'][0]
    L += ['## 2 主结果（seed %d，主种子）' % r0['seed'], '',
          '训练量：旧池 %d 点（%d 类）｜ 新池 %d 点（%d 类）｜ 训练耗时 旧 %.1fs / 新 %.1fs。' % (
              r0['n_train']['old'], r0['n_train_classes']['old'],
              r0['n_train']['new'], r0['n_train_classes']['new'],
              r0['fit_seconds']['old'], r0['fit_seconds']['new']),
          '',
          '### 2.1 总口径（全点 = 已剔除真值 8 类点）', '',
          '| 省 | n | 口径 | 旧池 | 新池 | Δ（pp） | 判读 |', '|---|---|---|---|---|---|---|']
    for tag in [t for t, _ in PROV2 if t in prov_idx]:
        d = r0['provs'][tag]
        for key, lab in (('OA', '24 类 OA'), ('m9', '9 大类 OA'), ('macroF1', 'macroF1(24 类)')):
            dl = d['delta'].get(key)
            jd = d['delta']['judge_24'] if key == 'OA' else (d['delta']['judge_9'] if key == 'm9' else '—')
            L.append('| %s | %d | %s | %.4f | %.4f | %s | %s |' % (
                tag, d['n_eval'], lab, d['old']['all'][key], d['new']['all'][key],
                '%+.4f (%+.1fpp)' % (dl, dl * 100) if dl is not None else '—', jd))
    c = r0['combined']
    for key, lab in (('OA', '24 类 OA'), ('m9', '9 大类 OA'), ('macroF1', 'macroF1(24 类)')):
        dl = c['delta'].get(key)
        jd = c['delta']['judge_24'] if key == 'OA' else (c['delta']['judge_9'] if key == 'm9' else '—')
        L.append('| **合计** | %d | %s | %.4f | %.4f | %s | %s |' % (
            c['n_eval'], lab, c['old']['all'][key], c['new']['all'][key],
            '%+.4f (%+.1fpp)' % (dl, dl * 100) if dl is not None else '—', jd))
    L += ['']
    # §2.2 分层
    L += ['### 2.2 按 src 分层（强制分报）', '',
          '| 省 | 分层 | n | 旧池 OA | 新池 OA | ΔOA | 旧池 9 大类 OA | 新池 9 大类 OA | Δ9 大类 |',
          '|---|---|---|---|---|---|---|---|---|']
    for tag in [t for t, _ in PROV2 if t in prov_idx]:
        d = r0['provs'][tag]
        for skey, sname, dl_key in (('fcs10', 'fcs10 系', 'delta_fcs10'),
                                    ('nonfcs10', '非 fcs10', 'delta_nonfcs10')):
            if skey not in d['old'] or skey not in d['new']:
                L.append('| %s | %s | — | 缺席（n<10） | | | | | |' % (tag, sname))
                continue
            o, nw = d['old'][skey], d['new'][skey]
            dl = d[dl_key]
            L.append('| %s | %s | %d | %.4f | %.4f | %s | %.4f | %.4f | %s |' % (
                tag, sname, o['n'], o['OA'], nw['OA'],
                _dp(dl['OA_pp']), o['m9'], nw['m9'], _dp(dl['m9_pp'])))
    for skey, sname, dl_key in (('fcs10', 'fcs10 系', 'delta_fcs10'),
                                ('nonfcs10', '非 fcs10', 'delta_nonfcs10')):
        if skey in c['old'] and skey in c['new']:
            o, nw, dl = c['old'][skey], c['new'][skey], c[dl_key]
            L.append('| **合计** | %s | %d | %.4f | %.4f | %s | %.4f | %.4f | %s |' % (
                sname, o['n'], o['OA'], nw['OA'], _dp(dl['OA_pp']), o['m9'], nw['m9'], _dp(dl['m9_pp'])))
    L += ['',
          '> 读法：全点 OA 受 FCS10 标注惯例主导（AGENTS 硬约束 5/15）；'
          '分层 Δ 是模型差异的干净窗口。', '']
    # §2.3 逐类 F1
    for tag in [t for t, _ in PROV2 if t in prov_idx]:
        d = r0['provs'][tag]
        L += per_class_lines({'all': d['old']['all']}, {'all': d['new']['all']}, names,
                             '%s · 逐类 F1（n 为真值点数；n<30 只作合并级参考）' % tag)
        n8o, n8n = d['n_pred8']['old'], d['n_pred8']['new']
        L += ['', '> 预测 8 类次数（该类真值已剔）：旧池 %d ｜ 新池 %d（新池训练无 8 类，恒 0）' % (n8o, n8n), '']
    # §3 含 8 全点参考
    L += ['## 3 "含 8 类全点"参考（两臂均列；与 §2 不可比，仅供参考）', '',
          '| 省 | n(全) | 旧池 OA | 新池 OA | Δ | 旧池 9 大类 OA | 新池 9 大类 OA | Δ |',
          '|---|---|---|---|---|---|---|---|']
    for tag in [t for t, _ in PROV2 if t in prov_idx]:
        d = r0['provs'][tag]
        of, nf = d['old_full8'], d['new_full8']
        L.append('| %s | %d | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |' % (
            tag, of['n'], of['OA'], nf['OA'], nf['OA'] - of['OA'], of['m9'], nf['m9'], nf['m9'] - of['m9']))
    n8_tot = sum(d['n_truth8'] for d in r0['provs'].values())
    if n8_tot == 0:
        L += ['',
              '> 说明：三省留出窗内真值=8（针阔混交林）点为 **0**（该类留出点全部落在黑/闽或省外窗），'
              '故本表与 §2 数值相同——保留以备案，两池在本次三省口径下无"8 类可比性"差异。', '']
    else:
        L += ['',
              '> 说明：真值 8 类点共 %d（三省），对旧池是"可命中"、对新池是"必错"；'
              '上表仅示该口径差异量级，**不得用于两池优劣判读**。' % n8_tot, '']
    # §4 多种子
    if len(out['runs']) > 1:
        L += ['## 4 多种子复核（doc38 纪律：<2pp 需多种子）', '',
              '| 省 | 口径 | ' + ' | '.join('seed %d Δ' % r['seed'] for r in out['runs']) +
              ' | 均值 | 极差 |', '|---|---|' + '---|' * (len(out['runs']) + 2)]
        for tag in ['合计'] + [t for t, _ in PROV2 if t in prov_idx]:
            ds = [r['combined'] if tag == '合计' else r['provs'][tag] for r in out['runs']]
            for key, lab in (('OA', '24 类 OA'), ('m9', '9 大类 OA')):
                vals = [d['delta'].get(key + '_pp') for d in ds]
                if any(v is None for v in vals):
                    L.append('| %s | %s | %s | — | — |' % (tag, lab, ' | '.join(['—'] * len(vals))))
                    continue
                L.append('| %s | %s | %s | %+.2f | %.2f |' % (
                    tag, lab, ' | '.join('%+.2f' % v for v in vals),
                    float(np.mean(vals)), float(np.max(vals) - np.min(vals))))
        L += ['']
    # §5 限制
    L += ['## 5 诚实性声明与限制', '']
    for s in out['limitations']:
        L.append('- %s' % s)
    L += ['- 两臂唯一变量是训练池；优化器/特征/留出/评估口径全部相同。',
          '- 结果方向若与"新池更好"的预期相反，本报告如实呈现（预注册判据先于结果写定）。',
          '']
    # §6 结论
    c = out['runs'][0]['combined']
    dl = c['delta']
    L += ['## 6 结论（seed %d，子采样口径）' % out['runs'][0]['seed'], '',
          '- 三省合计：24 类 OA %.4f → %.4f（Δ %+.4f = %+.1fpp，**%s**）；'
          '9 大类 OA %.4f → %.4f（Δ %+.1fpp，**%s**）。' % (
              c['old']['all']['OA'], c['new']['all']['OA'], dl['OA'], dl['OA_pp'], dl['judge_24'],
              c['old']['all']['m9'], c['new']['all']['m9'], dl['m9_pp'], dl['judge_9']),
          '- 分省判读：%s。' % '；'.join(
              '%s 24 类 %s（%+.1fpp）/ 9 大类 %s（%+.1fpp）' % (
                  t, out['runs'][0]['provs'][t]['delta']['judge_24'], out['runs'][0]['provs'][t]['delta']['OA_pp'],
                  out['runs'][0]['provs'][t]['delta']['judge_9'], out['runs'][0]['provs'][t]['delta']['m9_pp'])
              for t in [x for x, _ in PROV2 if x in prov_idx]),
          '- 非 fcs10 层 Δ：%+.1fpp（24 类 OA）/ %+.1fpp（9 大类）；fcs10 层：%+.1fpp / %+.1fpp。' % (
              c['delta_nonfcs10']['OA_pp'], c['delta_nonfcs10']['m9_pp'],
              c['delta_fcs10']['OA_pp'], c['delta_fcs10']['m9_pp']),
          '- 定位：本结果仅作第二次 GEE 生产**事前方向参照**（本地 RF 子采样，无 GEE）；'
          '生产口径数字以产成图后的 c4_eval_second 评估为准。']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
