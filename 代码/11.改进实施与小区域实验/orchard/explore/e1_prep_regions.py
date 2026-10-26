# -*- coding: utf-8 -*-
"""e1_prep_regions.py — 园地先验实验·区域准备与冻结注册（零 RF）

* 输入：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（新池，cl 已 24 码，**不再 remap**）
        F:/lc_work/v31_exp/data/yearly/eval_2023_w{2,3,4}.parquet（五窗开发参考）
* 规则源：doc70（20 分钟小区域实验规范 v1.0）；协调者指令（4 区×2 臂×3 种子；空间块从训练排除）
* 门槛：本脚本零训练/零预测；只做区域定义、空间块拆分、训练窗剔除与冻结注册
* 输出：<WS>/prereg/regions_frozen.json、<WS>/prereg/e1_block_split.csv、
        <WS>/explore/e1_window_counts.csv、<WS>/cache/<region>_train_window.parquet
* 用法：PYTHONDONTWRITEBYTECODE=1 python e1_prep_regions.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁')
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import c4_eval_second as C4
import p5common as PC
import v31_common as VC

POOL2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'
FEATS = ['A%02d' % i for i in range(64)]
YEAR = 2023
BLOCK = 0.25            # 空间块边长（度）；doc70 §6.1 允许 0.25/0.5
DEV_MOD = 60            # 稳定哈希 <60 → 开发块，其余确认块
N_LIMIT = 20000         # 每模型最多 20,000 点（doc70 §4.3）

# 区域注册：bbox 为地理小框（评价域）；tile 为所属生产 2°瓦；训练框 = tile ± 2°
REGIONS = [
    dict(rid='SC1', prov='四川省', ptag='川', win='w2',
         bbox=(106.0, 32.0, 107.0, 33.0), tile='T1607', tile_box=(105.0, 32.0, 107.0, 34.0)),
    dict(rid='SC2', prov='四川省', ptag='川', win='w2',
         bbox=(107.0, 32.0, 108.0, 33.0), tile='T1707', tile_box=(107.0, 32.0, 109.0, 34.0)),
    dict(rid='GD', prov='广东省', ptag='粤', win='w3',
         bbox=(112.0, 23.0, 113.0, 24.0), tile='T1902', tile_box=(111.0, 22.0, 113.0, 24.0)),
    dict(rid='NX', prov='宁夏回族自治区', ptag='宁', win='w4',
         bbox=(106.0, 38.0, 107.0, 39.0), tile='T1610', tile_box=(105.0, 38.0, 107.0, 40.0)),
]
ARMS = ['baseline', 'OCAP']
SEEDS = [7, 11, 23]


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def block_id(lon, lat, step=BLOCK):
    """0.25° 空间块 id：'bx_by'（bx=floor(lon/step), by=floor(lat/step)）。"""
    bx = np.floor(np.asarray(lon, float) / step).astype('int64')
    by = np.floor(np.asarray(lat, float) / step).astype('int64')
    return np.array(['%d_%d' % (a, b) for a, b in zip(bx, by)], dtype=object)


def stable_dev(bid):
    """稳定哈希分块：md5(bid) 首字节 %100 < DEV_MOD → 开发块。"""
    h = hashlib.md5(bid.encode('utf-8')).hexdigest()
    return int(h[:2], 16) % 100 < DEV_MOD


def split_blocks(ref, blocks):
    """稳定哈希定序 + 贪心配比：按 md5(block) 排序，累积参考点数 < 60% 的块归 dev，其余 conf。
    确定性、与结果无关；目标约 60/40 点数配比（doc70 §6.1）。返回 (dev_blocks, conf_blocks)。"""
    cnt = ref.block.value_counts().to_dict()
    order = sorted(blocks, key=lambda b: hashlib.md5(b.encode('utf-8')).hexdigest())
    total = int(len(ref))
    target = 0.6 * total
    dev, acc = [], 0
    for b in order:
        if acc < target or not dev:
            dev.append(b)
            acc += int(cnt.get(b, 0))
        else:
            break
    dev_set = set(dev)
    conf = [b for b in order if b not in dev_set]
    return dev, conf


def main():
    os.makedirs(os.path.join(WS, 'prereg'), exist_ok=True)
    os.makedirs(os.path.join(WS, 'cache'), exist_ok=True)
    os.makedirs(os.path.join(WS, 'explore'), exist_ok=True)
    names = VC.V31_NAMES()

    # ---- 参考点：五窗 eval，排除双池+错位，归省 ----
    frames = []
    for w in ['w2', 'w3', 'w4']:
        d = pd.read_parquet(os.path.join(EVAL_DIR, 'eval_2023_%s.parquet' % w),
                            columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + FEATS)
        d['win'] = w
        frames.append(d)
    ev = pd.concat(frames, ignore_index=True)
    keep, xd = C4.build_exclusions(ev)
    ev = ev[keep].reset_index(drop=True)
    prov_full, _ = C4.assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['pf'] = prov_full
    ev['t24'] = VC.to_v31(ev.class_new.to_numpy().astype('int64'))
    ev['isf'] = PC.is_fcs10(ev.src).to_numpy()
    ev['block'] = block_id(ev.lon.to_numpy(), ev.lat.to_numpy())
    emit('参考点 %d（排除明细 %s）' % (len(ev), xd))

    # ---- 池：整池只读一次 ----
    pool = pq.read_table(POOL2, columns=['point_id', 'lon', 'lat', 'cl', 'src'] + FEATS).to_pandas()
    emit('池装载 %d 点 / %d 类' % (len(pool), pool.cl.nunique()))
    assert pool.cl.min() >= 1 and pool.cl.max() <= 24, '池 cl 非 24 码空间'

    reg_out = {}
    blk_rows = []
    wc_rows = []
    for R in REGIONS:
        rid = R['rid']
        x0, y0, x1, y1 = R['bbox']
        tx0, ty0, tx1, ty1 = R['tile_box']
        tb = (tx0 - 2.0, ty0 - 2.0, tx1 + 2.0, ty1 + 2.0)   # tile ± 2°
        # 本区参考点（框内 + 目标省）
        mref = ((ev.win == R['win']) & (ev.pf == R['prov']) &
                (ev.lon >= x0) & (ev.lon < x1) & (ev.lat >= y0) & (ev.lat < y1))
        ref = ev[mref].copy()
        blocks = sorted(ref.block.unique().tolist())
        dev_blocks, conf_blocks = split_blocks(ref, blocks)
        n_dev = int(ref.block.isin(dev_blocks).sum())
        n_conf = int(ref.block.isin(conf_blocks).sum())
        # 训练窗（tile±2°）内池点
        mw = ((pool.lon >= tb[0]) & (pool.lon < tb[2]) & (pool.lat >= tb[1]) & (pool.lat < tb[3]))
        win = pool[mw].copy()
        n_win_before = len(win)
        win['block'] = block_id(win.lon.to_numpy(), win.lat.to_numpy())
        excl_blocks = set(dev_blocks) | set(conf_blocks)      # 注册：dev+conf 全排除
        keep_m = ~win.block.isin(excl_blocks)
        win_k = win[keep_m].copy()
        n_win_after = len(win_k)
        # 位置键零重叠自检（训练窗 vs 本区参考点）
        def pk(lon, lat, nd=6):
            a = np.round(np.asarray(lon, float) * 10 ** nd).astype('int64') + 200_000_000
            b = np.round(np.asarray(lat, float) * 10 ** nd).astype('int64') + 100_000_000
            return a * 400_000_000 + b
        inter = len(set(pk(win_k.lon.to_numpy(), win_k.lat.to_numpy()).tolist()) &
                    set(pk(ref.lon.to_numpy(), ref.lat.to_numpy()).tolist()))
        # 缓存（只留必要列）
        cache_fp = os.path.join(WS, 'cache', '%s_train_window.parquet' % rid)
        win_k[['point_id', 'lon', 'lat', 'cl', 'src'] + FEATS].reset_index(drop=True).to_parquet(cache_fp, index=False)
        # 参考点缓存（评价域；含 split/block/t24/isf）
        refc = ref[['row_id', 'lon', 'lat', 't24', 'isf', 'block', 'win'] + FEATS].copy()
        refc['split'] = np.where(refc.block.isin(dev_blocks), 'dev', 'conf')
        refc_fp = os.path.join(WS, 'cache', '%s_ref.parquet' % rid)
        refc.reset_index(drop=True).to_parquet(refc_fp, index=False)
        # 类计数
        cnt = win_k.cl.value_counts().sort_index()
        reg_out[rid] = dict(
            rid=rid, prov=R['prov'], ptag=R['ptag'], win=R['win'],
            bbox=list(R['bbox']), tile=R['tile'], tile_box=list(R['tile_box']),
            training_bbox=list(tb), refn=int(len(ref)),
            refn_nonfcs10=int((~ref.isf).sum()), refn_fcs10=int(ref.isf.sum()),
            ref_by_class={int(c): int(v) for c, v in ref.t24.value_counts().sort_index().items()},
            ref_by_class_dev={int(c): int(v) for c, v in
                              ref[ref.block.isin(dev_blocks)].t24.value_counts().sort_index().items()},
            ref_by_class_conf={int(c): int(v) for c, v in
                               ref[ref.block.isin(conf_blocks)].t24.value_counts().sort_index().items()},
            n_dev_cl4=int((ref[ref.block.isin(dev_blocks)].t24 == 4).sum()),
            n_conf_cl4=int((ref[ref.block.isin(conf_blocks)].t24 == 4).sum()),
            n_conf_nonfcs10=int((~ref[ref.block.isin(conf_blocks)].isf).sum()),
            n_blocks=len(blocks), n_dev_blocks=len(dev_blocks), n_conf_blocks=len(conf_blocks),
            n_dev_pts=n_dev, n_conf_pts=n_conf, dev_ratio=round(n_dev / max(len(ref), 1), 3),
            block_step=BLOCK, dev_mod=DEV_MOD,
            dev_blocks=dev_blocks, conf_blocks=conf_blocks,
            exclude_blocks_rule='dev+conf (本轮空间留出设计；与生产 classify 不同)',
            n_window_before=int(n_win_before), n_window_after=int(n_win_after),
            n_window_excluded=int(n_win_before - n_win_after),
            n_train_limit=N_LIMIT, window_class_counts={int(c): int(v) for c, v in cnt.items()},
            poskey_overlap_train_ref=int(inter), cache=cache_fp, ref_cache=refc_fp,
        )
        for b in blocks:
            sub = ref[ref.block == b]
            blk_rows.append(dict(rid=rid, block=b, split=('dev' if b in dev_blocks else 'conf'),
                                 n=int(len(sub)), n_nonfcs10=int((~sub.isf).sum()),
                                 n_cl4=int((sub.t24 == 4).sum()), n_cl2=int((sub.t24 == 2).sum()),
                                 n_cl6=int((sub.t24 == 6).sum()), n_cl1=int((sub.t24 == 1).sum())))
        wc_rows.append(dict(rid=rid, win=R['win'], refn=int(len(ref)), n_win_before=int(n_win_before),
                            n_win_after=int(n_win_after), n_excl_blocks=len(excl_blocks),
                            cl2_in_window=int(cnt.get(2, 0)), poskey_overlap=int(inter)))
        emit('%s %s refn=%d(dev %d/conf %d) 训练窗 %d→%d (剔 %d, 块 %d) 位置键重叠=%d 类2窗内=%d' % (
            rid, R['tile'], len(ref), n_dev, n_conf, n_win_before, n_win_after,
            n_win_before - n_win_after, len(excl_blocks), inter, cnt.get(2, 0)))

    # ---- 冻结注册 JSON ----
    reg = dict(
        spec='doc70 实施规范 v1.0（20 分钟小区域实验）',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        year=YEAR, class_space='v31 24 类码（新池 cl 已 24 码，**不 remap**）',
        pool=dict(path=POOL2, n=int(len(pool)), classes=sorted(int(c) for c in pool.cl.unique())),
        arms={
            'baseline': '池原样局部均匀随机抽样：rng(seed).permutation(窗内点) 取前 20000（复现生产 randomColumn 机制，非逐点复制）',
            'OCAP': '同一 permutation 顺序接受：类 2 园地最多 200（=1%×20000），其余类不设限，补足至 20000；不足 200 保留全部',
        },
        seeds=SEEDS, n_train_limit=N_LIMIT,
        rf_params=dict(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000, n_jobs=10,
                       note='sklearn RF ≠ GEE smileRF（本地筛选器）'),
        deadline_s=1200, unit='region × arm × 3 seeds',
        block_step=BLOCK, dev_mod=DEV_MOD, dev_ratio_target=0.6,
        spatial_holdout='训练窗内剔除本区全部 0.25° 块（dev+conf）；如剔后 <20000 用实际 N（不偷回）',
        eval='同点评价：全点/fcs10/非fcs10；OA24、macro9、逐类 F1/PA/UA；dev+conf 分报，conf 只评一次',
        uncertainty='0.25° 块 bootstrap 10000 次（配对 Δ）；报告有效块数/点数/种子均值±sd',
        main_gate=dict(
            primary_stratum='非FCS10 内部确认层（conf blocks ∩ 非fcs10）',
            primary_metric='macro9 OA（另报 OA24）',
            target_class_f1='目标类 F1 亦在非fcs10 确认层判读（川=类4 常绿阔叶林；粤=类4/类5；宁=类22 裸地/类3 灌溉耕地）',
            full_and_fcs10='全点与 FCS10 层完整报告，不得用以掩盖主门',
            insufficient_support_rule='某小框确认层目标类 n<30 → 判"支持不足"，不得改看全点层或调主门/阈值来制造 cap 涨分；如实标 PROMISING/REJECT/支持不足',
            confirm_once='确认层只评一次，不用于挑臂或调阈值',
        ),
        gates=dict(target_dF1=0.03, group_noninferior_d_pt=-0.01, group_noninferior_ci=-0.02,
                   new_regression_dF1=-0.05),
        regions=reg_out,
    )
    fp = os.path.join(WS, 'prereg', 'regions_frozen.json')
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1, default=str)
    pd.DataFrame(blk_rows).to_csv(os.path.join(WS, 'prereg', 'e1_block_split.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(wc_rows).to_csv(os.path.join(WS, 'explore', 'e1_window_counts.csv'),
                                 index=False, encoding='utf-8-sig')
    # 代码/注册哈希
    h = hashlib.md5(open(fp, 'rb').read()).hexdigest()
    emit('冻结注册 → %s (md5 %s)' % (fp, h))
    emit('完成')


if __name__ == '__main__':
    main()
