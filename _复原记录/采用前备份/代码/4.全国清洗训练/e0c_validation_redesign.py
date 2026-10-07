# -*- coding: utf-8 -*-
"""
e0c_validation_redesign.py — P1.2 验证池重构 v2：空间块留出（评审阶段H2 本意）
* 背景1（e0 诊断）：旧验证池 99.6% 点距最近训练点 <5km → OA 必然乐观
* 背景2（实测）：对全国训练集做逐点 ≥10km 隔离仅剩 283 候选/6 类（全在无人区）
  —— 1D 距离规则在密集全国样本上结构性不可行，必须用空间块留出
* 规则：20km 等面积块(Albers)整块从训练中扣除；块内侵蚀 5km 的内部点作验证
  （保证验证点到任何参与训练的点 ≥5km，满足评审 H2 的 5-10km 要求）；
  块选择按 类别稀缺度 升序贪心，每类目标 min(max(30, 5%), 200) 点，
  生态区兜底各 ≥1 块；总留出 ≤8% 训练点。
* 配套产物: holdout_blocks.json —— 评估训练时必须剔除的块键清单（e3 消费）
* 用法: python e0c_validation_redesign.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from pyproj import Transformer

PROJ = r'Z:/Mywork/论文/中国土地覆盖数据'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '5.样本重建'))
import s0_conf as C
from s1_geom import province_of
from lc_conf import class_name

WORK = os.path.join(PROJ, '样本底座')  # unused guard
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
OUT = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2.parquet')
BLOCK = 25000      # 25km 等面积块（侵蚀 5km 后内部仍 ≥5km 隔离）
ERODE = 5000       # 块内侵蚀 5km 只留内部点作验证
HOLDOUT_BUDGET = 0.08
GRID = 0.05        # 验证点之间网格去重（≥3.6km 间距）
TARGET_MIN, TARGET_MAX, TARGET_RATIO = 50, 500, 0.06

TR = Transformer.from_crs('EPSG:4326',
                          '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 '
                          '+datum=WGS84 +units=m', always_xy=True)

ZONE_OF_PROV = {
    '黑龙江省': '东北', '吉林省': '东北', '辽宁省': '东北',
    '北京市': '华北', '天津市': '华北', '河北省': '华北', '山西省': '华北',
    '山东省': '华北', '河南省': '华北',
    '上海市': '长江中下游', '江苏省': '长江中下游', '安徽省': '长江中下游',
    '湖北省': '长江中下游', '湖南省': '长江中下游', '江西省': '长江中下游', '浙江省': '长江中下游',
    '福建省': '华南', '广东省': '华南', '广西壮族自治区': '华南', '海南省': '华南',
    '台湾省': '华南', '香港特别行政区': '华南', '澳门特别行政区': '华南',
    '重庆市': '西南', '四川省': '西南', '贵州省': '西南', '云南省': '西南',
    '西藏自治区': '青藏', '青海省': '青藏',
    '新疆维吾尔自治区': '西北', '甘肃省': '西北', '宁夏回族自治区': '西北',
    '陕西省': '西北', '内蒙古自治区': '西北',
}

def main():
    t0 = time.time()
    train = pd.read_parquet(os.path.join(WORKR, 'r7_train.parquet'))
    print(f'train {len(train):,}', flush=True)
    # 省份兜底（121 等抽稀路径 province 为空）
    miss = train['province'].isna() | (train['province'].astype(str) == '未匹配')
    if miss.any():
        train.loc[miss, 'province'] = province_of(train.loc[miss, 'lon'].to_numpy(),
                                                  train.loc[miss, 'lat'].to_numpy())
    train['zone'] = train['province'].map(ZONE_OF_PROV).fillna('其他')

    X, Y = TR.transform(train.lon.to_numpy(), train.lat.to_numpy())
    bk = (np.floor(X / BLOCK).astype(int) * 100000 +
          np.floor(Y / BLOCK).astype(int))
    # 块内侵蚀网格键（1km 粒度判断是否在内部）
    inner = ((X - np.floor(X / BLOCK) * BLOCK > ERODE) &
             (np.ceil(X / BLOCK) * BLOCK - X > ERODE) &
             (Y - np.floor(Y / BLOCK) * BLOCK > ERODE) &
             (np.ceil(Y / BLOCK) * BLOCK - Y > ERODE))
    train['bk'] = bk
    train['inner'] = inner

    # 块 → (类计数, 总数, zone)
    blk_class = train.groupby(['bk', 'class_new']).size().unstack(fill_value=0)
    blk_total = train.groupby('bk').size()
    blk_zone = train.groupby('bk')['zone'].agg(lambda s: s.value_counts().index[0])
    blocks = blk_total.index.to_numpy()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(blocks))
    cls_arr = blk_class.loc[blocks].to_numpy()
    col_of = {c: i for i, c in enumerate(blk_class.columns)}
    tot_arr = blk_total.loc[blocks].to_numpy()
    zone_arr = blk_zone.loc[blocks].to_numpy()
    print(f'含样本块: {len(blocks):,}', flush=True)

    n_by_c = train.class_new.value_counts()

    def greedy_select(covered, sel, need_classes, budget_left, use_inner=False):
        """按类稀缺度升序选块；need_classes=None 时全类；use_inner=用侵蚀后计数判断"""
        classes = sorted(n_by_c.index) if need_classes is None else need_classes
        used = int(tot_arr[np.array(sorted(sel))].sum()) if sel else 0
        for c in classes:
            ci = col_of[c]
            if use_inner:
                have = inner_cls[covered, ci].sum() if covered.any() else 0
            else:
                have = cls_arr[covered, ci].sum() if covered.any() else 0
            target = min(max(TARGET_MIN, int(TARGET_RATIO * n_by_c[c])), TARGET_MAX)
            if have >= target:
                continue
            cand = np.where((cls_arr[:, ci] > 0) & (~covered))[0]
            cand = cand[np.argsort(-cls_arr[cand, ci] +
                                   rng.random(len(cand)) * 0.999)]
            for bi in cand:
                if used + int(tot_arr[bi]) > budget_left:
                    break
                sel.append(bi)
                covered[bi] = True
                used += int(tot_arr[bi])
                have += int(inner_cls[bi, ci]) if use_inner else int(cls_arr[bi, ci])
                if have >= target:
                    break
        return used

    covered = np.zeros(len(blocks), dtype=bool)
    sel = []
    budget = HOLDOUT_BUDGET * len(train)
    used = greedy_select(covered, sel, None, budget)
    # 第二轮：侵蚀后不足目标的类补块（按侵蚀后计数）
    sel_arr = np.array(sorted(sel), dtype=int)
    inner_arr = train[train.bk.isin(set(blocks[sel_arr].tolist()))].groupby(
        'bk')['inner'].sum().reindex(blocks[sel_arr]).fillna(0).to_numpy()
    inner_cls = np.zeros((len(blocks), cls_arr.shape[1]), dtype=np.int64)
    sub = train[train.bk.isin(set(blocks.tolist()))]
    ic = sub[sub.inner].groupby(['bk', 'class_new']).size().unstack(fill_value=0)
    for c in ic.columns:
        if c in col_of:
            inner_cls[np.isin(blocks, ic.index.to_numpy()), col_of[c]] = ic[c].reindex(
                blocks[np.isin(blocks, ic.index.to_numpy())]).fillna(0).to_numpy()
    short = [c for c in sorted(n_by_c.index)
             if inner_cls[sel_arr, col_of[c]].sum() <
             min(max(TARGET_MIN, int(TARGET_RATIO * n_by_c[c])), TARGET_MAX) * 0.8]
    if short and used < budget:
        used = greedy_select(covered, sel, short, budget, use_inner=True)
    if used > budget:
        print(f'预算用尽', flush=True)
    # 生态区兜底：每区至少 1 块
    for z in set(ZONE_OF_PROV.values()) | {'其他'}:
        if not covered[zone_arr == z].any():
            zd = np.where(zone_arr == z)[0]
            if len(zd):
                bi = zd[np.argmax(cls_arr[zd].sum(1))]
                sel.append(int(bi)); covered[bi] = True; used += int(tot_arr[bi])
    sel_blocks = blocks[np.array(sorted(sel))]

    holdout = train[train.bk.isin(set(sel_blocks.tolist()))]
    valp = holdout[holdout.inner].copy()
    # 验证点之间 0.05° 网格去重
    valp['gk'] = (np.floor(valp.lon / GRID).astype(int) * 100000 +
                  np.floor(valp.lat / GRID).astype(int))
    valp = valp.groupby('gk', group_keys=False).apply(
        lambda g: g.iloc[rng.integers(len(g))]).reset_index(drop=True)

    # 旧外部验证点不满足隔离，单独存档（不混入主验证池）
    old_val = pd.read_parquet(C.R1_VALID,
                              columns=['lon', 'lat', 'class_new', 'src', 'src_conf', 'year'])
    keep_old = old_val[old_val.class_new > 0].copy()
    keep_old.to_parquet(OUT.replace('validation_pool_v2', 'validation_external_legacy'),
                        index=False)
    valp['val_src'] = 'train_block_holdout'
    valp['bk_key'] = valp.bk
    val = valp[['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf',
                'zone', 'province', 'val_src', 'bk_key']].copy()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    val.to_parquet(OUT, index=False)

    # holdout 块键清单（评估训练必须剔除）
    jout = OUT.replace('.parquet', '_holdout_blocks.json')
    json.dump({'rule': '评估训练时剔除这些 20km Albers 块内全部训练点；'
                       '验证点=块内侵蚀5km内部点，到训练点距离≥5km',
               'block_m': BLOCK, 'erode_m': ERODE,
               'n_blocks': int(len(sel_blocks)),
               'block_keys': sel_blocks.tolist()},
              open(jout, 'w'), indent=1)

    rep = {
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'blocks_selected': int(len(sel_blocks)),
        'train_in_holdout_blocks': int(len(holdout)),
        'holdout_ratio': round(len(holdout) / len(train), 4),
        'val_from_train': int(len(valp)),
        'val_external_legacy': int(len(keep_old)),
        'val_total': int(len(val)),
        'classes_covered': int(val.class_new.nunique()),
        'by_class': {class_name(int(c)): int(n) for c, n in
                     val.class_new.value_counts().sort_index().items()},
        'by_zone': {k: int(v) for k, v in val.zone.value_counts().items()},
        'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(rep, open(OUT.replace('.parquet', '_summary.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: rep[k] for k in
                      ['blocks_selected', 'train_in_holdout_blocks', 'holdout_ratio',
                       'val_from_train', 'val_total', 'classes_covered']},
                     ensure_ascii=False, indent=2))
    print('by_class:', rep['by_class'])
    print('by_zone:', rep['by_zone'])
    print('输出:', OUT)

if __name__ == '__main__':
    main()
