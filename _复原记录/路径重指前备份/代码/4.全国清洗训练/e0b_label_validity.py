# -*- coding: utf-8 -*-
"""
e0b_label_validity.py — P1.1 标签时间有效期先验（评审阶段D2，只读 r7 不回写）
* 为 r7_train 每点生成 valid_from / valid_to（该标签可用的嵌入年份区间）
  + qc_scope（yearly=湿地/灌丛等需逐年QC）
* 规则为评审 D2 初始先验，最终由 P2 年度嵌入 QC 校准
* 输出: 数据/本地处理/样本重建/r7_train_validity.parquet + r7_validity_summary.json
* 用法: python e0b_label_validity.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/中国土地覆盖数据'
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
TRAIN = os.path.join(WORKR, 'r7_train.parquet')
OUT = os.path.join(WORKR, 'r7_train_validity.parquet')

# src 前缀 → (valid_from_offset, valid_to_offset) 相对标签年份；None=相对规则见后
# 评审 D2 初始先验：底座=长期稳定候选；FCS10-2023=2022-2024；2020专题=2019-2021；
# 东北作物=原年份±0；园地/湿地专题=对应年份±1
RULES = {
    'v2_glc_fcs30d2020_7yr_stable': (2017, 2024),   # 七年稳定底座：长期候选，逐年QC把关
    'glc_fcs10_2023':               (2022, 2024),   # FCS10 2023 单年教师（含 shrub/band130 变体）
    'gisa_gisd':                    (2017, 2024),   # 不透水面（首城市化年，地物持续）
    'gwl_fcs30_2020_raster':        (2019, 2021),
    'gwl_fcs30_stable':             (2017, 2022),   # 2000-2022 稳定湿地
    'worldcereal':                  (-1, 1),
    'teamap':                       (-1, 1),
    'citrus':                       (-1, 1),
    'esri_nat':                     (-1, 1),
    'ne_crops':                     (0, 0),         # 东北作物逐年份标签
    'aomc':                         (-1, 1),
    'eglc':                         (-1, 1),
    'galf':                         (-1, 1),
    'gpw_grass':                    (-1, 1),
    'glc12':                        (-1, 1),
    'rubber':                       (-1, 1),
    'china_saltmarsh':              (-1, 1),
}
# 逐点逐年 QC 优先类（湿地亚类+灌丛，评审 §二.4/5）
YEARLY_QC_CLASSES = {180, 181, 182, 183, 184, 185, 186, 120, 121}
YEAR_LO, YEAR_HI = 2017, 2024

def rule_for(src):
    for pfx, r in RULES.items():
        if src.startswith(pfx):
            return r
    return None

def main():
    t0 = time.time()
    df = pd.read_parquet(TRAIN, columns=['src', 'year', 'class_new'])
    n = len(df)
    vf = np.full(n, -1, dtype=np.int16)
    vt = np.full(n, -1, dtype=np.int16)
    src = df.src.to_numpy()
    yr = df.year.to_numpy().astype(int)
    unmatched = {}
    for i in range(n):
        s = str(src[i]); y = int(yr[i])
        r = rule_for(s)
        if r is None:
            unmatched[s] = unmatched.get(s, 0) + 1
            lo, hi = y - 1, y + 1           # 兜底：标签年 ±1
        elif r[0] > 1900:                    # 绝对年份规则
            lo, hi = r
        else:                                # 相对偏移规则
            lo, hi = y + r[0], y + r[1]
        vf[i] = max(lo, YEAR_LO)
        vt[i] = min(hi, YEAR_HI)
    ok = vf <= vt
    df['valid_from'] = vf
    df['valid_to'] = vt
    df['qc_scope'] = np.where(df.class_new.isin(YEARLY_QC_CLASSES), 'yearly', 'standard')
    # 无效区间（year 在 2017-2024 之外且窗口被裁没）标记
    df.loc[~ok, ['valid_from', 'valid_to']] = 0   # 0=任何生产年都不可用
    out = pd.read_parquet(TRAIN)
    out['valid_from'] = df.valid_from.to_numpy()
    out['valid_to'] = df.valid_to.to_numpy()
    out['qc_scope'] = df.qc_scope.to_numpy()
    out.to_parquet(OUT, index=False)

    rep = {
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'n': int(n),
        'usable_any_year': int(ok.sum()),
        'unusable': int((~ok).sum()),
        'yearly_qc_points': int((df.qc_scope == 'yearly').sum()),
        'by_src_rule': {},
        'unmatched_src_fallback_y1': unmatched,
        'window_hist': {f'{a}-{b}': int(c) for (a, b), c in
                        pd.Series(list(zip(vf.tolist(), vt.tolist()))).value_counts()
                        .sort_index().head(20).items()},
        'elapsed_min': round((time.time() - t0) / 60, 1)}
    # 每规则点数
    for pfx in RULES:
        m = df.src.str.startswith(pfx)
        if m.any():
            rep['by_src_rule'][pfx] = {
                'n': int(m.sum()),
                'median_window': f"{int(df.loc[m, 'valid_from'].median())}-"
                                 f"{int(df.loc[m, 'valid_to'].median())}"}
    json.dump(rep, open(OUT.replace('.parquet', '_summary.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: rep[k] for k in ['n', 'usable_any_year', 'unusable',
                                          'yearly_qc_points']}, ensure_ascii=False))
    for k, v in rep['by_src_rule'].items():
        print(f"  {k:32s} {v['n']:>9,}  window {v['median_window']}")
    if unmatched:
        print('  兜底(±1)源:', unmatched)
    print('输出:', OUT)

if __name__ == '__main__':
    main()
