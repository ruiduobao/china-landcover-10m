# -*- coding: utf-8 -*-
"""k4_calib57.py — 用 doc57 的 454 个随机点分层判读结果，量测可迁移到全国的分层精度

* 输入：data/shrub/judged_all.csv（doc57 判读）+ data/shrub_nat/judged_feats.parquet（k2 特征）
* 为什么只用 doc57：后期 expA/A2/B/C/C2/C3/C4/gsnx 判读集是为森林/湿地等**别的科学问题**抽的样框，
        其点上"DW 灌丛标签"与"是否灌丛"的关系受样框选择偏差影响（实测 289 个 dwl==6 点里只有 8 个
        真灌丛），**不能**用来估计候选规则的精度；doc57 是"随机点→按证据分层→判读"的无偏设计。
* 规则源（预注册，口径与 doc57 完全一致）：
        L1 = dwl==6 且 (wc20 或 lc20)          —— 双源一致
        L2 = dwl==6 且 dwp≥0.4                 —— 单源高概率
        L3 = (wc20 或 lc20) 且 dwl!=6          —— 单源低共识
        L4 = dwl==6 且 dwp<0.4                 —— 单源低概率
        OSM/FCS10 = 对应框（本脚本不管，另按源统计）
        物理屏幕变体：S1 rh98∈[1.5,8]；S2 rh98_mx11<8；S3 tc<30
* 门槛：每层报 n/命中/precision/Wilson CI；n<20 只报合并级
* 输出：data/shrub_nat/calib57.json + data/shrub_nat/calib57.md
* 用法：python k4_calib57.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def strat_of(df):
    dwp, dwl = df['dwp'].to_numpy(float), df['dwl'].to_numpy(float)
    multi = ((df['wc20'] > 0) | (df['lc20'] > 0) | (df['mcd'] > 0)).to_numpy()
    out = np.array(['其他'] * len(df), dtype=object)
    is6 = dwl == 6
    out[np.where(is6 & multi)] = 'L1'
    out[np.where(is6 & ~multi & (dwp >= 0.4))] = 'L2'
    out[np.where((~is6) & multi)] = 'L3'
    out[np.where(is6 & ~multi & (dwp < 0.4))] = 'L4'
    return out


def main():
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    d57['point_id'] = d57['point_id'].astype(str)
    # doc57 的 point_id 形如 SH-四川-FCS10-0000；judged_pool 里被改写成 JP-doc57-xxxx，用坐标回连
    key57 = set(zip(np.round(d57['lon'].to_numpy(float), 6), np.round(d57['lat'].to_numpy(float), 6)))
    k_all = zip(np.round(j['lon'].to_numpy(float), 6), np.round(j['lat'].to_numpy(float), 6))
    j['is57'] = [k in key57 for k in k_all]
    sub = j[j['is57']].copy()
    emit('doc57 判读点 %d / 454（坐标回连）' % len(sub))
    lab = {k: v for k, v in zip(zip(np.round(d57['lon'].to_numpy(float), 6),
                                    np.round(d57['lat'].to_numpy(float), 6)),
                                zip(d57['Q1_shrub'], d57['Q2_evergreen']))}
    sub['is_shrub'] = [lab[k][0] for k in zip(np.round(sub['lon'].to_numpy(float), 6),
                                              np.round(sub['lat'].to_numpy(float), 6))]
    sub['ever'] = [lab[k][1] for k in zip(np.round(sub['lon'].to_numpy(float), 6),
                                          np.round(sub['lat'].to_numpy(float), 6))]
    jd = sub[sub['is_shrub'].isin(['是', '否'])].copy()
    jd['stratum'] = strat_of(jd)
    y = (jd['is_shrub'] == '是').to_numpy()
    emit('可判读 %d，其中灌丛 %d' % (len(jd), y.sum()))

    def phys(df, use_s1=True, use_s2=True, use_s3=True):
        r98 = df['rh98'].to_numpy(float)
        mx = df['rh98_mx11'].to_numpy(float)
        tc = df['tc'].to_numpy(float)
        m = np.ones(len(df), bool)
        if use_s1:
            m &= (np.isfinite(r98) & (r98 >= 1.5) & (r98 <= 8.0)) | ~np.isfinite(r98)
        if use_s2:
            m &= (~np.isfinite(mx)) | (mx < 8.0)
        if use_s3:
            m &= (~np.isfinite(tc)) | (tc < 30)
        return m

    out = dict(n=int(len(jd)), n_pos=int(y.sum()), strata={}, phys={})
    lines = ['# doc57 分层精度校准（可迁移到全国的唯一无偏设计） %s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 校准点：doc57 454 个随机点分层判读（坐标回连 %d，可判读 %d，灌丛 %d）' % (len(sub), len(jd), y.sum()),
             '- 分层口径与 doc57 原文一致（dwl==6 为 DW 众数为灌丛）；物理屏幕见 `phys()`',
             '', '## 分层精度', '', '| 层 | n | 灌丛 | precision | 95%CI |', '|---|---|---|---|---|']
    for s in ('L1', 'L2', 'L3', 'L4', '其他'):
        m = (jd['stratum'] == s).to_numpy()
        n, k = int(m.sum()), int(y[m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        out['strata'][s] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                wilson=[lo, hi])
        lines.append('| %s | %d | %d | %s | [%.2f,%.2f] |' % (
            s, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi))
        emit('%s：n=%d 灌=%d precision=%s' % (s, n, k, ('%.3f' % prec) if prec is not None else '—'))
    lines += ['', '## 物理屏幕的增益（在 L1/L2/L4 上叠加）', '',
              '| 规则 | n | 灌丛 | precision | 95%CI | 相对原始层 |', '|---|---|---|---|---|---|']
    for s in ('L1', 'L2', 'L4'):
        base = (jd['stratum'] == s).to_numpy()
        for tag, args in (('+S1', (True, False, False)), ('+S1+S2+S3', (True, True, True)),
                          ('+S2+S3', (False, True, True))):
            m = base & phys(jd, *args)
            n, k = int(m.sum()), int(y[m].sum())
            if n == 0:
                continue
            prec = k / n
            lo, hi = wilson(k, n)
            b_n, b_k = int(base.sum()), int(y[base].sum())
            b_p = b_k / b_n if b_n else 0
            out['phys']['%s%s' % (s, tag)] = dict(n=n, shrub=k, precision=round(prec, 4),
                                                  wilson=[lo, hi], base=round(b_p, 4))
            lines.append('| %s%s | %d | %d | %.3f | [%.2f,%.2f] | %+.3f |' % (
                s, tag, n, k, prec, lo, hi, prec - b_p))
            emit('%s%s：n=%d 灌=%d precision=%.3f（原层 %.3f）' % (s, tag, n, k, prec, b_p))
    json.dump(out, open(os.path.join(OUTD, 'calib57.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'calib57.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ calib57.json + calib57.md')


if __name__ == '__main__':
    main()
