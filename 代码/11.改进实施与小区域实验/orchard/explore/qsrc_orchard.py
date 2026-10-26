# -*- coding: utf-8 -*-
"""qsrc_orchard.py — 园地质量来源 helper（**仅元数据统计，零 RF、零新预测**）

* 输入：乔灌园地v7_2023_WGS84.gpkg（属性表 uid/lon/lat/subtype/grade/srcset/ct_prov/ct_km）
        F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（cl==2 的 point_id 对 gpkg uid，100% 命中）
        <WS>/prereg/regions_frozen.json（4 区训练窗 bbox）
* 规则源：doc69 §7.2（园地优先可核实管理地块；不把其他区域源质量等级迁移）；
        08 报告（真园率：OSM 27.6% / 柑橘 13.8% / 茶 6.2% / GLC 5.9% / 苹果 0%；C_专题产品(存疑)=柑橘）
* 门槛：本脚本只输出分布与可用计数，**不做任何抽样/训练/预测**；规则为"建议"，待主 agent 批准
* 输出：<WS>/orchard/qsrc/orchard_grade_distribution.{csv,md}、
        <WS>/orchard/qsrc/orchard_window_availability.csv、orchard_qrule_proposal.json
* 用法：PYTHONDONTWRITEBYTECODE=1 python qsrc_orchard.py
"""
import os, sys, json, time, sqlite3
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True

GPKG = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg'
POOL2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
TABLE = '乔灌园地v7_2023_WGS84'

# 建议质量规则（基于 08 报告判读证据，非臆造"全国 A 最高"）
# Q1（最小、可辩护）：排除 grade == 'C_专题产品(存疑)'（= 柑橘 2020，判读真园率 13.8% 且源本身标"存疑"）
# Q2（更严）：排除全部 C_* 与 单产品/专题产品 B（茶/柑橘等单产品源）
QRULES = {
    'Q1_minimal': dict(
        desc="排除 grade == 'C_专题产品(存疑)'（柑橘 2020；08 报告判读真园率 13.8%，源标'存疑'）",
        exclude_grades=['C_专题产品(存疑)'], exclude_srcset=['citrus']),
    'Q2_strict': dict(
        desc="排除全部 C_* 与单产品 B（C_专题产品(存疑)/C_单产品/B(专题产品)/B(单产品)）——仅留 A_* 与多源 B",
        exclude_grades=['C_专题产品(存疑)', 'C_单产品', 'B(专题产品)', 'B(单产品)'], exclude_srcset=[]),
}


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def load_gpkg_meta():
    c = sqlite3.connect(GPKG)
    cols = [r[1] for r in c.execute('pragma table_info("%s")' % TABLE)]
    keep = [x for x in ['uid', 'lon', 'lat', 'subtype', 'grade', 'srcset', 'ct_prov', 'ct_km'] if x in cols]
    df = pd.read_sql_query('select %s from "%s"' % (','.join(keep), TABLE), c)
    c.close()
    return df, cols


def main():
    outdir = os.path.join(WS, 'orchard', 'qsrc')
    os.makedirs(outdir, exist_ok=True)
    reg = json.load(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), encoding='utf-8'))

    g, cols = load_gpkg_meta()
    emit('gpkg 元数据 %d 行，列=%s' % (len(g), cols))

    # 全国分布
    L = ['# 园地质量来源分布（v7 2023 gpkg，元数据；%d 点）' % len(g), '']
    for col in ['grade', 'subtype', 'srcset']:
        if col not in g:
            continue
        vc = g[col].value_counts()
        L += ['## %s' % col, '', '| 值 | n | 占比 |', '|---|---|---|']
        for k, v in vc.items():
            L.append('| %s | %d | %.2f%% |' % (k, v, 100 * v / len(g)))
        L.append('')
    # grade × subtype 交叉（核心：哪些 grade 对应哪个作物源）
    if 'grade' in g and 'subtype' in g:
        ct = pd.crosstab(g['grade'], g['subtype'])
        ct.to_csv(os.path.join(outdir, 'orchard_grade_x_subtype.csv'), encoding='utf-8-sig')
        L += ['## grade × subtype 交叉（点）', '', ct.to_markdown(), '']
    g[['grade', 'subtype', 'srcset']].value_counts().rename('n').reset_index().to_csv(
        os.path.join(outdir, 'orchard_grade_distribution.csv'), index=False, encoding='utf-8-sig')

    # 池 cl2 ↔ gpkg uid 对齐（核 100%）
    import pyarrow.parquet as pq
    p = pq.read_table(POOL2, columns=['point_id', 'lon', 'lat', 'cl']).to_pandas()
    p2 = p[p.cl == 2]
    uid_map = g.set_index(g.uid.astype(str))
    hit = p2.point_id.astype(str).isin(set(uid_map.index))
    L += ['## 池 cl2 ↔ gpkg uid 对齐', '',
          '- 池 cl2 n=%d；命中 gpkg uid=%d（%.2f%%）' % (len(p2), int(hit.sum()), 100 * hit.mean()), '']
    pool_meta = uid_map.reindex(p2.point_id.astype(str).to_numpy()).reset_index(drop=True)
    pool_meta = pool_meta.assign(lon=p2.lon.to_numpy(), lat=p2.lat.to_numpy())

    # 4 区训练窗可用计数（元数据；只统计窗内池 cl2 点）
    rows = []
    for rid, R in reg['regions'].items():
        x0, y0, x1, y1 = R['training_bbox']
        m = ((pool_meta.lon >= x0) & (pool_meta.lon < x1) &
             (pool_meta.lat >= y0) & (pool_meta.lat < y1))
        sub = pool_meta[m]
        rec = dict(region=rid, tile=R['tile'], training_bbox=R['training_bbox'],
                   n_orchard_in_window=int(len(sub)))
        for gr, v in sub.grade.value_counts().items():
            rec['grade::%s' % gr] = int(v)
        for qn, q in QRULES.items():
            excl = sub.grade.isin(q['exclude_grades']) | (sub.srcset.isin(q['exclude_srcset']) if q['exclude_srcset'] else False)
            rec['avail_%s' % qn] = int((~excl).sum())
        rows.append(rec)
    av = pd.DataFrame(rows)
    av.to_csv(os.path.join(outdir, 'orchard_window_availability.csv'), index=False, encoding='utf-8-sig')
    L += ['## 4 区原训练窗园地可用计数（元数据）', '',
          '| 区 | 窗内园地 | Q1 可用 | Q2 可用 |', '|---|---|---|---|']
    for r in rows:
        L.append('| %s | %d | %d | %d |' % (r['region'], r['n_orchard_in_window'],
                                            r.get('avail_Q1_minimal', 0), r.get('avail_Q2_strict', 0)))
    L.append('')
    L += ['## 各 grade 全国分布（供规则选择）', '']
    for k, v in g.grade.value_counts().items():
        L.append('- `%s`：%d' % (k, v))
    L.append('')
    open(os.path.join(outdir, 'orchard_grade_distribution.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    # 规则建议 JSON
    prop = dict(
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        evidence='08 报告判读真园率：OSM 27.6% / 柑橘 13.8% / 茶 6.2% / GLC 5.9% / 苹果 0%；'
                 'doc69 §7.2：不把其他区域源质量等级迁移；C_专题产品(存疑) 即柑橘。',
        note='以下为**建议规则，待主 agent 批准**；本脚本不训练、不预测。',
        rules=QRULES,
        recommended='Q1_minimal',
        national_grade_counts={k: int(v) for k, v in g.grade.value_counts().items()},
        window_availability=rows,
        pool_cl2_gpkg_uid_hit=round(float(hit.mean()), 4),
    )
    json.dump(prop, open(os.path.join(outdir, 'orchard_qrule_proposal.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('→ orchard/qsrc/orchard_grade_distribution.{md,csv} + orchard_qrule_proposal.json')
    for r in rows:
        emit('  %s 窗内园地=%d Q1可用=%d Q2可用=%d' % (r['region'], r['n_orchard_in_window'],
                                                     r.get('avail_Q1_minimal', 0), r.get('avail_Q2_strict', 0)))


if __name__ == '__main__':
    main()
