# -*- coding: utf-8 -*-
"""r2_blind_pack.py — 盲判包：中性 ID、sidecar 隔离、60/40 空间块拆分、分批材料

* 输入（只读）：private/candidates_private.parquet（r1 产物）
* 规则源：技术文档/72 协议（主 agent 冻结）；skill references/判读协议.md §2 盲判纪律
* 盲判面（blind/、materials/）**只有**：中性 point_id（B####）、lon、lat、影像源与 date 状态；
  先验标签 / role / 分组 / 配对 / 坡向 / 来源 / 两版类别 **一律只在 private/**（判读代理不得读）。
* 拆分：R 块 = repair-dev（60%），V 块 = sealed-confirm（40%）；预测揭示前冻结（private/split_blocks.json）。
* 表单列用**轴标签**（结构/叶型轴/物候轴/实际类别），**不设"正确/错误"列**——先盲标实际类，
  解盲匹配在后（r6_deblind_stats.py）；防止"正确/错误"提示先验。
* 输出：blind/points_blind.csv（B####,lon,lat）、blind/points_series.csv（源+date 状态，无先验）、
        blind/form_skeleton.csv、materials/batch_XX.csv + 表单骨架 + batch_manifest.json、
        private/uid_map.csv（B#### ↔ uid/role/group/pair/prior）、private/split_blocks.json
* 用法：python r2_blind_pack.py [--batch-size 40]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

FORM_COLS = ['point_id', '结构大类（林/灌/草/耕/园/水/建成/裸/湿/冰）', '叶型轴（针/阔/混交/无法判定）',
             '物候轴（常绿/落叶/无法判定）', '实际类别（24类中文名，或"无法判定"）',
             '覆盖度或管理特征（纯林/果园行栽/梯坎茶园/大棚/地膜/淹水等）',
             '置信（高/中/低）', '证据（影像特征+影像源+日期/季节+辅助指标）', '无法判定原因（云/阴影/分辨率/混合像元/无日期）']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser(description='盲判包生成')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    a = ap.parse_args()
    ensure_dirs()
    t0 = time.time()

    priv = pd.read_parquet(os.path.join(PRIV, 'candidates_private.parquet'))
    emit('私有候选 %d 行（对 %d 行 / 争议 %d 行）' % (
        len(priv), (priv['group'] == 'forest_pair').sum(), (priv['group'] == 'dispute').sum()))

    pts = []
    for i, r in priv.iterrows():
        if r['group'] == 'forest_pair':
            for side in ('a', 'b'):
                pts.append(dict(
                    src_row=int(i), role=r['role'], group='forest_pair', box=r['box'],
                    pair_id='P%03d' % (i if r['role'] == ROLE_R else 1000 + i),
                    slope_cls=r['%s_slope_cls' % side], lon=float(r['%s_lon' % side]), lat=float(r['%s_lat' % side]),
                    cl_pool=(int(r['%s_cl_pool' % side]) if r['%s_cl_pool' % side] == r['%s_cl_pool' % side] else None),
                    src_label=str(r['%s_src' % side]), aspect=float(r['%s_aspect' % side]),
                    elev=float(r['%s_elev' % side]), slope=float(r['%s_slope' % side]),
                    c_v1=(int(r['%s_c_v1' % side]) if r['%s_c_v1' % side] == r['%s_c_v1' % side] else None),
                    c_v2=(int(r['%s_c_v2' % side]) if r['%s_c_v2' % side] == r['%s_c_v2' % side] else None),
                    change='', block=str(r['block']), dist_km=r['dist_km']))
        else:
            pts.append(dict(
                src_row=int(i), role=r['role'], group='dispute', box=r['box'], pair_id='',
                slope_cls='', lon=float(r['a_lon']), lat=float(r['a_lat']),
                cl_pool=(int(r['a_cl_pool']) if r['a_cl_pool'] == r['a_cl_pool'] else None),
                src_label=str(r['a_src']), aspect=(float(r['a_aspect']) if r['a_aspect'] == r['a_aspect'] else None),
                elev=(float(r['a_elev']) if r['a_elev'] == r['a_elev'] else None),
                slope=(float(r['a_slope']) if r['a_slope'] == r['a_slope'] else None),
                c_v1=int(r['a_c_v1']), c_v2=int(r['a_c_v2']), change=str(r['change']),
                block=str(r['block']), dist_km=None))
    p = pd.DataFrame(pts)
    emit('展开点位 %d（R %d / V %d；林型对 %d / 争议 %d）' % (
        len(p), (p['role'] == ROLE_R).sum(), (p['role'] == ROLE_V).sum(),
        (p['group'] == 'forest_pair').sum(), (p['group'] == 'dispute').sum()))

    # ---- 中性 ID：随机序分配（不按组/角色/配对分组，避免顺序泄漏） ----
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(p))
    p = p.iloc[order].reset_index(drop=True)
    p['point_id'] = ['%s%0*d' % (ID_PREFIX, ID_WIDTH, i + 1) for i in range(len(p))]
    p['uid'] = [stable_uid(rl, '%s|%.6f|%.6f' % (gp, lo, la))
                for rl, gp, lo, la in zip(p['role'], p['group'], p['lon'], p['lat'])]

    # ---- 盲判面（无先验） ----
    blind = p[['point_id', 'lon', 'lat']].copy()
    blind['lon'] = blind['lon'].round(6)
    blind['lat'] = blind['lat'].round(6)
    blind.to_csv(os.path.join(BLIND, 'points_blind.csv'), index=False, encoding='utf-8-sig')

    # 影像源与日期状态（Esri/Google 高分底图无日期元数据 → 一律 unknown，不捏造）
    series = pd.DataFrame(dict(
        point_id=p['point_id'],
        image_source='Esri World Imagery (z18, 备源 Google)',
        date_status='unknown',
        date_note='高分底图瓦片无日期元数据（skill 影像源手册 §3.2）→ 不写年份；有日期证据仅来自 S2 时序',
        s2_status='pending', aef_year=S2_YEAR))
    series.to_csv(os.path.join(BLIND, 'points_series.csv'), index=False, encoding='utf-8-sig')

    with open(os.path.join(BLIND, 'form_skeleton.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(FORM_COLS)
        for pid in p['point_id']:
            w.writerow([pid] + [''] * (len(FORM_COLS) - 1))

    # ---- 私有 sidecar（唯一映射；judge 不得读） ----
    side = p.copy()
    side['cl_name'] = [CL24.get(int(c), '') if c == c and c is not None else '' for c in side['cl_pool']]
    side['c_v1_name'] = [CL24.get(int(c), '') if c == c else '' for c in side['c_v1']]
    side['c_v2_name'] = [CL24.get(int(c), '') if c == c else '' for c in side['c_v2']]
    side.to_csv(os.path.join(PRIV, 'uid_map.csv'), index=False, encoding='utf-8-sig')
    jdump(os.path.join(PRIV, 'split_blocks.json'), dict(
        frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        rule='R 块 = repair-dev（可训练，绝不评价）；V 块 = sealed-confirm（整块从训练排除，标签封存）',
        repair_dev_blocks=sorted(set(p.loc[p['role'] == ROLE_R, 'block'])),
        sealed_confirm_blocks=sorted(set(p.loc[p['role'] == ROLE_V, 'block'])),
        counts=dict(R=int((p['role'] == ROLE_R).sum()), V=int((p['role'] == ROLE_V).sum())),
        note='预测揭示前冻结；V 标签由 audit 封存，实验只拿坐标+特征'))

    # ---- 分批材料（sanitized） ----
    bs = a.batch_size
    man = []
    for bi in range(0, len(blind), bs):
        k = bi // bs + 1
        sub = blind.iloc[bi:bi + bs]
        bdir = os.path.join(MAT, 'batch_%02d' % k)
        os.makedirs(bdir, exist_ok=True)
        sub.to_csv(os.path.join(bdir, 'blind_batch.csv'), index=False, encoding='utf-8-sig')
        with open(os.path.join(bdir, 'form_skeleton.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(FORM_COLS)
            for pid in sub['point_id']:
                w.writerow([pid] + [''] * (len(FORM_COLS) - 1))
        man.append(dict(batch=k, n=len(sub), dir=bdir,
                        blind_batch=os.path.join(bdir, 'blind_batch.csv'),
                        form=os.path.join(bdir, 'form_skeleton.csv'),
                        sheets=os.path.join(bdir, 'sheets'),
                        point_ids=list(sub['point_id'])))
    jdump(os.path.join(MAT, 'batch_manifest.json'), dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'), batch_size=bs, batches=man,
        blind_columns=list(blind.columns), form_columns=FORM_COLS,
        discipline='blind 面不含先验/role/分组/配对/来源/坡向；唯一映射见 private/uid_map.csv（judge 不得读）'))
    emit('盲判包完成 %d 点 / %d 批（%.0fs）→ blind/ + materials/' % (len(blind), len(man), time.time() - t0))


if __name__ == '__main__':
    main()
