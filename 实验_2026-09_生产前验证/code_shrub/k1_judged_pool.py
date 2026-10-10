# -*- coding: utf-8 -*-
"""k1_judged_pool.py — 统一历史目视判读集为「灌木校准集」（本地，零 GEE）

* 输入：data/m3/exp{A,A2,B,C,C2,C3,C4}_judged.csv、gsnx_ref_judged.csv、
        data/shrub/judged_all.csv（doc57）
* 规则源（预注册标签口径，各实验 Q1 语义不同，必须显式统一）：
        doc57 / expB：Q1 即"是否灌丛" → Q1=是/否 直接映射
        expA（Q1=是否森林）/ expA2 / expC 系（Q1=是否湿地）：以 **Q2 地类名**为准——
            Q2∈{常绿灌丛,落叶灌丛} → is_shrub=是；其余 24 类名 → is_shrub=否
        is_shrub ∈ {是,否,无法判读}；shrub_type 从 Q2 的类名取（常绿/落叶），判不出留"待判"
* 门槛：坐标 lon/lat 必填且在中国范围；重复 orig_id 去重保留首个
* 输出：data/shrub_nat/judged_pool.csv（point_id,lon,lat,is_shrub,shrub_type,conf,src_exp,note,q1_raw,q2_raw）
* 用法：python k1_judged_pool.py
"""
import glob, json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import pandas as pd

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
M3 = os.path.join(WORK, 'data', 'm3')
SHRUB_CLASS = {'常绿灌丛', '落叶灌丛'}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def norm(v):
    return str(v).strip() if v is not None else ''


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(M3, 'exp*_judged.csv'))) + \
             [os.path.join(M3, 'gsnx_ref_judged.csv')]:
        d = pd.read_csv(f, encoding='utf-8-sig')
        name = os.path.basename(f).replace('_judged.csv', '')
        q1_is_shrub = name in ('expB',)          # expB 的 Q1 就是"是否灌丛"
        for r in d.itertuples():
            lon = getattr(r, 'lon', None); lat = getattr(r, 'lat', None)
            if pd.isna(lon) or pd.isna(lat):
                continue
            q1, q2 = norm(getattr(r, 'Q1', '')), norm(getattr(r, 'Q2', ''))
            if q1_is_shrub:
                iss = q1 if q1 in ('是', '否', '无法判读') else '无法判读'
                stype = q2 if q2 in ('常绿', '落叶') else '待判'
            else:
                iss = '是' if q2 in SHRUB_CLASS else ('无法判读' if q1 == '无法判读' and q2 == '' else '否')
                stype = '常绿' if q2 == '常绿灌丛' else ('落叶' if q2 == '落叶灌丛' else '待判')
            rows.append(dict(lon=float(lon), lat=float(lat), is_shrub=iss, shrub_type=stype,
                             conf=norm(getattr(r, 'conf', '')), src_exp=name,
                             note=norm(getattr(r, 'note', ''))[:40], q1_raw=q1, q2_raw=q2))
    # doc57（Q1 本身就是是否灌丛）
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    for r in d57.itertuples():
        rows.append(dict(lon=float(r.lon), lat=float(r.lat),
                         is_shrub=r.Q1_shrub if r.Q1_shrub in ('是', '否', '无法判读') else '无法判读',
                         shrub_type=(r.Q2_evergreen if r.Q2_evergreen in ('常绿', '落叶') else '待判'),
                         conf=norm(r.conf), src_exp='doc57', note=norm(r.note)[:40],
                         q1_raw=r.Q1_shrub, q2_raw=r.Q2_evergreen))
    df = pd.DataFrame(rows)
    df = df[(df.lon > 73) & (df.lon < 136) & (df.lat > 17) & (df.lat < 55)].reset_index(drop=True)
    df = df.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    df.insert(0, 'point_id', ['JP-%s-%05d' % (r.src_exp, i) for i, r in enumerate(df.itertuples())])
    df.to_csv(os.path.join(OUTD, 'judged_pool.csv'), index=False, encoding='utf-8-sig')
    emit('校准集 %d 点：%s' % (len(df), df['is_shrub'].value_counts().to_dict()))
    emit('  正样本类型分布 %s' % df[df.is_shrub == '是']['shrub_type'].value_counts().to_dict())
    emit('  来源 %s' % df['src_exp'].value_counts().to_dict())
    emit('→ ' + os.path.join(OUTD, 'judged_pool.csv'))


if __name__ == '__main__':
    main()
