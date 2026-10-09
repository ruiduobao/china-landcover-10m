# -*- coding: utf-8 -*-
"""n12_merge_judges.py — 合并子代理判读 CSV → 真值表（三实验通用）

* 输入：data/m3/exp{A,B,C}_judge/judge_*.csv（point_id,Q1,Q2,conf,note；UTF-8-SIG）
        + data/m3/exp{A,B,C}_*_points.csv（点表）＋ exp{A,B,C}_features.csv（特征）
* 规则源：按 point_id 去重（保留首个非空判读；重复行记入 duplicates）；Q2 类名映射 24 类码（RULES 同 n2_nx_exp）；
  Q1=无法判读 的点保留在表中但标 invalid。
* 门槛：覆盖率（有判读/总点）与 Q2 未映射率必须打印；缺判点列出。
* 输出：data/m3/exp{A,B,C}_judged.csv（point_id,lon,lat,+点表列,Q1,Q2,conf,note,truth_code,invalid,+特征列）
        results/d2/exp{A,B,C}_judge_stats.json
* 用法：python n12_merge_judges.py --set A|B|C
"""
import csv
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import v31_common as VC
from n2_nx_exp import code_of

M3D = r'F:/lc_work/v31_exp/data/m3'
PTS = {'A': 'expA_forest_points.csv', 'B': 'expB_humid_shrub_points.csv', 'C': 'expC_wetland_points.csv',
       'A2': 'expA2_points.csv', 'C2': 'expC2_points.csv',
       'C3': 'expC3_points.csv'}


def main():
    a = sys.argv[1:]
    key = a[a.index('--set') + 1].upper()
    jd = os.path.join(M3D, 'exp%s_judge' % key)
    files = sorted(glob.glob(os.path.join(jd, 'judge_*.csv')))
    rows, dup, seen = [], 0, set()
    for fp in files:
        for r in csv.DictReader(open(fp, encoding='utf-8-sig')):
            pid = (r.get('point_id') or '').strip()
            if not pid:
                continue
            if pid in seen:
                dup += 1
                continue
            seen.add(pid)
            rows.append({'point_id': pid, 'Q1': (r.get('Q1') or '').strip(),
                         'Q2': (r.get('Q2') or '').strip(), 'conf': (r.get('conf') or '').strip(),
                         'note': (r.get('note') or '').strip(), 'file': os.path.basename(fp)})
    j = pd.DataFrame(rows)
    pts = pd.read_csv(os.path.join(M3D, PTS[key]), encoding='utf-8-sig')
    m = pts.merge(j, on='point_id', how='left')
    m['invalid'] = (~m['Q1'].isin(['是', '否']))
    m['truth_code'] = m['Q2'].map(lambda s: code_of(str(s) or '') or 0).astype(int)
    fp = os.path.join(M3D, 'exp%s_features.csv' % key)
    if os.path.exists(fp):
        m = m.merge(pd.read_csv(fp, encoding='utf-8-sig').drop(columns=['lon', 'lat']), on='point_id', how='left')
    out = os.path.join(M3D, 'exp%s_judged.csv' % key)
    m.to_csv(out, index=False, encoding='utf-8-sig')
    stat = {'set': key, 'n_points': int(len(m)), 'n_judged': int(m.Q1.notna().sum()),
            'n_valid': int((~m.invalid & m.Q1.notna()).sum()), 'n_invalid': int((m.invalid & m.Q1.notna()).sum()),
            'n_missing': int(m.Q1.isna().sum()), 'duplicates': int(dup), 'files': len(files),
            'Q1_dist': m['Q1'].fillna('(缺)').value_counts().to_dict(),
            'Q2_top': m[m.Q1.notna()]['Q2'].fillna('(空)').value_counts().head(20).to_dict(),
            'unmapped_Q2': int(((m.Q2.fillna('') != '') & (m.truth_code == 0)).sum())}
    VC.jsave(stat, os.path.join(VC.RES, 'd2', 'exp%s_judge_stats.json' % key))
    print(json.dumps(stat, ensure_ascii=False, indent=1))
    print('→', out)


if __name__ == '__main__':
    main()
