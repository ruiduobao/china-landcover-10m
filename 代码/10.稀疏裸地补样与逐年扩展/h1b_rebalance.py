# -*- coding: utf-8 -*-
"""h1b_rebalance.py — 采集后分窗配额的再平衡（只改"每窗保留数"，不动采集规则）

* 背景：东区黄土高原东的稀疏/裸地候选远多于预注册小配额（原 100/200），
  按面积与候选量上调；总稀疏仍在 5 万类阈值以内（42,978 + 北 5,500 + 东 1,300 ≈ 49.8k 上限）。
* 输入/输出：data/candidates/cand_east.parquet（就地重写）；north 不动
* 用法：python h1b_rebalance.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1_harvest_fcs10 as H  # noqa

# 东区新配额：(sparse, bare)
NEW_CAPS = {
    '华北(113-120,36-38.5)': (200, 500),
    '黄淮山东(114-122,32-36)': (120, 300),
    '华东华南(114-125,18-32)': (30, 200),
    '黄土高原东(107-113.5,34-38.5)': (900, 600),
    '其余': (50, 400),
}


def main():
    fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'data', 'candidates', 'cand_east.parquet')
    d = pd.read_parquet(fp)
    before = d.groupby(['fam', 'subwin']).size().unstack(0).fillna(0).astype(int)
    names = [s[0] for s in H.SUBBOX['east']]
    keep = []
    for fam in ('sparse', 'bare'):
        sub = d[d.fam == fam]
        for k, nm in enumerate(names):
            cap = NEW_CAPS[nm][0 if fam == 'sparse' else 1]
            g = sub[sub.subwin == k]
            if len(g) > cap:
                g = g.sample(n=cap, random_state=7)
            keep.append(g)
    out = pd.concat(keep).reset_index(drop=True)
    out.to_parquet(fp, index=False)
    after = out.groupby(['fam', 'subwin']).size().unstack(0).fillna(0).astype(int)
    print('重平衡前：\n%s' % before.to_string())
    print('重平衡后：\n%s' % after.to_string())
    print('总计：%s' % out.groupby('fam').size().to_dict())
    s = {'n': int(len(out)), 'by_fam': {k: int(v) for k, v in out.fam.value_counts().items()},
         'caps': NEW_CAPS}
    json.dump(s, open(fp.replace('.parquet', '_rebalanced.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
