# -*- coding: utf-8 -*-
"""a6_derive_verdicts.py — 别名表构建/归档 + 解盲后"正确/错误"判定表生成

* 输入：
    build-alias：v31_map.json（24 类 names/merges/old_names）→ 显式别名词表
    derive     ：--form 盲批填表（blind_id, 实际类别[, 各轴列]）、--sidecar 私有侧表（blind_id, label_before, source, role）
* 规则源：v31_map.json 的 names（权威类名）+ merges（郁闭/疏→类码，数据驱动）+ 本脚本内置的
          显式扩展别名（每条可追；不做模糊/语义自动猜测）；主 agent 2026-10-15 确认的口径
* 门槛：只做**精确匹配**（归一后逐字符相等）+ 已登记别名；未匹配 → verdict=未匹配，
          单列 unknown_alias.csv 供人工裁决，**不猜、不计入一致率分母**
* 输出：build-alias → <ws>/audit/registry/alias_map_v1.json（含来源 md5、active/candidate 两级）
        derive      → <outdir>/verdicts_<pass>.csv（blind_id, verdict, prior_code, actual_code, source, role）
                      + <outdir>/unknown_alias_<pass>.csv
* 用法：python a6_derive_verdicts.py build-alias
        python a6_derive_verdicts.py derive --form qc/pass1.csv --sidecar private/sidecar.csv --pass pass1
* 说明：verdict∈{确认,否定,未知,未匹配}；a3 只把 确认/否定/未知 计入三态一致率，未匹配单列。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
V31_MAP = r'F:/lc_work/v31_exp/config/v31_map.json'
UNKNOWN_PREFIXES = ('无法', '未知', '不可判', '判不出', '不确定')

# 显式扩展别名（active）：逐条登记，均为该类码的常见同义写法；有歧义的一律放 candidate 待裁决
EXT_ACTIVE = {
    '1': ['旱地', '雨养耕地', '旱作耕地'],
    '2': ['园地', '果园', '茶园', '橡胶园', '苗圃'],
    '3': ['水浇地', '灌溉农田', '灌区耕地'],
    '4': ['常绿阔叶', '阔叶常绿', '常绿阔叶树', 'EBF'],
    '5': ['落叶阔叶', '阔叶落叶', 'DBF'],
    '6': ['常绿针叶', '针叶常绿', '常绿针叶树', 'ENF'],
    '7': ['落叶针叶', '针叶落叶', 'DNF'],
    '8': ['针阔混交', '混交林', '针阔混交林（针阔无法定）'],
    '9': ['常绿灌木', '常绿灌丛（灌木）', '常绿灌木林'],
    '10': ['落叶灌木', '落叶灌丛（灌木）'],
    '11': ['草原'],
    '12': ['苔藓', '地衣'],
    '13': ['稀疏植被（稀疏草地）'],
    '14': ['木本湿地', '灌丛沼泽'],
    '15': ['草沼', '草本湿地', '沼泽草地'],
    '16': ['河湖滩地', '滩地', '河滩地', '湖滩地'],
    '17': ['盐碱湿地', '内陆盐沼'],
    '18': ['红树', '红树林湿地'],
    '19': ['海滨盐沼', '沿海盐沼'],
    '20': ['潮间带', '泥滩'],
    '21': ['建成区', '建设用地', '城镇', '城市', '人工表面', '建筑用地'],
    '22': ['裸露地表', '裸土'],
    '23': ['水域', '水面', '湖泊', '河流', '水库', '坑塘'],
    '24': ['冰雪', '冰川', '积雪', '雪盖'],
}
# candidate：有歧义或与邻类可能混淆，**未启用**，待主 agent 裁决后再进 active
EXT_CANDIDATE = {
    '3': ['水田', '水稻田'],
    '11': ['草甸'],
    '13': ['稀疏草地'],
    '14': ['森林沼泽'],
    '22': ['沙地', '裸岩', '戈壁', '盐碱地'],
    '23': ['水田（灌溉）'],
}


def norm(v):
    s = str(v if v is not None else '').strip()
    for a, b in (('（', '('), ('）', ')'), ('／', '/'), ('，', ','), ('：', ':')):
        s = s.replace(a, b)
    s = s.replace(' ', '').replace('\u3000', '')
    s = s.rstrip('。.;；,，')
    return s


def key(s):
    """匹配键：归一 + ASCII 小写（中文不受影响）。"""
    return norm(s).lower()


def fh(path):
    h = hashlib.sha256()
    h.update(open(path, 'rb').read())
    return h.hexdigest()


def build_alias(ws, force=False):
    d = json.load(open(V31_MAP, encoding='utf-8'))
    names, merges, old = d['names'], d['merges'], d.get('old_names', {})
    codes = {}
    for c, nm in sorted(names.items(), key=lambda kv: int(kv[0])):
        al = list(EXT_ACTIVE.get(c, []))
        cand = list(EXT_CANDIDATE.get(c, []))
        # 数据驱动：merges 把旧码并入本码，旧码名（含郁闭/疏）即本码的等价写法
        for m, srcs in merges.items():
            if m == c:
                for s in srcs:
                    if str(s) in old:
                        al.append(old[str(s)])
        codes[c] = {'name': nm, 'aliases_active': sorted(set(al)),
                    'aliases_candidate': sorted(set(cand))}
    idx = {}
    dup = {}
    for c, v in codes.items():
        for a in [v['name']] + v['aliases_active']:
            k = key(a)
            if k in idx and idx[k] != c:
                dup.setdefault(k, []).append(c)
            idx[k] = c
    rec = {
        'schema': 'class_alias_map/v1',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'generated_by': 'audit_subagent（独立审计）',
        'source': {'v31_map': V31_MAP, 'md5': hashlib.md5(open(V31_MAP, 'rb').read()).hexdigest(),
                   'sha256': fh(V31_MAP)},
        'normalization': '去空格/全角转半角/去尾标点/ASCII 小写；仅此，无模糊与语义猜测',
        'match_policy': ['精确匹配（归一后逐字符相等）',
                         '未匹配 → verdict=未匹配，单列 unknown_alias 交人工裁决，不计入一致率分母',
                         'candidate 别名未启用，需主 agent 裁决后移入 active'],
        'unknown_markers': list(UNKNOWN_PREFIXES),
        'codes': codes,
        'alias_index_active': idx,
        'duplicate_keys_across_codes': dup,
    }
    reg = os.path.join(ws, 'audit', 'registry')
    os.makedirs(reg, exist_ok=True)
    out = os.path.join(reg, 'alias_map_v1.json')
    if os.path.exists(out) and not force:
        old_rec = json.load(open(out, encoding='utf-8'))
        if old_rec.get('alias_index_active') == idx:
            print('别名表已存在且一致，未改写：%s' % out)
            return out
        raise SystemExit('已存在不同版本的别名表 %s；确认后用 --force 覆盖（会留 supersede）' % out)
    if os.path.exists(out):
        rec['supersedes'] = {'generated_at': json.load(open(out, encoding='utf-8')).get('generated_at')}
    tmp = out + '.tmp'
    json.dump(rec, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    n_act = sum(len(v['aliases_active']) for v in codes.values())
    n_can = sum(len(v['aliases_candidate']) for v in codes.values())
    print('别名表：24 码｜active 扩展别名 %d 条｜candidate（未启用）%d 条｜索引键 %d 个'
          % (n_act, n_can, len(idx)))
    if dup:
        print('警告：跨码重复键 %s（需裁决）' % json.dumps(dup, ensure_ascii=False))
    print('→ %s（来源 v31_map md5 %s）' % (out, rec['source']['md5']))
    return out


def sid(r):
    for k in ('blind_id', 'point_id', 'id'):
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ''


def id_col_of(rows, prefer):
    if not rows:
        return prefer
    cols = list(rows[0].keys())
    for c in (prefer, 'blind_id', 'point_id', 'id'):
        if c in cols:
            return c
    return cols[0]


def apply_map(rows, map_path):
    if not map_path:
        return rows
    m = json.load(open(map_path, encoding='utf-8'))
    inv = {v: k for k, v in m.items() if v and not str(v).startswith('__')}
    return [{inv.get(k, k): v for k, v in r.items()} for r in rows]


def derive(ws, form, sidecar, tag, outdir=None, alias=None, sidecar_map=None, id_col=None):
    amap = json.load(open(alias or os.path.join(ws, 'audit', 'registry', 'alias_map_v1.json'),
                          encoding='utf-8'))
    idx = amap['alias_index_active']
    forms = list(csv.DictReader(open(form, encoding='utf-8-sig')))
    idc = id_col_of(forms, id_col or 'blind_id')
    srows = (__import__('pandas').read_parquet(sidecar).to_dict('records')
             if sidecar.lower().endswith(('.parquet', '.pq'))
             else list(csv.DictReader(open(sidecar, encoding='utf-8-sig'))))
    side = {sid(r): r for r in apply_map(srows, sidecar_map)}
    outd = outdir or os.path.join(ws, 'qc')
    os.makedirs(outd, exist_ok=True)
    rows, unmatched = [], []
    for r in forms:
        bid = str(r[idc]).strip()
        raw = str(r.get('实际类别', '') or '').strip()
        s = side.get(bid, {})
        prior = str(s.get('label_before', '') or '').strip()
        prior = prior.split('.')[0] if prior else ''      # 兼容 4.0 之类浮点写法
        if not raw or raw.startswith(UNKNOWN_PREFIXES):
            v, ac = '未知', ''
        else:
            ac = idx.get(key(raw), '')
            if not ac:
                v = '未匹配'
                unmatched.append({'blind_id': bid, '实际类别原文': raw,
                                  'prior_code': prior, '建议': '待人工裁决是否登记为别名'})
            else:
                v = '确认' if prior and ac == prior else '否定'
        rows.append({'blind_id': bid, 'verdict': v, 'prior_code': prior, 'actual_code': ac,
                     'source': str(s.get('source', '')), 'role': str(s.get('role', ''))})
    fp = os.path.join(outd, 'verdicts_%s.csv' % tag)
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['blind_id', 'verdict', 'prior_code', 'actual_code', 'source', 'role'])
        w.writeheader()
        w.writerows(rows)
    up = os.path.join(outd, 'unknown_alias_%s.csv' % tag)
    with open(up, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['blind_id', '实际类别原文', 'prior_code', '建议'])
        w.writeheader()
        w.writerows(unmatched)
    from collections import Counter
    c = Counter(r['verdict'] for r in rows)
    print('判定表 %s：%s（未匹配 %d 已单列，不进入一致率分母）' % (tag, dict(c), len(unmatched)))
    print('→ %s ｜ 未匹配清单 %s' % (fp, up))


def main():
    ap = argparse.ArgumentParser(description='别名表构建/归档 + 解盲后判定表生成')
    ap.add_argument('--ws', default=WS_DEFAULT)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build-alias')
    b.add_argument('--force', action='store_true')
    d = sub.add_parser('derive')
    d.add_argument('--form', required=True)
    d.add_argument('--sidecar', required=True)
    d.add_argument('--pass', dest='tag', required=True, help='如 pass1 / pass2')
    d.add_argument('--outdir', default=None)
    d.add_argument('--alias', default=None)
    d.add_argument('--sidecar-map', default=None, help='JSON：{标准列名: 本侧表列名}')
    d.add_argument('--id-col', default=None)
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    if a.cmd == 'build-alias':
        build_alias(ws, force=a.force)
    else:
        derive(ws, a.form, a.sidecar, a.tag, a.outdir, a.alias, a.sidecar_map, a.id_col)


if __name__ == '__main__':
    main()
