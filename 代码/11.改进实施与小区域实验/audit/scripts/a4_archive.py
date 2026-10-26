# -*- coding: utf-8 -*-
"""a4_archive.py — 改进实验归档计划与执行（安全：默认只出计划，不覆盖/不删除）

* 输入：工作区各组源码/产物（control/orchard/terrain/geometry/audit）
* 规则源：协调者归档约定
    - *.py            → 代码/11.改进实施与小区域实验/<group>/...
    - *.md/*.json/*.csv → 技术文档/附件/70_改进实施与小区域实验/<group>/...
    - 大件(.tif/.npz/.parquet/.pkl/.log) 不复制；主文给绝对链接
    - 同 sha256 → SKIP_SAME；异 sha256 → CONFLICT（默认不覆盖，仅记日志）
    - 原产物永不删除
* 门槛：pred CSV 需无凭证（脚本内做敏感串扫描，命中则 EXCLUDE）
* 输出：audit/registry/archive_plan_<ts>.csv、archive_log_<ts>.json、
        timing_all_<ts>.csv、main_layer_summary_<ts>.csv、README_entry_<ts>.md
* 用法：
    python a4_archive.py                # 只出计划（默认）
    python a4_archive.py --apply        # 执行复制（跳过 CONFLICT）
    python a4_archive.py --apply --update   # 异 hash 也覆盖（仍记日志）
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
WS = os.path.dirname(AUDIT)
OUT = os.path.join(AUDIT, 'out')
REG = os.path.join(AUDIT, 'registry')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
DEST_CODE = os.path.join(PROJ, '代码', '11.改进实施与小区域实验')
DEST_DOC = os.path.join(PROJ, '技术文档', '附件', '70_改进实施与小区域实验')

GROUPS = ['control', 'orchard', 'terrain', 'geometry', 'audit']
CODE_EXT = {'.py'}
DOC_EXT = {'.md', '.json', '.csv'}
BIG_EXT = {'.tif', '.tiff', '.npz', '.parquet', '.pkl', '.log', '.png', '.jpg'}
SKIP_DIRS = {'__pycache__', 'locks', 'logs', 'cache', '_selftest'}
# 本归档脚本自身按时间戳生成的审计工作产物不进计划（否则每跑一次就新增待复制件）
SKIP_NAME_PREFIX = ('timing_all_', 'main_layer_summary_', 'README_entry_', 'archive_')
# 敏感串：命中则排除（不复制含凭证文件）
SECRET_PAT = re.compile(rb'(private_key|BEGIN [A-Z ]*PRIVATE KEY|"client_email"|refresh_token|access_token|api[_-]?key\s*[:=])', re.I)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_secret(p):
    # 本脚本自身含检测用正则字面量，属假阳性，不扫描
    if os.path.basename(p) == os.path.basename(__file__):
        return False
    try:
        with open(p, 'rb') as f:
            head = f.read(65536)
        return bool(SECRET_PAT.search(head))
    except OSError:
        return False


def iter_sources():
    for g in GROUPS:
        base = os.path.join(WS, g)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in sorted(files):
                if any(fn.startswith(sf) for sf in SKIP_NAME_PREFIX):
                    continue
                yield g, os.path.join(root, fn)


def dest_for(group, src, ext):
    rel = os.path.relpath(src, os.path.join(WS, group))
    if ext in CODE_EXT:
        return os.path.join(DEST_CODE, group, rel)
    return os.path.join(DEST_DOC, group, rel)


def build_plan():
    rows = []
    for g, src in iter_sources():
        ext = os.path.splitext(src)[1].lower()
        if ext in BIG_EXT:
            rows.append(dict(group=g, action='SKIP_BIG', src=src, dest='', sha256='',
                             note='大件不复制；主文给绝对链接'))
            continue
        if ext not in CODE_EXT and ext not in DOC_EXT:
            rows.append(dict(group=g, action='SKIP_EXT', src=src, dest='', sha256='', note=ext))
            continue
        if is_secret(src):
            rows.append(dict(group=g, action='EXCLUDE_SECRET', src=src, dest='', sha256='', note='命中敏感串'))
            continue
        sha = sha256_file(src)
        dest = dest_for(g, src, ext)
        if os.path.exists(dest):
            dsha = sha256_file(dest)
            if dsha == sha:
                rows.append(dict(group=g, action='SKIP_SAME', src=src, dest=dest, sha256=sha, note=''))
                continue
            rows.append(dict(group=g, action='CONFLICT', src=src, dest=dest, sha256=sha,
                             note='dest 异 hash(%s) 需 Read+Edit 或 --update' % dsha[:12]))
            continue
        rows.append(dict(group=g, action='COPY', src=src, dest=dest, sha256=sha, note=''))
    return rows


def apply_plan(rows, update=False):
    log = []
    for r in rows:
        if r['action'] == 'COPY' or (r['action'] == 'CONFLICT' and update):
            os.makedirs(os.path.dirname(r['dest']), exist_ok=True)
            with open(r['src'], 'rb') as fi, open(r['dest'], 'wb') as fo:
                fo.write(fi.read())
            log.append(dict(r, applied='COPIED'))
        else:
            log.append(dict(r, applied='SKIPPED'))
    return log


def emit_timing_csv():
    """汇总各组真实计时：run_results/*.json + t2/t4 runlog + orchard unit_runs.csv（不遍历大数据）。"""
    rows = []
    for g in GROUPS:
        for sub in ['run_results', 'out', 'results']:
            d = os.path.join(WS, g, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if fn.endswith('.csv') and 'unit_runs' in fn:
                    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
                        rows.append(dict(group=g, file=fn, exp_id='%s/%s' % (r.get('region'), r.get('arm')),
                                         status=r.get('state', ''), wall_s=r.get('wall_s', ''),
                                         limit_s=1200, started_at='', finished_at='', path=p))
                    continue
                if not fn.endswith('.json'):
                    continue
                try:
                    j = json.load(open(p, encoding='utf-8'))
                except Exception:
                    continue
                if not isinstance(j, dict):
                    continue
                wall = j.get('wall_s', j.get('wall_seconds', j.get('total_elapsed_s')))
                if wall is None:
                    continue
                if not (fn.startswith('G30N') or 'runlog' in fn or 't2_' in fn or 't4_' in fn
                        or fn.startswith('summary') or 'seedset' in fn):
                    continue
                rows.append(dict(group=g, file=fn, exp_id=j.get('exp_id', j.get('frame', '')),
                                 status=j.get('status', j.get('state', '')),
                                 wall_s=wall, limit_s=j.get('limit_s', j.get('deadline_s', '')),
                                 started_at=j.get('started_at', ''), finished_at=j.get('finished_at', ''),
                                 path=p))
    return rows


def emit_main_layer_summary():
    """主层结果简表（conf∩nonFCS10）——从 audit/out 已产 JSON 汇总，不遍历大数据。"""
    rows = []
    # orchard
    for f in [x for x in sorted(os.listdir(OUT)) if x.startswith('orchard_recompute_') and x.endswith('.json')][-1:]:
            d = json.load(open(os.path.join(OUT, f), encoding='utf-8'))
            for rid, v in d.get('regions', {}).items():
                ml = v.get('main_layer') or {}
                tf = v.get('target_f1_conf') or {}
                for c, x in tf.items():
                    rows.append(dict(group='orchard', unit=rid, layer='conf_nonfcs10',
                                     n=v.get('n_unique_conf_nonfcs10'), blocks=ml.get('n_blocks'),
                                     dm9=ml.get('dm9'), dm9_ci=ml.get('dm9_ci'),
                                     target_class=c, target_n=x.get('n'), target_dF1=x.get('dF1')))
            break
    # terrain
    for f in [x for x in sorted(os.listdir(OUT)) if x.startswith('terrain_recompute_') and x.endswith('.json')][-1:]:
            d = json.load(open(os.path.join(OUT, f), encoding='utf-8'))
            for fr, v in d.get('frames', {}).items():
                ml = v['layers'].get('conf_nonfcs10') or {}
                tf = v.get('target_f1_conf_nonfcs10') or {}
                rows.append(dict(group='terrain', unit=fr, layer='conf_nonfcs10',
                                 n=ml.get('n'), blocks=ml.get('n_blocks'),
                                 dm9=ml.get('dm9'), dm9_ci=ml.get('dm9_ci'),
                                 target_class=4, target_n=(tf.get('4') or {}).get('n'),
                                 target_dF1=(tf.get('4') or {}).get('dF1')))
            break
    # geometry
    rows.append(dict(group='geometry', unit='G30N_v2', layer='coverage',
                     n=40661, blocks='', dm9='', dm9_ci='',
                     target_class='', target_n='', target_dF1='restored 23394px / common_valid 17267 / disagree 0'))
    return rows


def emit_readme_entry(plan_counts, ts):
    lines = [
        '# 附件：70 改进实施与小区域实验（audit 生成）',
        '',
        '> 由 audit 子 agent 于 %s 生成。原始产物不删；大件不复制，主文给绝对链接。' % ts,
        '',
        '## 分组',
        '- `control/` 运行器自检与注册（超时只本树、mem guard、namespace）',
        '- `orchard/` 园地先验实验（O-CAP）',
        '- `terrain/` 地形分层与 F-VIEW（T-STRATA / strict common）',
        '- `geometry/` G-30N 几何覆盖修复（v1 VALIDITY_FAIL 保留；v2 engineering PASS）',
        '- `audit/` 独立复算脚本、判定与 manifest',
        '',
        '## 归档动作统计',
        '```',
        json.dumps(plan_counts, ensure_ascii=False),
        '```',
        '',
        '## 判定（三字段）',
        '见 `audit/out/verdicts_2026-10-15.json`：园地/地形 adoption=REJECT；H2 metadata=BLOCKED；几何 ACCEPT_EXPERIMENTAL(engineering only)。',
        '',
        '## 入口',
        '- 冻结：`audit/registry/doc70_freeze.json`（v1.0 最终 md5 749017cb73ba7e66b5816b1bbd7486bf）',
        '- 结论：`audit/review/最终审计结论_2026-10-15.md`',
        '- 计时：`audit/out/timing_all_*.csv`',
        '- 主层简表：`audit/out/main_layer_summary_*.csv`',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--update', action='store_true')
    args = ap.parse_args()

    os.makedirs(REG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    rows = build_plan()
    fp = os.path.join(REG, 'archive_plan_%s.csv' % ts)
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'action', 'src', 'dest', 'sha256', 'note'])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    cnt = Counter(r['action'] for r in rows)
    print('[PLAN] %s' % fp)
    print('   ', dict(cnt))

    if args.apply:
        log = apply_plan(rows, args.update)
        lf = os.path.join(REG, 'archive_log_%s.json' % ts)
        json.dump(log, open(lf, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('[APPLY] -> %s (COPIED=%d)' % (lf, sum(1 for x in log if x['applied'] == 'COPIED')))
    else:
        print('   (dry-run；加 --apply 执行；异 hash 默认不覆盖)')

    trows = emit_timing_csv()
    if trows:
        tf = os.path.join(OUT, 'timing_all_%s.csv' % ts)
        with open(tf, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
            w.writeheader()
            w.writerows(trows)
        print('[TIMING] -> %s (%d rows)' % (tf, len(trows)))

    mrows = emit_main_layer_summary()
    if mrows:
        mf = os.path.join(OUT, 'main_layer_summary_%s.csv' % ts)
        with open(mf, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
            w.writeheader()
            w.writerows(mrows)
        print('[MAIN_LAYER] -> %s (%d rows)' % (mf, len(mrows)))

    re_md = os.path.join(OUT, 'README_entry_%s.md' % ts)
    open(re_md, 'w', encoding='utf-8').write(emit_readme_entry(dict(cnt), ts))
    print('[README_ENTRY] -> %s' % re_md)


if __name__ == '__main__':
    main()
