# -*- coding: utf-8 -*-
"""a0_freeze_registry.py — doc72/技能脚本冻结哈希登记（必须首批判读前执行）

* 输入：--doc72（默认 技术文档/72_盲判证据驱动样本修复与小分类实施协议.md）
        --extra（可选，追加要冻结的路径，可多次）
* 规则源：doc72 §3.1/§5（本协议在材料判读前冻结）、doc70 §4.1（注册字段与指纹）、doc71 §2（冻结记录先例）
* 门槛：doc72 必须存在；冻结必须在**首批判读前**完成——若工作区 blind/ qc/ materials/ private/
        下已存在任何产物，仍记录哈希但把 freeze_status 标为 LATE_FREEZE（不伪称首判前）
* 输出：<ws>/audit/registry/doc72_freeze.json（原子写）+ freeze_manifest_<ts>.csv（md5/sha256/size/mtime）
* 用法：python a0_freeze_registry.py
        python a0_freeze_registry.py --ws F:/lc_work/label_repair_2023_20261015 --extra <path>
* 说明：只读哈希，不修改任何输入；已存在 final 冻结时默认拒绝覆盖（--force 才supersede，且留痕）。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
CODE12 = os.path.join(PROJ, '代码', '12.盲判修复与小分类')
AUDIT_CODE = os.path.join(CODE12, 'audit')
SKILL = r'C:/Users/Administrator/.zcode/skills/rs-sample-labeling'
CODE11_AUDIT = os.path.join(PROJ, '代码', '11.改进实施与小区域实验', 'audit', 'scripts')
WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
DOC72_DEFAULT = os.path.join(PROJ, '技术文档', '72_盲判证据驱动样本修复与小分类实施协议.md')

# 首判前应保持为空的目录（出现文件即视为可能已开始判读）
FIRST_PASS_WATCH = ['blind', 'qc', 'materials', 'private', 'results', 'experiments']
REF_DOCS = ['69_三省两轮分类与样本质量综合评价及改进建议.md',
            '70_样本与分类改进实施方案及20分钟小区域实验规范.md',
            '71_改进实施与小区域分类实验结果汇总.md']


def digest(path, chunk=1 << 20):
    m, s = hashlib.md5(), hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
            m.update(b)
            s.update(b)
    return m.hexdigest(), s.hexdigest(), n


def info(path):
    if not os.path.exists(path):
        return None
    md5, sha, size = digest(path)
    st = os.stat(path)
    return {'path': path.replace('\\', '/'), 'size_bytes': size,
            'mtime': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'md5': md5, 'sha256': sha}


def collect(paths, tag, recursive=False):
    out = {}
    for p in paths:
        p = p.replace('\\', '/')
        if os.path.isdir(p):
            files = []
            if recursive:
                for root, dirs, fs in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
                    files += [os.path.join(root, f) for f in fs]
            else:
                files = [os.path.join(p, f) for f in sorted(os.listdir(p))
                         if os.path.isfile(os.path.join(p, f))]
            for fp in sorted(files):
                f = os.path.basename(fp)
                if f.endswith(('.pyc',)):
                    continue
                rel = os.path.relpath(fp, p).replace('\\', '/')
                r = info(fp)
                if r:
                    out['%s/%s' % (tag, rel)] = r
        else:
            r = info(p)
            if r:
                out['%s/%s' % (tag, os.path.basename(p))] = r
    return out


def watch_dirs(ws):
    """首判前应空；返回非空目录及其文件（作为 LATE_FREEZE 证据）。"""
    hits = {}
    for d in FIRST_PASS_WATCH:
        dd = os.path.join(ws, d)
        if not os.path.isdir(dd):
            continue
        fs = []
        for root, _, files in os.walk(dd):
            for f in files:
                if f.endswith('.pyc'):
                    continue
                fs.append(os.path.relpath(os.path.join(root, f), ws).replace('\\', '/'))
        if fs:
            hits[d] = sorted(fs)[:20]
    return hits


def main():
    ap = argparse.ArgumentParser(description='doc72 与技能脚本冻结哈希登记（首判前）')
    ap.add_argument('--ws', default=WS_DEFAULT)
    ap.add_argument('--doc72', default=DOC72_DEFAULT)
    ap.add_argument('--extra', nargs='*', default=[], help='追加冻结路径（文件或目录）')
    ap.add_argument('--force', action='store_true', help='已存在 final 冻结时 supersede（留痕）')
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    reg = os.path.join(ws, 'audit', 'registry')
    os.makedirs(reg, exist_ok=True)
    out_json = os.path.join(reg, 'doc72_freeze.json')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists(a.doc72):
        raise SystemExit('doc72 不存在：%s（先等主 agent 落盘再冻结）' % a.doc72)
    d72 = info(a.doc72)
    refs = collect([os.path.join(PROJ, '技术文档', x) for x in REF_DOCS], 'doc')
    skill = collect([os.path.join(SKILL, 'SKILL.md'),
                     os.path.join(SKILL, 'scripts'),
                     os.path.join(SKILL, 'references')], 'skill')
    code11 = collect([CODE11_AUDIT], 'code11_audit')
    code12 = collect([CODE12], 'code12', recursive=True)
    extra = collect(a.extra, 'extra') if a.extra else {}

    hits = watch_dirs(ws)
    status = 'PRE_FIRST_PASS' if not hits else 'LATE_FREEZE'
    rec = {
        'schema': 'label_repair_freeze/v1',
        'recorded_by': 'audit_subagent（source-aware 独立审计，不参与盲判）',
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'workspace': ws,
        'purpose': '在首批判读/材料制作前冻结 doc72 与所用技能脚本指纹；判读后本文件只读',
        'freeze_status': status,
        'first_pass_watch': {'dirs': FIRST_PASS_WATCH, 'nonempty': hits},
        'doc72': d72,
        'reference_docs': refs,
        'skill_scripts': skill,
        'code11_audit_scripts': code11,
        'code12_audit_scripts': code12,
        'extra': extra,
        'does_not_cover': [
            'R/V/Q 点位与选择注册（另由 selection_registry.json 在预测前登记）',
            '主采纳门（另由 gate_freeze.json 在 R QC 后、V 解盲前登记）',
            '影像材料与判读结果（本轮为产物，不入冻结）'],
    }
    if status == 'LATE_FREEZE':
        rec['late_freeze_note'] = ('工作区已存在产物，无法证明本哈希记录早于首批判读；'
                                   '后续报告不得称"首判前冻结"。命中：%s' % json.dumps(hits, ensure_ascii=False))
    if os.path.exists(out_json) and not a.force:
        old = json.load(open(out_json, encoding='utf-8'))
        if old.get('freeze_status') in ('PRE_FIRST_PASS', 'LATE_FREEZE') and not old.get('superseded'):
            raise SystemExit('已存在冻结记录 %s（%s）。如需重记请加 --force（会留 supersede 痕迹）'
                             % (out_json, old.get('recorded_at')))
    history = []
    if os.path.exists(out_json):
        old = json.load(open(out_json, encoding='utf-8'))
        history = list(old.get('history', []))
        history.append({'recorded_at': old.get('recorded_at'),
                        'freeze_status': old.get('freeze_status'),
                        'doc72_md5': (old.get('doc72') or {}).get('md5'),
                        'n_skill': len(old.get('skill_scripts') or {}),
                        'n_code12': len(old.get('code12_audit_scripts') or {}),
                        'nonempty_watch': sorted((old.get('first_pass_watch') or {}).get('nonempty', {}).keys())})
        rec['supersedes'] = {'recorded_at': old.get('recorded_at'),
                             'doc72_md5': (old.get('doc72') or {}).get('md5')}
        # 归档上一版记录本体，保留"首判前"那一刻的完整证据
        arc = os.path.join(reg, 'doc72_freeze_%s.json'
                           % (old.get('recorded_at') or 'old').replace('-', '').replace(':', '').replace('T', '_'))
        if not os.path.exists(arc):
            json.dump(old, open(arc, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            rec['archived_previous'] = arc.replace('\\', '/')
    rec['history'] = history
    tmp = out_json + '.tmp'
    json.dump(rec, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, out_json)

    man = os.path.join(reg, 'freeze_manifest_%s.csv' % ts)
    with open(man, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'key', 'path', 'size_bytes', 'mtime', 'md5', 'sha256'])
        for grp, d in (('doc72', {'doc72': d72}), ('ref', refs), ('skill', skill),
                       ('code11', code11), ('code12', code12), ('extra', extra)):
            for k, r in d.items():
                w.writerow([grp, k, r['path'], r['size_bytes'], r['mtime'], r['md5'], r['sha256']])
    print('freeze_status=%s  doc72 md5=%s sha256=%s' % (status, d72['md5'], d72['sha256']))
    print('冻结文件 %d 项（skill %d / code11 %d / code12 %d / ref %d）→ %s'
          % (1 + len(skill) + len(code11) + len(code12) + len(refs) + len(extra),
             len(skill), len(code11), len(code12), len(refs), out_json))
    print('manifest → %s' % man)
    if status == 'LATE_FREEZE':
        print('警告：检测到工作区已有产物，freeze_status=LATE_FREEZE（见 nonempty）')


if __name__ == '__main__':
    main()
