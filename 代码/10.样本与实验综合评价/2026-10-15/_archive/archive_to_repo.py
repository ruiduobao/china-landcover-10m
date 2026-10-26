# -*- coding: utf-8 -*-
"""archive_to_repo.py — 综合评价证据轻量归档（只归档小体积证据，不放大图/密钥）

* 职责：把 `quality_review_2026-10-15/` 下的代码与证据文件，按分组复制到项目仓库：
    - 代码（.py）  → `代码/10.样本与实验综合评价/2026-10-15/<group>/`
    - 证据（.md/.csv/.json）→ `技术文档/附件/69_样本与实验综合评价/<group>/`
  分组（group）：experiment / terrain / seams / samples / methodology（另含 _archive 脚本组，只进代码侧）。
* 白名单：仅 .py / .md / .csv / .json；**排除** gpkg / parquet / tif / tiff / png / jpg / jpeg / ovr / npz / npy / zip 及 chips 等目录。
* 排除模式（大小写不敏感）：cred / token / secret / password / passwd / _key / .key / account / accounts / cfg / config ——
  含凭证、令牌、账号配置的件一律不进归档（json 仅允许统计/证据件，配置件命中上述模式即被排除）。
* 冲突规则：目标已存在——同 sha256 → 跳过并记录 same；**内容不同 → 记录 CONFLICT 并拒绝覆盖**（退出码 3）。
* 大件保护：单文件 > 50 MB 的候选件跳过并记录（轻量附件约束）。
* 默认 dry-run（只列表不写盘）；`--apply` 才真正复制并落 manifest 与 README。
* `--groups a,b` 可只归档指定分组（例：先 experiment,terrain,seams,samples,_archive，最后 methodology/03）。
  manifest 为**合并式**：重跑时同 `src_rel` 行覆盖旧行，其余保留，两趟归档结果合并在同一份 CSV。
* 用法：
    python archive_to_repo.py            # 预览（不写任何文件）
    python archive_to_repo.py --apply    # 执行归档（等指令后再跑）
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJ = Path(r'F:/地理所/论文/中国土地覆盖数据_2017-2024')
REVIEW = Path(r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15')
CODE_DEST_ROOT = PROJ / '代码' / '10.样本与实验综合评价' / '2026-10-15'
ATTACH_DEST_ROOT = PROJ / '技术文档' / '附件' / '69_样本与实验综合评价'

GROUPS = ['experiment', 'terrain', 'seams', 'samples', 'methodology']

CODE_EXT = {'.py'}
ATTACH_EXT = {'.md', '.csv', '.json'}
EXTRA_GROUP = '_archive'          # 只进代码侧（脚本与运行说明）
CODE_ONLY_GROUP = True

NAME_BLOCK = re.compile(
    r'(cred|token|secret|password|passwd|_key|\.key|account|accounts|cfg|config)',
    re.IGNORECASE,
)
DIR_BLOCK = {'chips', '__pycache__', 'vis', 'sheets', 'zoom', 'images'}
SIZE_LIMIT = 50 * 1024 * 1024      # 50 MB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dest_for(group: str, src: Path):
    """返回 (destination Path, side) ；side ∈ {code, attach}。"""
    if group == EXTRA_GROUP:
        return CODE_DEST_ROOT / group / src.name, 'code'
    if src.suffix.lower() in CODE_EXT:
        return CODE_DEST_ROOT / group / src.name, 'code'
    return ATTACH_DEST_ROOT / group / src.name, 'attach'


def iter_candidates(groups):
    for group in dict.fromkeys(groups + [EXTRA_GROUP]):
        gdir = REVIEW / group if group in GROUPS else REVIEW / '_archive'
        if not gdir.is_dir():
            print(f'[skip] 分组目录不存在：{gdir}')
            continue
        for root, dirs, files in os.walk(gdir):
            dirs[:] = [d for d in dirs if d.lower() not in DIR_BLOCK]
            for fn in sorted(files):
                p = Path(root) / fn
                if p.suffix.lower() not in (CODE_EXT | ATTACH_EXT):
                    continue
                if NAME_BLOCK.search(fn):
                    yield group, p, 'skipped-pattern', '命中凭证/配置排除模式'
                    continue
                if p.stat().st_size > SIZE_LIMIT:
                    yield group, p, 'skipped-size', f'> {SIZE_LIMIT // (1 << 20)} MB'
                    continue
                yield group, p, None, ''


def write_readme(manifest_rows, ts):
    attach_readme = ATTACH_DEST_ROOT / 'README_附件说明.md'
    code_readme = CODE_DEST_ROOT / 'README.md'
    ATTACH_DEST_ROOT.mkdir(parents=True, exist_ok=True)
    CODE_DEST_ROOT.mkdir(parents=True, exist_ok=True)
    attach_readme.write_text(
        '# 69 号文档附件说明（自动生成）\n\n'
        f'- 生成时间：{ts}\n'
        '- 内容：三省两轮分类与样本质量综合评价的**轻量证据件**（md/csv/json），按分组存放：\n'
        '  `experiment/` 实验与对照｜`terrain/` 坡向/地形诊断｜`seams/` 接缝与合幅诊断｜'
        '`samples/` 样本与质检统计｜`methodology/` 方法学与证据指引。\n'
        '- 排除：栅格/矢量大件（tif/gpkg/parquet/png/chips）与任何凭证/账号配置；\n'
        '- 对应代码：`代码/10.样本与实验综合评价/2026-10-15/`（同分组）；\n'
        '- 文件级溯源与哈希：见同目录 `_manifest_archive_2026-10-15.csv`。\n'
        '- 复现归档：`python F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/'
        '_archive/archive_to_repo.py --apply`（幂等；同哈希跳过、不同内容拒写并报 CONFLICT）。\n',
        encoding='utf-8')
    code_readme.write_text(
        '# 综合评价归档代码与证据说明（自动生成）\n\n'
        f'- 生成时间：{ts}\n'
        '- 本目录为 `quality_review_2026-10-15/` 各分组的 `*.py` 归档副本（含 `_archive/` 归档脚本）。\n'
        '- 证据件（md/csv/json）在 `技术文档/附件/69_样本与实验综合评价/`。\n'
        '- 归档脚本：`_archive/archive_to_repo.py`（dry-run 默认；`--apply` 执行；`--groups` 可分趟）。\n'
        f'- 本次动作统计：copied={sum(1 for r in manifest_rows if r["action"] == "copied")} '
        f'updated={sum(1 for r in manifest_rows if r["action"] == "updated")} '
        f'same={sum(1 for r in manifest_rows if r["action"] == "same")} '
        f'conflict={sum(1 for r in manifest_rows if r["action"] == "conflict")}。\n',
        encoding='utf-8')


def merge_manifest(man: Path, new_rows):
    """把新行并入既有 manifest（键 = src_rel；新行覆盖旧行），按 dst_rel 排序。"""
    fields = ['time', 'group', 'side', 'file', 'src_rel', 'dst_rel', 'size', 'sha256', 'action', 'note']
    rows = {}
    if man.exists():
        with open(man, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                rows[r.get('src_rel', '')] = {k: r.get(k, '') for k in fields}
    for r in new_rows:
        rows[r['src_rel']] = r
    out = sorted(rows.values(), key=lambda r: (r.get('dst_rel', ''), r.get('src_rel', '')))
    for p in (man, CODE_DEST_ROOT / man.name):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正复制（默认仅预览）')
    ap.add_argument('--groups', default='', help='逗号分隔的分组子集，如 experiment,terrain,seams,samples,_archive；缺省=全部分组')
    ap.add_argument('--update', action='store_true', help='允许用源覆盖“目标存在但内容不同”的件（默认拒写并记 conflict）')
    args = ap.parse_args()
    if args.groups:
        groups = [g.strip() for g in args.groups.split(',') if g.strip()]
        bad = [g for g in groups if g not in GROUPS and g != EXTRA_GROUP]
        if bad:
            print(f'未知分组：{bad}（可用：{GROUPS + [EXTRA_GROUP]}）')
            return 2
    else:
        groups = list(GROUPS)
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    rows, conflicts = [], 0

    for group, src, skip, why in iter_candidates(groups):
        dst, side = dest_for(group, src)
        rel_src = src.relative_to(REVIEW).as_posix()
        rel_dst = dst.as_posix().replace(PROJ.as_posix() + '/', '')
        if skip:
            rows.append(dict(time=ts, group=group, side=side, file=src.name,
                             src_rel=rel_src, dst_rel=rel_dst, size=src.stat().st_size,
                             sha256='', action=skip, note=why))
            continue
        dig = sha256_of(src)
        if dst.exists():
            if sha256_of(dst) == dig:
                rows.append(dict(time=ts, group=group, side=side, file=src.name,
                                 src_rel=rel_src, dst_rel=rel_dst, size=src.stat().st_size,
                                 sha256=dig, action='same', note='目标同哈希，跳过'))
                continue
            if args.update:
                if args.apply:
                    shutil.copy2(src, dst)
                rows.append(dict(time=ts, group=group, side=side, file=src.name,
                                 src_rel=rel_src, dst_rel=rel_dst, size=src.stat().st_size,
                                 sha256=dig, action='updated' if args.apply else 'dry-run-update',
                                 note='源已更新，按 --update 覆盖目标'))
                continue
            conflicts += 1
            rows.append(dict(time=ts, group=group, side=side, file=src.name,
                             src_rel=rel_src, dst_rel=rel_dst, size=src.stat().st_size,
                             sha256=dig, action='conflict', note='目标存在且内容不同，拒绝覆盖（可用 --update）'))
            continue
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        rows.append(dict(time=ts, group=group, side=side, file=src.name,
                         src_rel=rel_src, dst_rel=rel_dst, size=src.stat().st_size,
                         sha256=dig, action='copied' if args.apply else 'dry-run', note=''))

    n_copy = sum(1 for r in rows if r['action'] in ('copied', 'dry-run'))
    n_upd = sum(1 for r in rows if r['action'] in ('updated', 'dry-run-update'))
    n_same = sum(1 for r in rows if r['action'] == 'same')
    n_skip = sum(1 for r in rows if r['action'].startswith('skipped'))
    n_conf = sum(1 for r in rows if r['action'] == 'conflict')
    print(f'候选 {len(rows)}：可复制 {n_copy} / 更新 {n_upd} / 同哈希跳过 {n_same} / 排除 {n_skip} / 冲突 {n_conf}')
    for r in rows:
        tag = r['action']
        print(f'  [{tag}] {r["group"]}/{r["file"]} -> {r["dst_rel"]}' + (f'  ({r["note"]})' if r['note'] else ''))

    if args.apply:
        ATTACH_DEST_ROOT.mkdir(parents=True, exist_ok=True)
        CODE_DEST_ROOT.mkdir(parents=True, exist_ok=True)
        man1 = ATTACH_DEST_ROOT / '_manifest_archive_2026-10-15.csv'
        merged = merge_manifest(man1, rows)
        write_readme(merged, ts)
        print(f'manifest（合并式，{len(merged)} 行）：{man1}')
        print(f'manifest 副本：{CODE_DEST_ROOT / man1.name}')
        print(f'README：{ATTACH_DEST_ROOT / "README_附件说明.md"}；{CODE_DEST_ROOT / "README.md"}')
    else:
        print('（dry-run：未写盘。加 --apply 执行；可分趟 --groups a,b,...）')

    return 3 if conflicts else 0


if __name__ == '__main__':
    sys.exit(main())
