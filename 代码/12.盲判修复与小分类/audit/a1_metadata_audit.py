# -*- coding: utf-8 -*-
"""a1_metadata_audit.py — R/V/Q 侧表、严格盲包、角色/配对、空间分离与来源分层元数据审计

* 输入：
    --sidecar  ：私有侧表（parquet/csv），R/V/Q 全量点，必含
                 sample_uid, namespace, original_id, year, lon, lat, role, block_025, source, blind_id
                 可选 pair_id, pair_side, stratum, tile_id, label_before, pred_v1, pred_v2, native_px
    --blind    ：送判读者的盲包（csv）；--stage points 只允许 blind_id,lon,lat
                 --stage form 只允许 blind_id,实际类别,置信,证据,影像源,日期,指标提示
    --seen-blocks（可选）：旧已看 0.25° 块清单（csv: block,source 或 json: {block: source}）
    --sheet-map（可选）：build_sheets.py 的 sheet_map.json（核验配对点不同表）
* 规则源：doc72 §2/§3.1/§4/§5；判读协议 §2（盲判纪律）；doc69 §2.3（编号碰撞）/§9.2（空间独立）
* 门槛：盲包出现任何先验/角色/来源/分层/臂/配对/坡向字段 = FAIL；blind_id 非不透明 = FAIL；
        R 块 ∩ V 块 ≠ ∅ = FAIL；配对跨 role = FAIL；盲包含"结论（正确/错误）"列 = FAIL；
        V ∩ 旧已看块 = 记录（不设绝对无交门，doc72 未加）；来源分层缺层只报不判。
* 输出：<ws>/audit/out/metadata_audit_<ts>.json（含逐项 PASS/WARN/FAIL 与 gate）
* 用法：python a1_metadata_audit.py --sidecar private/sidecar.parquet --blind blind/points_blind.csv
        python a1_metadata_audit.py ... --stage form --sheet-map blind/sheet_map.json
* 说明：只读审计；schema 缺失显式报错，不猜测、不自动修补侧表。
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
SEEN_DEFAULT = (r'F:/lc_work/improve_2023_pilot_20261015/terrain/out/fview_g2/'
                r't4_fview_registration_G2v2.json')

REQ_SIDE = ['sample_uid', 'lon', 'lat', 'role', 'block_025', 'source', 'blind_id']
REC_SIDE = ['namespace', 'original_id', 'year']   # 谱系用，缺只报 INFO
ALLOW_POINTS = {'blind_id', 'point_id', 'lon', 'lat'}
ALLOW_FORM = {'blind_id', 'point_id', '实际类别', '置信', '证据', '影像源', '日期', '指标提示',
              '结构大类（林/灌/草/耕/园/水/建成/裸/湿/冰）', '叶型轴（针/阔/混交/无法判定）',
              '物候轴（常绿/落叶/无法判定）', '实际类别（24类中文名，或"无法判定"）',
              '覆盖度或管理特征（纯林/果园行栽/梯坎茶园/大棚/地膜/淹水等）',
              '置信（高/中/低）', '证据（影像特征+影像源+日期/季节+辅助指标）',
              '无法判定原因（云/阴影/分辨率/混合像元/无日期）'}
FORBIDDEN_VERDICT_COLS = ('结论', '正确', '错误', '是否一致', 'verdict')
# 影像溯源/计时类字段名：允许出现在 fetch_log 等元数据里（不代表先验泄漏）
ALLOW_META_FIELDS = {'id', 'source', 'provider', 'zoom', 'mpp', 'box_m', 'time', 'date',
                     'error', 'retries', 'ok', 'bytes', 'status', 'url', 'sheet', 'n',
                     'points', 'seq', 'img_date', 'crs', 'w', 'h'}
FORBIDDEN_TOKENS = ['prior', 'label', 'class', 'source', 'stratum', 'strata', 'arm',
                    'pair', 'aspect', 'yang', 'yin', 'role', 'orchard', 'forest',
                    'shrub', 'grass', '先验', '园地', '林', '灌', '坡', '阳', '阴', '角色']
CRED_PATTERNS = [r'AIza[0-9A-Za-z_\-]{20,}', r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
                 r'"private_key"', r'"client_email"', r'\bghp_[0-9A-Za-z]{20,}',
                 r'\bsk-[0-9A-Za-z]{20,}']
BLIND_ID_RE = re.compile(r'^B\d{4,}$')


def read_table(path):
    low = path.lower()
    if low.endswith(('.parquet', '.pq')):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    if low.endswith('.json'):
        d = json.load(open(path, encoding='utf-8'))
        return d if isinstance(d, list) else [{'block': k, 'source': v} for k, v in d.items()]
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def haversine(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def norm_block(b):
    """块键归一：'0p25_428_120' / '428,120' / '428_120' → '428,120'（否则 R/V 交集会假空）。"""
    s = str(b if b is not None else '').strip()
    if not s:
        return ''
    if s.lower().startswith('0p25_'):
        s = s[5:]
    parts = [p for p in s.replace('_', ',').replace('/', ',').replace(' ', ',').split(',') if p]
    if len(parts) >= 2:
        try:
            return '%d,%d' % (int(float(parts[0])), int(float(parts[1])))
        except ValueError:
            return s
    return s


def bbox_to_blocks(bbox, step=0.25):
    """[w,s,e,n] → 覆盖的 0.25° 块键集合（块键 = floor(lon/step), floor(lat/step)）。"""
    w, s, e, n = [float(x) for x in bbox]
    out = set()
    for i in range(int(math.floor(w / step)), int(math.ceil(e / step))):
        for j in range(int(math.floor(s / step)), int(math.ceil(n / step))):
            out.add('%d,%d' % (i, j))
    return out


def load_seen_blocks(paths):
    """合并多个已看块来源：{block: source} 形式，或含 seen_blocks_excluded_from_V(bbox 列表)。"""
    seen = {}
    for p in paths or []:
        if not p or not os.path.exists(p):
            continue
        tag = os.path.basename(p)
        if p.lower().endswith('.json'):
            d = json.load(open(p, encoding='utf-8'))
            for k, v in (d.get('seen_conf_blocks_registered') or {}).items():
                seen[norm_block(k)] = str(v)
            for bb in (d.get('seen_blocks_excluded_from_V') or []):
                for b in bbox_to_blocks(bb):
                    seen.setdefault(b, 'seen_bbox@%s' % tag)
            if not isinstance(d, dict):
                for r in d:
                    seen.setdefault(norm_block(r.get('block')), str(r.get('source', tag)))
        else:
            for r in read_table(p):
                b = norm_block(r.get('block', r.get('block_025', '')))
                if b:
                    seen.setdefault(b, str(r.get('source', r.get('src', tag))))
    return seen


def scan_cred(paths):
    hits = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            txt = open(p, encoding='utf-8-sig', errors='ignore').read()
        except Exception:
            continue
        for pat in CRED_PATTERNS:
            for m in re.finditer(pat, txt):
                hits.append({'file': p.replace('\\', '/'), 'pattern': pat,
                             'at': m.start(), 'sample': txt[max(0, m.start() - 20):m.start() + 40]})
    return hits


def apply_sidecar_map(rows, map_path):
    """map 为 {标准列名: 本侧表列名}；本函数按反向把侧表列名改成标准名。"""
    if not map_path:
        return rows
    m = json.load(open(map_path, encoding='utf-8'))
    inv = {v: k for k, v in m.items() if v and not str(v).startswith('__')}
    return [{inv.get(k, k): v for k, v in r.items()} for r in rows]


def id_col_of(rows, prefer='blind_id'):
    """盲包 id 列自动识别（本批用 point_id，值为 B####）。"""
    if not rows:
        return prefer
    cols = list(rows[0].keys())
    for c in (prefer, 'blind_id', 'point_id', 'id'):
        if c in cols:
            return c
    return cols[0]


def blind_only(a, ws, outd, ts):
    """只审盲包形态（sidecar 未到位时可先跑）：列白名单/无先验相对列/ID 不透明/四路闭合/配对同表/凭证。"""
    checks = []

    def add(level, code, detail):
        checks.append({'level': level, 'code': code, 'detail': detail})

    blind = read_table(a.blind)
    if not blind:
        raise SystemExit('盲包为空：%s' % a.blind)
    idc = a.id_col or id_col_of(blind)
    bcols = list(blind[0].keys())
    allow = ALLOW_POINTS if a.stage == 'points' else ALLOW_FORM
    extra = [c for c in bcols if c not in allow]
    add('FAIL' if extra else 'PASS', 'B1_blind_columns',
        '盲包(stage=%s)含白名单外列：%s' % (a.stage, extra) if extra
        else '盲包列 %s ⊆ 白名单（id 列=%s）' % (bcols, idc))
    wrong = [c for c in bcols if any(c.startswith(k) for k in FORBIDDEN_VERDICT_COLS)]
    add('FAIL' if wrong else 'PASS', 'B2_no_prior_relative_verdict',
        '盲批表单出现先验相对列 %s（doc72 §3.1：正确/错误必须解盲后生成）' % wrong if wrong
        else '盲批表单无"结论/正确/错误"先验相对列')

    bids = [str(r[idc]).strip() for r in blind]
    bad = [b for b in bids if not BLIND_ID_RE.match(b)]
    d2 = sorted({b for b in bids if bids.count(b) > 1})
    tokhit = [b for b in bids if any(t in b.lower() or t in b for t in FORBIDDEN_TOKENS)]
    lvl = 'FAIL' if (bad or d2 or tokhit) else 'PASS'
    add(lvl, 'B3_blind_id_opaque',
        '非 B#### %d（%s）；重复 %d（%s）；含敏感词 %d（%s）' % (len(bad), bad[:3], len(d2), d2[:3], len(tokhit), tokhit[:3])
        if lvl == 'FAIL' else 'blind_id 全部形如 B####、唯一、无敏感词（%d）' % len(bids))
    bs = set(bids)

    # 四路闭合：盲包 == 表单 == 切片名 == sheet_map == batches.csv
    if a.form and os.path.exists(a.form):
        fr = read_table(a.form)
        fic = id_col_of(fr)
        fs = {str(r[fic]).strip() for r in fr}
        add('PASS' if fs == bs else 'FAIL', 'B4a_form_closure',
            '表单 %d 点与盲包一致' % len(fs) if fs == bs
            else '表单与盲包不一致：表单独有 %s / 盲包独有 %s' % (sorted(fs - bs)[:5], sorted(bs - fs)[:5]))
    if a.chips and os.path.isdir(a.chips):
        ch = {f[:-4] for f in os.listdir(a.chips) if f.lower().endswith('.png')}
        add('PASS' if ch == bs else 'WARN', 'B4b_chips_closure',
            '切片 %d 件与盲包一致且文件名仅含 blind_id' % len(ch) if ch == bs
            else '切片与盲包不一致：切片独有 %s / 缺失 %s' % (sorted(ch - bs)[:5], sorted(bs - ch)[:5]))
        ctk = [c for c in ch if any(t in c.lower() or t in c for t in FORBIDDEN_TOKENS)]
        add('FAIL' if ctk else 'PASS', 'B4c_chip_name_no_class',
            '切片文件名含类别/角色敏感词：%s' % ctk[:5] if ctk else '切片文件名无类别/角色敏感词')
    if a.sheet_map and os.path.exists(a.sheet_map):
        smap = json.load(open(a.sheet_map, encoding='utf-8'))
        allids = [str(i) for v in smap.values() for i in v]
        add('PASS' if set(allids) == bs and len(allids) == len(bs) else 'FAIL', 'B4d_sheetmap_closure',
            'sheet_map %d 表 / %d 点，与盲包一一对应' % (len(smap), len(allids))
            if set(allids) == bs and len(allids) == len(bs)
            else 'sheet_map 与盲包不闭合：缺 %s / 多 %s / 重复 %d'
                 % (sorted(bs - set(allids))[:5], sorted(set(allids) - bs)[:5], len(allids) - len(set(allids))))
        sizes = sorted({len(v) for v in smap.values()})
        add('INFO', 'B4e_sheet_size', '每表点数取值 %s（2×2 应为 [4]；>4 会缩水）' % sizes)
        if a.sidecar and os.path.exists(a.sidecar):
            side = apply_sidecar_map(read_table(a.sidecar), a.sidecar_map)
            sc = id_col_of(side)
            bid2sheet = {str(i): sh for sh, v in smap.items() for i in v}
            pairs = {}
            for r in side:
                p = str(r.get('pair_id', '') or '').strip()
                if p:
                    pairs.setdefault(p, []).append(str(r[sc]).strip())
            # 只核"成员全在本批 sheet_map 内"的配对（跨批配对不适用同表核）
            full = {p: v for p, v in pairs.items() if all(b in bid2sheet for b in v)}
            same = [p for p, v in full.items() if len({bid2sheet[b] for b in v}) < len(v)]
            add('FAIL' if same else 'PASS', 'B5_pair_not_same_sheet',
                '配对成员落在同一联系表 %d 对：%s' % (len(same), same[:5]) if same
                else '本批内完整配对 %d 对（全 %d 对），成员均不同表' % (len(full), len(pairs)))
        else:
            add('WARN', 'B5_pair_not_same_sheet', '未提供 sidecar，配对是否同表待补核')
    else:
        add('WARN', 'B4d_sheetmap_closure', '未提供 --sheet-map')

    if a.fetch_log and os.path.exists(a.fetch_log):
        fl = json.load(open(a.fetch_log, encoding='utf-8'))
        if isinstance(fl, dict):
            fk = set(fl.keys())
            inner = set()
            for v in fl.values():
                if isinstance(v, dict):
                    inner |= set(v.keys())
            add('PASS' if fk == bs else 'WARN', 'B6a_fetchlog_closure',
                'fetch_log 键 %d 个与盲包一致' % len(fk) if fk == bs
                else 'fetch_log 键与盲包不一致：多 %s / 缺 %s' % (sorted(fk - bs)[:5], sorted(bs - fk)[:5]))
            leak = [k for k in inner if k not in ALLOW_META_FIELDS
                    and any(t in k.lower() or t in k for t in FORBIDDEN_TOKENS)]
            add('FAIL' if leak else 'PASS', 'B6b_fetchlog_no_prior',
                'fetch_log 出现非影像溯源且含先验/角色语义的字段：%s' % leak if leak
                else 'fetch_log 字段 %s 均为影像溯源/计时类，无先验/角色语义' % sorted(inner))
            add('INFO', 'B6c_fetchlog_time_semantics',
                '注意 fetch_log 的 `time` 是**下载时间**，不是拍摄日期（doc72 §3.2）；'
                '判读证据里的日期必须来自影像源/元数据，不得把下载时间当拍摄日期')

    scan_targets = [a.blind, a.form, a.sheet_map, a.fetch_log] + (
        [os.path.join(a.chips, f) for f in os.listdir(a.chips)][:0] if a.chips and os.path.isdir(a.chips) else [])
    ch = scan_cred([x for x in scan_targets if x and os.path.exists(x)])
    add('FAIL' if ch else 'PASS', 'B7_credential_scan',
        '命中疑似凭证 %d 处：%s' % (len(ch), ch[:3]) if ch else '盲包/表单/sheet_map/fetch_log 未命中疑似凭证')

    gate = 'FAIL' if any(c['level'] == 'FAIL' for c in checks) else (
        'WARN' if any(c['level'] == 'WARN' for c in checks) else 'PASS')
    rec = {'schema': 'blind_only_audit/v1', 'gate': gate,
           'audited_at': datetime.now().isoformat(timespec='seconds'),
           'stage': a.stage, 'blind': a.blind, 'id_col': idc, 'n_blind': len(blind),
           'checks': checks,
           'note': 'sidecar 未到位，仅审盲包形态；C1/C2/C5/C7/C9/C11 待全量 sidecar 后补跑'}
    fp = a.out or os.path.join(outd, 'blind_only_audit_%s.json' % ts)
    json.dump(rec, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for c in checks:
        print('[%s] %s — %s' % (c['level'], c['code'], c['detail']))
    print('gate=%s（n=%d）→ %s' % (gate, len(blind), fp))
    return gate


def main():
    ap = argparse.ArgumentParser(description='R/V/Q 侧表与严格盲包元数据审计')
    ap.add_argument('--ws', default=WS_DEFAULT)
    ap.add_argument('--sidecar', default=None, help='私有侧表（--blind-only 时可省）')
    ap.add_argument('--blind', required=True)
    ap.add_argument('--blind-only', action='store_true', help='仅审盲包形态（sidecar 未到位时先跑）')
    ap.add_argument('--stage', choices=['points', 'form'], default='points')
    ap.add_argument('--seen-blocks', default=SEEN_DEFAULT)
    ap.add_argument('--sheet-map', default=None)
    ap.add_argument('--form', default=None, help='盲批表单骨架/填表（四路闭合核）')
    ap.add_argument('--chips', default=None, help='切片目录（文件名闭合核）')
    ap.add_argument('--fetch-log', default=None)
    ap.add_argument('--batches', default=None)
    ap.add_argument('--id-col', default=None)
    ap.add_argument('--sidecar-map', default=None, help='JSON：{标准列名: 本侧表列名} 重命名后再审')
    ap.add_argument('--expect-r', type=int, default=120)
    ap.add_argument('--expect-v', type=int, default=80)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    outd = os.path.join(ws, 'audit', 'out')
    os.makedirs(outd, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if a.blind_only:
        g = blind_only(a, ws, outd, ts)
        sys.exit(2 if g == 'FAIL' else 0)
    if not a.sidecar:
        raise SystemExit('全量审计需要 --sidecar（或加 --blind-only 先审盲包形态）')
    checks = []

    def add(level, code, detail):
        checks.append({'level': level, 'code': code, 'detail': detail})

    side = apply_sidecar_map(read_table(a.sidecar), a.sidecar_map)
    blind = read_table(a.blind)
    if not side:
        raise SystemExit('侧表为空：%s' % a.sidecar)
    cols = list(side[0].keys())
    missing = [c for c in REQ_SIDE if c not in cols]
    if missing:
        add('FAIL', 'C1_sidecar_schema', '侧表缺必需列：%s（现有：%s）' % (missing, cols))
    else:
        add('PASS', 'C1_sidecar_schema', '侧表 %d 行，必需列齐全' % len(side))
        rec_missing = [c for c in REC_SIDE if c not in cols]
        add('INFO' if rec_missing else 'PASS', 'C1b_sidecar_lineage_cols',
            '缺谱系列 %s（不影响本轮检查，但 doc72 §7.1 要求可回溯）' % rec_missing if rec_missing
            else '谱系列齐全 %s' % REC_SIDE)
    if missing:
        json.dump({'schema': 'metadata_audit/v1', 'gate': 'FAIL', 'checks': checks},
                  open(os.path.join(outd, 'metadata_audit_%s.json' % ts), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        raise SystemExit('侧表 schema 不合格，停止（不猜测列名）')

    # C2 主键唯一
    uids = [str(r['sample_uid']) for r in side]
    dup = sorted({u for u in uids if uids.count(u) > 1})
    add('FAIL' if dup else 'PASS', 'C2_sample_uid_unique',
        '重复 sample_uid %d 个：%s' % (len(dup), dup[:5]) if dup else 'sample_uid 唯一（%d）' % len(uids))

    # C3 blind_id 不透明且唯一
    bids = [str(r['blind_id']).strip() for r in side]
    bad = [b for b in bids if not BLIND_ID_RE.match(b)]
    tokhit = [b for b in bids if any(t in b.lower() or t in b for t in FORBIDDEN_TOKENS)]
    d2 = sorted({b for b in bids if bids.count(b) > 1})
    lvl = 'FAIL' if (bad or tokhit or d2) else 'PASS'
    add(lvl, 'C3_blind_id_opaque',
        '非 B#### 形式 %d（%s）；含敏感词 %d（%s）；重复 %d（%s）' % (
            len(bad), bad[:3], len(tokhit), tokhit[:3], len(d2), d2[:3]) if lvl == 'FAIL'
        else 'blind_id 全部形如 B####、无敏感词、唯一（%d）' % len(bids))

    # C4 盲包列白名单
    idc = a.id_col or id_col_of(blind)
    bcols = list(blind[0].keys()) if blind else []
    allow = ALLOW_POINTS if a.stage == 'points' else ALLOW_FORM
    extra = [c for c in bcols if c not in allow]
    if extra:
        add('FAIL', 'C4_blind_columns', '盲包(stage=%s)含白名单外列：%s（白名单：%s）'
            % (a.stage, extra, sorted(allow)))
    else:
        add('PASS', 'C4_blind_columns', '盲包列 %s ⊆ 白名单（id 列=%s）' % (bcols, idc))
    if a.stage == 'form':
        wrong = [c for c in bcols if any(c.startswith(k) for k in FORBIDDEN_VERDICT_COLS)]
        add('FAIL' if wrong else 'PASS', 'C4b_no_prior_relative_verdict',
            '盲批表单出现先验相对列 %s（doc72 §3.1：正确/错误必须解盲后生成）' % wrong if wrong
            else '盲批表单无"正确/错误"先验相对列')

    # C5 blind_id 集合一致
    bs = {str(r[idc]).strip() for r in blind}
    ss = set(bids)
    orph, miss = sorted(bs - ss), sorted(ss - bs)
    add('FAIL' if (orph or miss) else 'PASS', 'C5_id_set_match',
        '盲包独有 %d（%s）；侧表独有 %d（%s）' % (len(orph), orph[:5], len(miss), miss[:5])
        if (orph or miss) else '盲包与侧表 blind_id 集合一致（%d）' % len(bs))

    # C6 角色域与计划量
    roles = {}
    for r in side:
        roles[str(r['role'])] = roles.get(str(r['role']), 0) + 1
    other = {k: v for k, v in roles.items() if k not in ('R', 'V', 'Q')}
    add('FAIL' if other else 'PASS', 'C6_role_domain',
        '非法 role：%s' % other if other else 'role 计数 %s' % roles)
    for k, exp in (('R', a.expect_r), ('V', a.expect_v)):
        got = roles.get(k, 0)
        add('INFO' if got == exp else 'WARN', 'C6b_plan_%s' % k,
            '%s=%d（计划 %d，差 %+d）；doc72 §2 允许因材料/空间支持不足记录实际数量，不复制凑数'
            % (k, got, exp, got - exp))

    # C7 配对完整性：同对同 role
    pairs = {}
    for r in side:
        p = str(r.get('pair_id', '') or '').strip()
        if p:
            pairs.setdefault(p, []).append(str(r['role']))
    mixed = {p: sorted(set(v)) for p, v in pairs.items() if len(set(v)) > 1}
    small = {p: len(v) for p, v in pairs.items() if len(v) < 2}
    lvl = 'FAIL' if mixed else ('WARN' if small else 'PASS')
    add(lvl, 'C7_pair_same_role',
        '跨 role 配对 %d：%s；成员<2 的对 %d：%s' % (len(mixed), list(mixed.items())[:3], len(small), list(small.items())[:3])
        if lvl != 'PASS' else '配对 %d 对，全部同 role、成员≥2' % len(pairs))

    # C8 配对不同联系表（可选）
    if a.sheet_map and os.path.exists(a.sheet_map):
        smap = json.load(open(a.sheet_map, encoding='utf-8'))
        bid2sheet = {}
        for sheet, ids in smap.items():
            for i in ids:
                bid2sheet[str(i)] = sheet
        full = {p: [str(r['blind_id']) for r in side
                    if str(r.get('pair_id', '') or '').strip() == p] for p in pairs}
        full = {p: v for p, v in full.items() if all(b in bid2sheet for b in v)}
        same = [p for p, v in full.items() if len({bid2sheet[b] for b in v}) < len(v)]
        add('FAIL' if same else 'PASS', 'C8_pair_not_same_sheet',
            '配对成员落在同一联系表 %d 对：%s' % (len(same), same[:5]) if same
            else '本 sheet_map 内完整配对 %d 对（全 %d 对），成员均不同表' % (len(full), len(pairs)))
    else:
        add('WARN', 'C8_pair_not_same_sheet', '未提供 --sheet-map，无法核验配对是否同表（配对泄漏风险未排除）')

    # C9 R/V 块级分离（块键先归一，否则 '0p25_428_120' vs '428,120' 会假空）
    def blocks(role):
        return {norm_block(r['block_025']) for r in side if str(r['role']) == role}
    bR, bV = blocks('R'), blocks('V')
    inter = sorted(bR & bV)
    add('FAIL' if inter else 'PASS', 'C9_RV_block_disjoint',
        'R 与 V 共用块 %d 个：%s（doc72 §2 要求空间分离）' % (len(inter), inter[:8]) if inter
        else 'R(%d 块) ∩ V(%d 块) = ∅（块键已归一）' % (len(bR), len(bV)))

    # C9b R/V 点级最小距离（记录，不设绝对门）
    rp = [(float(r['lon']), float(r['lat'])) for r in side if str(r['role']) == 'R']
    vp = [(float(r['lon']), float(r['lat'])) for r in side if str(r['role']) == 'V']
    if rp and vp:
        dmin = min(haversine(x1, y1, x2, y2) for x1, y1 in rp for x2, y2 in vp)
        add('WARN' if dmin < 1000 else 'INFO', 'C9b_RV_min_distance',
            'R-V 最近点 %.0f m（<1 km 时空间自相关风险高，doc69 §9.2：1 km 不自动等于独立）' % dmin)

    # C10 V 与旧已看块：只记录（doc72 未设"与全部历史评价块无交"的绝对门）
    seen_paths = [a.seen_blocks, os.path.join(ws, 'private', 'experiment_context.json')]
    seen = load_seen_blocks(seen_paths)
    if seen:
        ov = sorted(bV & set(seen))
        add('INFO', 'C10_V_vs_seen_blocks',
            'V 块 %d，其中与旧已看块重合 %d（%s）；doc72 未设"与全部历史评价块无交"的绝对门，'
            '此处只记录实际使用范围，重合即写清来源：%s'
            % (len(bV), len(ov), ov[:8], {b: seen.get(b) for b in ov[:8]}))
        add('INFO', 'C10b_V_actual_range',
            'V 实际使用块 %d 个（新块 %d）；R 实际使用块 %d 个；已看块清单合并来源 %s（共 %d 块）'
            % (len(bV), len(bV - set(seen)), len(bR),
               [os.path.basename(p) for p in seen_paths if p and os.path.exists(p)], len(seen)))
    else:
        add('WARN', 'C10_V_vs_seen_blocks', '未找到旧已看块清单（%s），V 空间范围未与历史评价块对照' % seen_paths)

    # C11 来源分层覆盖（不扔不利源）
    src = {}
    for r in side:
        src.setdefault(str(r['source']), {'R': 0, 'V': 0})
        src[str(r['source'])][str(r['role'])] = src[str(r['source'])].get(str(r['role']), 0) + 1
    add('INFO', 'C11_source_strata', '来源×角色计数：%s' % json.dumps(src, ensure_ascii=False))
    only_r = [k for k, v in src.items() if v.get('R', 0) and not v.get('V', 0)]
    if only_r:
        add('INFO', 'C11b_source_only_in_R',
            '仅出现在 R 的来源 %s：报告须保留该层，不得只报有利来源' % only_r)

    # C12 凭证扫描
    scan_targets = [a.blind, a.sidecar]
    for d in ('blind', 'materials', 'private'):
        dd = os.path.join(ws, d)
        if os.path.isdir(dd):
            for root, _, files in os.walk(dd):
                scan_targets += [os.path.join(root, f) for f in files]
    ch = scan_cred(scan_targets)
    add('FAIL' if ch else 'PASS', 'C12_credential_scan',
        '命中疑似凭证 %d 处：%s' % (len(ch), ch[:3]) if ch else '盲包/材料/私有目录未命中疑似凭证')

    gate = 'FAIL' if any(c['level'] == 'FAIL' for c in checks) else (
        'WARN' if any(c['level'] == 'WARN' for c in checks) else 'PASS')
    rec = {'schema': 'metadata_audit/v1', 'gate': gate,
           'audited_at': datetime.now().isoformat(timespec='seconds'),
           'stage': a.stage, 'sidecar': a.sidecar, 'blind': a.blind,
           'n_side': len(side), 'n_blind': len(blind), 'roles': roles,
           'checks': checks,
           'policy': ['本审计为 source-aware（知来源/角色），不参与盲判，不读判读结论',
                      'V 点级真值不进入实验代理可见产物']}
    fp = a.out or os.path.join(outd, 'metadata_audit_%s.json' % ts)
    json.dump(rec, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for c in checks:
        print('[%s] %s — %s' % (c['level'], c['code'], c['detail']))
    print('gate=%s → %s' % (gate, fp))
    if gate == 'FAIL':
        sys.exit(2)


if __name__ == '__main__':
    main()
