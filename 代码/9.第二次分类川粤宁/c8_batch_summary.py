# -*- coding: utf-8 -*-
"""c8_batch_summary.py — 第二次分类（川/粤/宁 40 瓦，2023）生产收官摘要生成器
* 输入（全部只读）：F:/lc_work/prod2nd_2023/{ledger/board_2nd.csv, ledger/eecu_ledger_2nd.csv,
  run/batch/state.json, qa/{tile}_qa.json + {tile}_面积.csv, plan/tiles_secondrun.csv,
  config.json}；驱动日志 F:/lc_work/v31_exp/logs/prod2.log（可选，缺失只降级不报错）
* 规则源：列口径与状态语义对齐 run/prod2_driver.py（board()/qa()/ledger()/poll_tile()）；
  成本带 OVER_LIMIT=24.0（第一次五省批 2026-10-04 G4 裁决）；预算 950 EECU·h（config.json budget_eecu）；
  第一次对照数用小批台账（prod5p_2023，只读）同口径复算，不用手写估值
* 门槛：覆盖率双口径分报（coverage_px_pct / coverage_km2_pct，doc 43 纪律）；逐瓦数字只允许来自
  board_2nd.csv + qa json（+ state.json 的保真/赛跑、日志事件），脚本内不写任何人工估值
* 输出：<reports>/batch_summary_2nd.md（版式对齐 prod5p_2023/reports/batch_summary.md）
  + <reports>/eecu_ledger_2nd_normalized.csv（双 schema 台账 → 统一 9 列）
  --smoke：改用第一次五省批台账（只读）在 <reports>/_smoke/ 出同版式样例，证明版式与计算
* 用法：python 代码/9.第二次分类川粤宁/c8_batch_summary.py [--reports-dir DIR] [--smoke] [--log FP]
"""
import os
import re
import sys
import csv
import json
import math
import time
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------------- 路径常量（一律 raw 绝对路径） ----------------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
ROOT2 = r'F:/lc_work/prod2nd_2023'                      # 第二次分类工作区（输入，只读）
ROOT1 = r'F:/lc_work/prod5p_2023'                       # 第一次五省批（冒烟输入，只读，勿写）
LOG2 = r'F:/lc_work/v31_exp/logs/prod2.log'             # 第二次驱动日志
LOG1 = r'F:/lc_work/v31_exp/logs/b1_batch.log'          # 第一次五省批驱动日志（冒烟）
DEFAULT_REPORTS = os.path.join(ROOT2, 'reports')        # 唯一允许写出的根目录
BUDGET_DEFAULT = 950.0                                  # config.json 缺失时的预算回退值
OVER_LIMIT = 24.0                                       # 内陆单瓦 EECU 上限（G4 校准带）
QA_EDGE = 0.01                                          # 孤类阈值（与驱动 qa() 一致，仅用于读数说明）
SNAPSHOT_MAX_ROWS = 200                                 # 逐瓦表行数上限（超限只截断显示，不截断计算）

PROV_FULL = {'川': '四川', '粤': '广东', '宁': '宁夏', '闽': '福建', '黑': '黑龙江',
             '渝': '重庆', '贵': '贵州', '云': '云南', '甘': '甘肃', '陕': '陕西',
             '青': '青海', '藏': '西藏', '新': '新疆', '内': '内蒙古', '桂': '广西'}
ACTIVE_STATES = ('SUBMITTED', 'PENDING', 'RUNNING', 'READY')   # 驱动 ACTIVE 同口径
BOARD_KNOWN = ('QA_OK', 'BLOCKED', 'DONE', 'FAILED', 'WAIT') + ACTIVE_STATES


# ---------------- 基础 IO ----------------
def jload(fp, default=None):
    """容错读 JSON；失败返回 default（缺失不抛异常，交由口径核对段报出）。"""
    try:
        with open(fp, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def read_rows(fp):
    """读 CSV 为 (header_or_None, rows, n_header_skipped)；自动吃 BOM。
    表头识别 = 首列出现 'time'/'tile'（数据行首列是日期或瓦片号，不会撞）——
    台账只在**文件首行**有 5 列上传段表头、生产段 8 列行没有表头，故表头可能只覆盖前半段；
    这里对任意位置的表头行一律剔除并计数，避免把表头当数据行。行保留原始列数（双 schema 分支靠它）。"""
    if not os.path.exists(fp):
        return None, [], 0
    with open(fp, encoding='utf-8-sig', newline='') as f:
        raw = [r for r in csv.reader(f) if r and any(x.strip() for x in r)]
    if not raw:
        return None, [], 0
    head = raw[0] if str(raw[0][0]).strip().lower() in ('time', 'tile') else None
    rows = [r for r in raw if str(r[0]).strip().lower() not in ('time', 'tile')]
    return head, rows, len(raw) - len(rows)


def fnum(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def mt(paths):
    """返回给定时点列表中的最大 mtime（数据时点），无有效文件返回 None。"""
    ts = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return max(ts) if ts else None


def tstr(ts, fmt='%Y-%m-%d %H:%M'):
    return time.strftime(fmt, time.localtime(ts)) if ts else '—'


def pearson(xs, ys):
    """皮尔逊相关系数（stdlib 实现，n<3 或零方差返回 None）。"""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# ---------------- 数据装载 ----------------
def load_plan(fp):
    """plan/tiles_secondrun.csv（列 tile,ix,iy,x0,y0,x1,y1,land_frac,prov,prov_area_km2）
    → {tile: {prov, land_frac, zone, prov_area_km2}}；zone 判定与驱动一致（land_frac<1=沿海）。"""
    _, rows, _ = read_rows(fp)
    out = {}
    for r in rows:
        if len(r) < 9:
            continue
        lf = fnum(r[7], 1.0)
        out[r[0].strip()] = dict(prov=r[8].strip(), land_frac=lf,
                                 zone='沿海' if lf < 1.0 else '内陆',
                                 prov_area_km2=fnum(r[9], 0.0) if len(r) > 9 else None)
    return out


def load_board(fp):
    """ledger/board_2nd.csv → 有序 dict 列表（tile,batch,state,acct,asset,eecu,qa_done）。"""
    head, rows, _ = read_rows(fp)
    if not head:
        return []
    idx = {k.strip(): i for i, k in enumerate(head)}
    need = ['tile', 'state', 'acct', 'eecu']
    if any(k not in idx for k in need):
        return []

    def cell(r, k):
        return r[idx[k]].strip() if k in idx and idx[k] < len(r) else ''

    out = []
    for r in rows:
        if len(r) < len(head):
            r = r + [''] * (len(head) - len(r))
        out.append(dict(tile=cell(r, 'tile'), batch=cell(r, 'batch'), state=cell(r, 'state'),
                        acct=cell(r, 'acct'), asset=cell(r, 'asset'), eecu=fnum(cell(r, 'eecu')),
                        qa_done=cell(r, 'qa_done') in ('1', 'True')))
    return out


def bucket_state(s):
    """board 状态 → 汇总桶（与驱动 board() 的取值域一一对应）。"""
    if s in ACTIVE_STATES:
        return '在跑'
    if s == 'WAIT':
        return '待提交'
    return s if s in ('QA_OK', 'DONE', 'BLOCKED', 'FAILED') else '其他'


def load_qa(qa_dir, tile):
    """qa/{tile}_qa.json（双口径覆盖率/海面/类数/面积）；不存在返回 None。"""
    return jload(os.path.join(qa_dir, '%s_qa.json' % tile))


def _norm_core(raw):
    """双 schema 台账行 → 统一 9 列 [time,phase,acct,tile,task_id,desc,state,eecu_h,note]。
    分支规则（按列数，不按行序，保证乱序/交错也可用）：
      5 列 = 上传段（c1_upload_pool_v2.py：time,acct,task_id,desc,eecu_h）→ phase=upload，空列补空
      8 列 = 生产段（prod2_driver.ledger()：time,acct,tile,task_id,desc,state,eecu_h,note）→ phase=prod
    其他列数 → phase=raw 并按位补空，计入 anomalies。返回 (rows, schema 计数, anomalies)。"""
    out, sch, anomalies = [], {5: 0, 8: 0}, []
    for r in raw:
        if len(r) == 5:
            sch[5] += 1
            out.append([r[0], 'upload', r[1], '', r[2], r[3], '', r[4], ''])
        elif len(r) == 8:
            sch[8] += 1
            out.append([r[0], 'prod', r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
        else:
            anomalies.append(r)
            r2 = (list(r) + [''] * 9)[:9]
            out.append([r2[0], 'raw', r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], r2[7]])
    return out, sch, anomalies


def normalize_ledger(fp):
    """读台账 + 规范化 + 统计。返回 (rows, stats, anomalies)。"""
    head_row, raw, n_head = read_rows(fp)
    out, sch, anomalies = _norm_core(raw)
    prod = [x for x in out if x[1] == 'prod']
    stats = dict(n_total=len(out), n_upload=sch[5], n_prod=sch[8], n_raw=len(anomalies),
                 n_head=n_head, head_schema=(len(head_row) if head_row else 0),
                 tiles=sorted(set(x[3] for x in prod if x[3])),
                 states={}, notes={})
    for x in prod:
        stats['states'][x[6]] = stats['states'].get(x[6], 0) + 1
        stats['notes'][x[8]] = stats['notes'].get(x[8], 0) + 1
    return out, stats, anomalies


LOG_PATTERNS = [
    ('start', re.compile(r'b1_batch 启动：(.*)$')),
    ('round', re.compile(r'本轮汇总：EECU=([\d.]+)/(\d+)')),
    ('submit', re.compile(r'(T\d+) 提交路 (\w) @(\S+) → (\S+)')),
    ('tstate', re.compile(r'(T\d+) 路(\w) (\w+)→(\w+) EECU=([\d.]+)')),
    ('race', re.compile(r'(T\d+) 触发赛跑（(.*?)）')),
    ('fuse', re.compile(r'⛔\s*(.*)$')),
    ('dead', re.compile(r'账号 (\S+) 连续异常 (\d+) 次，摘除（(.*?)）')),
    ('over', re.compile(r'(T\d+) 非沿海 EECU=([\d.]+) > ([\d.]+)')),
    ('fid_re', re.compile(r'(T\d+) 保真未达 100%（第 (\d+) 次）')),
    ('dl_ok', re.compile(r'(T\d+) 下载完成（([\d.]+) 分钟，([\d.]+) MB）')),
    ('dl_fail', re.compile(r'(T\d+) 下载失败')),
    ('finish', re.compile(r'=== 总结（(.*?)）：EECU=([\d.]+)/(\d+)，完成 (\d+)/(\d+) ===')),
]


def parse_log(fp):
    """驱动日志 → 事件字典。缺失/不可读返回 {ok:False}（报告只降级注明，不报错）。"""
    ev = dict(ok=False, path=fp, n_lines=0, last_ts=None, start=[], round_last=None,
              submit=[], tstate=[], race=[], fuse=[], dead=[], over=[], fid_re=[],
              dl_ok=[], dl_fail=[], finish=[])
    if not fp or not os.path.exists(fp):
        return ev
    ev['ok'] = True
    ts_re = re.compile(r'^\[(\d\d-\d\d \d\d:\d\d:\d\d)\]')
    with open(fp, encoding='utf-8', errors='replace') as f:
        for line in f:
            ev['n_lines'] += 1
            m = ts_re.match(line)
            if m:
                ev['last_ts'] = m.group(1)
            for kind, rx in LOG_PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                if kind == 'round':
                    ev['round_last'] = dict(eecu=fnum(m.group(1)), cap=fnum(m.group(2)))
                elif kind == 'start':
                    ev['start'].append(m.group(1))
                elif kind == 'finish':
                    ev['finish'].append(m.groups())
                else:
                    ev[kind].append(m.groups())
                break
    return ev


# ---------------- 统计分析 ----------------
def analyze(src):
    """装载 + 计算。src = 路径字典（见 build_src）。返回上下文 ctx。"""
    plan = load_plan(src['plan'])
    board = load_board(src['board'])
    ledger_rows, led_stats, led_anom = normalize_ledger(src['ledger'])
    state = jload(src['state'], {}) or {}
    ev = parse_log(src['log'])
    cfg = jload(src['config'], {}) or {}
    budget = fnum(src.get('budget'), fnum(cfg.get('budget_eecu'), BUDGET_DEFAULT))

    tiles, seen = [], set()
    for b in board:                       # 台账（驱动 ORDER）为主序
        if b['tile'] not in seen:
            tiles.append(b)
            seen.add(b['tile'])
    plan_only = [t for t in plan if t not in seen]          # plan 有、board 无（应为本批外瓦片）
    board_only = [b['tile'] for b in board if b['tile'] not in plan]

    recs = []
    for b in tiles:
        t = b['tile']
        meta = plan.get(t, {})
        q = load_qa(src['qa'], t) or {}
        st_t = (state.get('tiles', {}) or {}).get(t, {}) or {}
        runs = st_t.get('runs', {}) or {}
        fid = st_t.get('fidelity', {}) or {}
        recs.append(dict(
            tile=t, batch=b['batch'] or st_t.get('batch', ''), state=b['state'],
            acct=b['acct'], eecu=b['eecu'], qa_done=b['qa_done'],
            prov=meta.get('prov') or q.get('prov') or '?',
            land_frac=(q.get('land_frac') if q.get('land_frac') is not None else meta.get('land_frac')),
            zone=q.get('zone') or meta.get('zone') or '?',
            prov_area_km2=meta.get('prov_area_km2'),
            qa_present=bool(q), qa=q,
            fidelity=fid.get('rate'),
            n_runs=len(runs), runs=runs,
            blocked=bool(st_t.get('blocked')),
        ))

    def bucket(s):
        return bucket_state(s)

    n_by = {}
    for r in recs:
        k = bucket_state(r['state'])
        n_by[k] = n_by.get(k, 0) + 1
    total_eecu = round(sum(r['eecu'] for r in recs), 2)

    # 状态分桶 + 成本带。口径关键：**在跑瓦的 eecu 是 GEE 进行中的累计读数（非终值）**，
    # 成本带/均值/超带判定只用终态瓦（QA_OK/DONE/BLOCKED/FAILED），与第一次五省批"收官后终值"同口径；
    # 在跑瓦只在概览里单列"在跑累计（非终值）"。
    def stats_of(rs):
        e = [r['eecu'] for r in rs if r['eecu'] > 0]
        if not e:
            return dict(n=0, mean=None, lo=None, hi=None)
        return dict(n=len(e), mean=sum(e) / len(e), lo=min(e), hi=max(e))

    TERMINAL = ('QA_OK', 'DONE', 'BLOCKED', 'FAILED')
    inland_all = [r for r in recs if r['land_frac'] == 1.0]
    coast_all = [r for r in recs if r['land_frac'] is not None and r['land_frac'] < 1.0]
    fin = [r for r in recs if r['state'] in TERMINAL]
    run = [r for r in recs if r['state'] not in TERMINAL and r['eecu'] > 0]
    fin_in = [r for r in fin if r['land_frac'] == 1.0]
    fin_co = [r for r in fin if r['land_frac'] is not None and r['land_frac'] < 1.0]
    s_in, s_co = stats_of(fin_in), stats_of(fin_co)
    s_run, s_run_in, s_run_co = stats_of(run), stats_of([r for r in run if r['land_frac'] == 1.0]), \
        stats_of([r for r in run if r['land_frac'] is not None and r['land_frac'] < 1.0])
    over = [r for r in fin_in if r['eecu'] > OVER_LIMIT]
    over_run = [r for r in run if r['land_frac'] == 1.0 and r['eecu'] > OVER_LIMIT]
    coast_sorted = sorted([r for r in coast_all if r['eecu'] > 0 or r['state'] in TERMINAL],
                          key=lambda r: (r['land_frac'] if r['land_frac'] is not None else 2))

    # 逐省小计（按 plan 中该省在本批的瓦序）
    provs = []
    for r in recs:
        if r['prov'] not in provs:
            provs.append(r['prov'])
    prov_tot = []
    for p in provs:
        rs = [r for r in recs if r['prov'] == p]
        prov_tot.append(dict(prov=p, full=PROV_FULL.get(p, p), n=len(rs),
                             qa_ok=sum(1 for r in rs if r['state'] == 'QA_OK'),
                             run=sum(1 for r in rs if bucket_state(r['state']) == '在跑'),
                             wait=sum(1 for r in rs if bucket_state(r['state']) == '待提交'),
                             eecu=round(sum(r['eecu'] for r in rs), 2),
                             s=stats_of([r for r in rs if r['state'] in TERMINAL])))

    multis = [r for r in recs if r['n_runs'] > 1]
    fid_bad = [r for r in recs if r['fidelity'] is not None and r['fidelity'] != 1.0]
    fid_ok = [r for r in recs if r['fidelity'] == 1.0]
    qa_cnt = sum(1 for r in recs if r['qa_present'])
    qa_done_cnt = sum(1 for r in recs if r['qa_done'])
    miss_qa = [r['tile'] for r in recs if r['state'] == 'QA_OK' and not r['qa_present']]

    # config.note 里的期望逐省计数（如"川24/粤10/宁6"）用于口径核对；解析失败则跳过
    exp_prov = {}
    for m in re.finditer(r'([\u4e00-\u9fa5])(\d+)\s*瓦?', str(cfg.get('note', ''))):
        if m.group(1) in PROV_FULL:
            exp_prov[m.group(1)] = int(m.group(2))

    finished = bool(state.get('finished'))
    why = state.get('why', '')
    last_round = (ev.get('round_last') or {})
    data_ts = mt([src['board'], src['ledger'], src['state']])
    extra_ts = mt([os.path.join(src['qa'], '%s_qa.json' % r['tile']) for r in recs[:5]])

    ctx = dict(src=src, plan=plan, board=board, state=state, ev=ev, cfg=cfg, budget=budget,
               recs=recs, n_by=n_by, total_eecu=total_eecu, terminal=TERMINAL,
               inland=inland_all, coast=coast_all, fin=fin, run=run,
               s_in=s_in, s_co=s_co, s_run=s_run, s_run_in=s_run_in, s_run_co=s_run_co,
               over=over, over_run=over_run, coast_sorted=coast_sorted, prov_tot=prov_tot,
               multis=multis, fid_bad=fid_bad, fid_ok=fid_ok, qa_cnt=qa_cnt,
               qa_done_cnt=qa_done_cnt, miss_qa=miss_qa, plan_only=plan_only,
               board_only=board_only, led_stats=led_stats, led_anom=led_anom,
               ledger_rows=ledger_rows, finished=finished, why=why, last_round=last_round,
               data_ts=data_ts, extra_ts=extra_ts, exp_prov=exp_prov,
               gen_ts=time.time(), smoke=bool(src.get('smoke')))
    return ctx


def ref_stats(src_ref):
    """第一次五省批同口径复算（只读）：用于对照行；不可用返回 None。"""
    if not os.path.exists(src_ref['board']):
        return None
    plan = load_plan(src_ref['plan'])
    board = load_board(src_ref['board'])
    if not board:
        return None
    vals = []
    for b in board:
        m = plan.get(b['tile'], {})
        lf = m.get('land_frac')
        vals.append((lf, b['eecu']))
    inl = [e for lf, e in vals if lf == 1.0 and e > 0]
    coa = [e for lf, e in vals if lf is not None and lf < 1.0 and e > 0]

    def st(e):
        return dict(n=len(e), mean=(sum(e) / len(e) if e else None),
                    lo=(min(e) if e else None), hi=(max(e) if e else None))
    return dict(n_tiles=len(board), total=round(sum(b['eecu'] for b in board), 1),
                s_in=st(inl), s_co=st(coa),
                over=[b['tile'] for b in board
                      if plan.get(b['tile'], {}).get('land_frac') == 1.0 and b['eecu'] > OVER_LIMIT],
                src=src_ref)


# ---------------- 渲染 ----------------
def fm(v, nd=2):
    return '—' if v is None else ('%.*f' % (nd, v))


def range_str(s):
    if s['n'] == 0:
        return '—'
    return '%.1f–%.1f' % (s['lo'], s['hi'])


def render(ctx, ref):
    s = ctx['src']
    L = []
    A = L.append
    stamp = tstr(ctx['gen_ts'], '%Y-%m-%d %H:%M')
    # ---- 头部 ----
    A('# %s生产总结%s' % (s['title'], '（SMOKE 自检）' if ctx['smoke'] else ''))
    A('')
    if ctx['smoke']:
        A('> **SMOKE 自检模式**：本文件由 `c8_batch_summary.py --smoke` 用**第一次五省批**数据')
        A('> （`F:/lc_work/prod5p_2023/`，只读）生成，仅验证版式与计算口径（对照 `reports/batch_summary.md`），')
        A('> **不是**第二次分类的交付摘要；正式摘要见 `<reports>/batch_summary_2nd.md`。')
        A('')
    A('* 生成时间：%s（`%s`）' % (stamp, s['script']))
    A('* **数据时点（快照）**：%s（board / 台账 / state.json 三者 mtime 最大值）%s'
      % (tstr(ctx['data_ts']), '；QA json 首瓦 mtime %s' % tstr(ctx['extra_ts'])
         if ctx['extra_ts'] else ''))
    A('* 驱动：`%s`（幂等断点续跑 + 单实例锁，批预算闸门 ≤%.0f EECU·h）；日志 `%s`（%s 行，末行 %s）'
      % (s['driver'], ctx['budget'], s['log'], ctx['ev']['n_lines'],
         ctx['ev']['last_ts'] or '—'))
    st = ('**收官**（state.why=%s，finished=%s）' % (ctx['why'] or '—', ctx['state'].get('finished', '—'))
          if ctx['finished'] else '**进行中**（state.json 无 finished/why，驱动仍在跑）')
    A('* 驱动状态：%s' % st)
    if ctx['last_round']:
        A('* 日志末轮汇总：EECU=%.2f/%.0f（快照时的在跑读数，非终值）'
          % (ctx['last_round']['eecu'], ctx['last_round']['cap']))
    A('* 冻结配置：%s；样本 %s；remap=%s'
      % (('`%s`' % ctx['cfg']['asset']) if ctx['cfg'].get('asset')
         else ('第一次五省批口径（见 %s）' % os.path.join(ROOT1, 'reports', 'batch_summary.md')
               if ctx['smoke'] else '—'),
         s['pool_note'], ctx['cfg'].get('remap', '—')))
    A('* 逐瓦数字来源：`%s`（状态/账号/EECU）+ `qa/{tile}_qa.json`（覆盖率双口径/海面/类数/面积）'
      % os.path.basename(s['board']))
    if ctx['smoke']:
        self_t = ['2026-10-14 03:53:27,e0p36771,36LGGALQLCDRGOOZHWQPNZCF,p2sh_0000_1014033557,0.0000']
        self_p = ['2026-10-14 13:49:30,e0p36771,T1304,A7XE62AUVT3TQZABZ2KYBWR3,'
                  'p5t_T1304_a_1014134929,SUBMITTED,0.0000,submit']
        o1 = normalize_ledger_rows([next(csv.reader([ln])) for ln in self_t])[0]
        o2 = normalize_ledger_rows([next(csv.reader([ln])) for ln in self_p])[0]
        A('* **规范化自检（固定样例，证明双 schema 分支）**：')
        A('  * 5 列样例 → `%s`' % ' | '.join(o1))
        A('  * 8 列样例 → `%s`' % ' | '.join(o2))
    A('')

    # ---- 一、总览 ----
    A('## 一、总览')
    A('')
    A('| 项 | 值 |')
    A('|---|---|')
    A('| 瓦片总数 | **%d**（%s） |' % (len(ctx['recs']),
      '，'.join('%s%d' % (p['full'], p['n']) for p in ctx['prov_tot'])))
    n = ctx['n_by']
    A('| 状态分布 | QA_OK **%d** / DONE %d / BLOCKED %d / FAILED %d / 在跑 %d / 待提交 %d%s |'
      % (n.get('QA_OK', 0), n.get('DONE', 0), n.get('BLOCKED', 0), n.get('FAILED', 0),
         n.get('在跑', 0), n.get('待提交', 0),
         ' / **其他 %d**（%s）' % (n['其他'], '，'.join(
             sorted(set(r['state'] for r in ctx['recs'] if r['state'] not in BOARD_KNOWN))))
         if n.get('其他') else ''))
    A('| 总 EECU·h（含在跑已回报） | **%.2f / %.0f**（余量 %.2f，占预算 %.1f%%） |'
      % (ctx['total_eecu'], ctx['budget'], ctx['budget'] - ctx['total_eecu'],
         100.0 * ctx['total_eecu'] / ctx['budget'] if ctx['budget'] else 0))
    if ctx['run']:
        A('| 其中：在跑瓦累计（**非终值**） | %d 瓦合计 %.2f（内陆 %d 瓦 %.2f / 沿海 %d 瓦 %.2f） |'
          % (len(ctx['run']), sum(r['eecu'] for r in ctx['run']),
             ctx['s_run_in']['n'], sum(r['eecu'] for r in ctx['run'] if r['land_frac'] == 1.0),
             ctx['s_run_co']['n'], sum(r['eecu'] for r in ctx['run']
                                       if r['land_frac'] is not None and r['land_frac'] < 1.0)))
    A('| 终态瓦（QA_OK/DONE/BLOCKED/FAILED） | %d 瓦，终值合计 %.2f EECU·h |'
      % (len(ctx['fin']), sum(r['eecu'] for r in ctx['fin'])))
    if not ctx['finished']:
        if ctx['s_in']['n']:
            avg = ctx['s_in']['mean']
            left = len(ctx['inland']) - ctx['s_in']['n']
            A('| 外推（进行中，仅供参考） | 终态内陆均值 %.2f × 剩余内陆 %d 瓦 ≈ 预计内陆增量 %.0f EECU·h%s |'
              % (avg, left, left * avg,
                 '（终态样本 n=%d<3，粗略）' % ctx['s_in']['n'] if ctx['s_in']['n'] < 3 else ''))
        else:
            A('| 外推（进行中，仅供参考） | 终态瓦尚不足，暂不外推（预算是硬闸门，驱动自带熔断） |')
    A('| 保真门（3000 点逐点） | %d/%d 报 100.0000%%%s |'
      % (len(ctx['fid_ok']), len(ctx['recs']),
         '' if not ctx['fid_bad'] else '；**<100%% %d 瓦**：%s'
         % (len(ctx['fid_bad']), '，'.join('%s(%.4f)' % (r['tile'], r['fidelity'])
                                          for r in ctx['fid_bad']))))
    A('| QA json | %d/%d 瓦已有（board.qa_done=%d）%s |'
      % (ctx['qa_cnt'], len(ctx['recs']), ctx['qa_done_cnt'],
         '' if not ctx['miss_qa'] else '；**QA_OK 但 json 缺失：%s**' % '，'.join(ctx['miss_qa'])))
    A('| 赛跑 / 熔断 / 摘除 | 多路瓦 %d；日志触发赛跑 %d 行（去重瓦 %d）；熔断 %d；摘除账号 %d |'
      % (len(ctx['multis']), len(ctx['ev']['race']), len(set(x[0] for x in ctx['ev']['race'])),
         len(ctx['ev']['fuse']), len(ctx['state'].get('dead_accts', {}) or {})))
    if ctx['state'].get('fuse'):
        A('| state.fuse | `%s` |' % ctx['state']['fuse'])
    if ctx['state'].get('over_tiles'):
        A('| state.over_tiles (EECU>%.0f) | %s |' % (OVER_LIMIT,
          '，'.join(ctx['state']['over_tiles'])))
    A('')

    # ---- 二、逐省小计 ----
    A('## 二、逐省小计')
    A('')
    A('| 省 | 瓦片 | QA_OK | 在跑 | 待提交 | EECU·h | 终态均值（实花>0） | 范围 |')
    A('|---|---|---|---|---|---|---|---|')
    for p in ctx['prov_tot']:
        A('| %s | %d | %d | %d | %d | %.2f | %s | %s |'
          % (p['full'], p['n'], p['qa_ok'], p['run'], p['wait'], p['eecu'],
             fm(p['s']['mean'], 2), range_str(p['s'])))
    A('| **合计** | **%d** | **%d** | **%d** | **%d** | **%.2f** | %s | — |'
      % (len(ctx['recs']), ctx['n_by'].get('QA_OK', 0), ctx['n_by'].get('在跑', 0),
         ctx['n_by'].get('待提交', 0), ctx['total_eecu'],
         fm((sum(r['eecu'] for r in ctx['fin']) / len(ctx['fin']))
            if any(r['eecu'] > 0 for r in ctx['fin']) else None, 2)))
    A('')

    # ---- 三、逐瓦台账 ----
    A('## 三、逐瓦台账（tile/省/zone/账号/状态/EECU/保真/QA 双口径）')
    A('')
    A('| tile | 批 | 省 | zone | land_frac | acct | state | EECU·h | 保真 | 覆盖px% | 覆盖km2% | 海面被分类% | 类数 | 面积km² | 孤类数 |')
    A('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    shown = ctx['recs'][:SNAPSHOT_MAX_ROWS]
    for r in shown:
        q = r['qa']
        inflight = r['state'] not in ctx['terminal']
        if bucket_state(r['state']) == '待提交' and r['eecu'] == 0:
            eec = '—'
        else:
            eec = '%.2f%s' % (r['eecu'], '†' if inflight and r['eecu'] > 0 else '')
        A('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
          % (r['tile'], r['batch'] or '—', r['prov'], r['zone'], fm(r['land_frac'], 3),
             r['acct'] or '—', r['state'], eec,
             ('%.4f' % r['fidelity']) if r['fidelity'] is not None else '—',
             fm(q.get('coverage_px_pct'), 2), fm(q.get('coverage_km2_pct'), 2),
             fm(q.get('sea_classified_pct'), 2), q.get('n_classes', '—'),
             fm(q.get('total_km2'), 0), (len(q.get('rare_classes', [])) if q else '—')))
    if len(ctx['recs']) > len(shown):
        A('| … | | | | | | | | | | | | | | |')
    A('')
    A('* 单位注记：覆盖px% = 像元口径；覆盖km2% = 面积口径（doc 43 纪律，混单位曾把沿海 65% 虚报成 99.99%）；')
    A('  海面被分类%% = coverage_px_pct − 100×land_frac（政策默认保留待用户定夺）；孤类 = <%.2f%% 的类数。' % QA_EDGE)
    A('* EECU·h 为 board 全路累计（含赛跑/失败路）；「—」= 未提交（无成本）；**† = 在跑中的累计读数，非终值**')
    A('  （GEE 边跑边报，收官时以终值替代）。')
    A('')

    # ---- 四、赛跑与异常 ----
    A('## 四、赛跑与异常（state.runs 多路 + 日志事件）')
    A('')
    li = ctx['led_stats']
    A('* 台账事件行：prod %d（状态分布 %s；note 分布 %s），upload %d，异常 schema %d；表头行 %d（%d 列）'
      % (li['n_prod'], '，'.join('%s=%d' % (k, v) for k, v in sorted(li['states'].items())),
         '，'.join('%s=%d' % (k, v) for k, v in sorted(li['notes'].items())),
         li['n_upload'], li['n_raw'], li.get('n_head', 0), li.get('head_schema', 0)))
    if ctx['multis']:
        A('* **多路瓦片（state.json runs>1）**：')
        for r in ctx['multis']:
            for suf in sorted(r['runs']):
                x = r['runs'][suf]
                A('  * %s 路%s @%s %s EECU=%.3f（%s）' % (r['tile'], suf, x.get('acct', '—'),
                                                          x.get('state', '—'), fnum(x.get('eecu')), x.get('desc', '')))
    else:
        A('* 多路瓦片：无（state.json 各瓦 runs 均 ≤1）')
    race_tiles = sorted(set(x[0] for x in ctx['ev']['race']))
    if ctx['ev']['race']:
        A('* **赛跑触发**：日志 %d 行 / 去重 %d 瓦（%s%s）；台账 RACE 行 %d。首次 5 行：'
          % (len(ctx['ev']['race']), len(race_tiles), '，'.join(race_tiles[:12]),
             '…' if len(race_tiles) > 12 else '', li['states'].get('RACE', 0)))
        for g in ctx['ev']['race'][:5]:
            A('  * %s：%s' % (g[0], g[1]))
        A('  * 口径提示：赛跑"次数"按原始事件行计；同瓦可逐轮重复写（第一次五省批日志 467 行 / 去重 19 瓦 /')
        A('    其发布报告写 73 次——三口径不等于数据错误，引用时须注明按哪一种。')
    else:
        A('* 赛跑触发：无')
    tfail = sorted(set(x[0] for x in ctx['ev']['tstate'] if x[3] == 'FAILED'))
    n_allfail = sum(v for k, v in li['notes'].items() if k.startswith('no winner after'))
    A('* **FAILED**：board=FAILED %d 瓦；日志状态→FAILED %d 瓦（%s）；台账 ALL_FAILED 行 %d'
      % (sum(1 for r in ctx['recs'] if r['state'] == 'FAILED'), len(tfail),
         '，'.join(tfail[:12]) or '—', n_allfail))
    if ctx['ev']['fuse']:
        A('* **熔断（⛔）** %d 行：' % len(ctx['ev']['fuse']))
        for g in ctx['ev']['fuse'][:10]:
            A('  * %s' % g[0])
    else:
        A('* 熔断：无')
    dead = ctx['state'].get('dead_accts', {}) or {}
    A('* **账号摘除**：state.dead_accts %d 个%s；日志摘除事件 %d 行'
      % (len(dead), ('（%s）' % '，'.join('%s:%s' % (k, v) for k, v in dead.items())) if dead else '',
         len(ctx['ev']['dead'])))
    for g in ctx['ev']['dead'][:10]:
        A('  * %s 连续异常 %s 次：%s' % (g[0], g[1], g[2]))
    if ctx['ev']['over']:
        over_u = []
        for g in ctx['ev']['over']:                       # 去重：(瓦, 阈值) 组合
            key = '%s=%.2f(>%s)' % (g[0], fnum(g[1]), g[2])
            if key not in over_u:
                over_u.append(key)
        A('* 超带告警（日志内建带，非本脚本判据）：%s' % '，'.join(over_u))
    if ctx['ev']['fid_re'] or ctx['ev']['dl_fail']:
        A('* 保真复跑 %d 次%s；下载失败 %d 次%s'
          % (len(ctx['ev']['fid_re']),
             ('（%s）' % '，'.join(set(g[0] for g in ctx['ev']['fid_re']))) if ctx['ev']['fid_re'] else '',
             len(ctx['ev']['dl_fail']),
             ('（%s）' % '，'.join(set(g[0] for g in ctx['ev']['dl_fail']))) if ctx['ev']['dl_fail'] else ''))
    for g in ctx['ev']['finish']:
        A('* 日志收官行：why=%s，EECU=%s/%s，完成 %s/%s' % (g[0], g[1], g[2], g[3], g[4]))
    if not ctx['ev']['ok']:
        A('* 日志不可读（%s 缺失）：本节仅依据 state.json 与台账。' % s['log'])
    A('')

    # ---- 五、EECU 成本校准 ----
    A('## 五、EECU 成本校准（对照第一次五省批）')
    A('')
    A('| 口径 | n（终态、实花>0） | 均值 | 范围 | 第一次（同口径复算） |')
    A('|---|---|---|---|---|')
    r_in = ('n=%d 均值 %s' % (ref['s_in']['n'], fm(ref['s_in']['mean'], 1))) if ref else '—'
    r_co = ('n=%d 均值 %s' % (ref['s_co']['n'], fm(ref['s_co']['mean'], 1))) if ref else '—'
    A('| 内陆（land_frac=1） | %d | %s | %s | %s |'
      % (ctx['s_in']['n'], fm(ctx['s_in']['mean'], 1), range_str(ctx['s_in']), r_in))
    A('| 沿海（land_frac<1） | %d | %s | %s | %s |'
      % (ctx['s_co']['n'], fm(ctx['s_co']['mean'], 1), range_str(ctx['s_co']), r_co))
    if ctx['run']:
        A('| 在跑瓦累计（非终值，**不进均值**） | %d | %s | %s | — |'
          % (ctx['s_run']['n'], fm(ctx['s_run']['mean'], 2), range_str(ctx['s_run'])))
    r = pearson([x['land_frac'] for x in ctx['coast_sorted'] if x['eecu'] > 0],
                [x['eecu'] for x in ctx['coast_sorted'] if x['eecu'] > 0])
    A('')
    A('* 成本带只算**终态瓦终值**——在跑瓦的 eecu 是进行中累计（见逐瓦表 † 标记），与第一次收官终值不可比；')
    A('  终态瓦不足时均值会缺号，属预期而非缺陷。')
    A('* 沿海 land_frac↔EECU 相关系数（终态实花>0）：%s%s'
      % ('%.3f' % r if r is not None else '—（终态样本不足，下表改列全部沿海瓦）',
         '；成本随 land_frac 单调走低 = 海面无 AEF 覆盖像元不参与分类，属结构性低价'
         if r is not None and ctx['coast_sorted'] else ''))
    if ctx['coast_sorted']:
        all_fin = all(x['state'] in ctx['terminal'] for x in ctx['coast_sorted'])
        A('* 沿海逐瓦（按 land_frac 升序%s）：' % ('，全部终态' if all_fin else '，含在跑累计'))
        for x in ctx['coast_sorted']:
            if bucket_state(x['state']) == '待提交':
                tag = '未提交'
            elif x['state'] in ctx['terminal']:
                tag = '终值'
            else:
                tag = '在跑累计'
            A('  * %s land_frac=%.3f EECU=%s（%s）' % (x['tile'], x['land_frac'], fm(x['eecu'], 2), tag))
    A('* 判据带：内陆 ≤%.0f EECU·h（第一次 2026-10-04 G4 裁决校准带）' % OVER_LIMIT)
    if ctx['over']:
        A('  * **本批终态超 %.0f 的内陆瓦片（%d）**：%s'
          % (OVER_LIMIT, len(ctx['over']),
             '，'.join('%s=%.2f' % (x['tile'], x['eecu']) for x in
                       sorted(ctx['over'], key=lambda v: -v['eecu']))))
    else:
        A('  * 本批终态内陆瓦片**无一超过 %.0f**（最大 %s）'
          % (OVER_LIMIT, fm(max([x['eecu'] for x in ctx['fin']
                                 if x['land_frac'] == 1.0], default=None), 2)))
    if ctx['over_run']:
        A('  * 在跑中已越过 %.0f 的内陆瓦（**待终值确认**）：%s'
          % (OVER_LIMIT, '，'.join('%s=%.2f†' % (x['tile'], x['eecu']) for x in ctx['over_run'])))
    elif ctx['run']:
        A('  * 在跑内陆瓦当前累计最大 %s（未越带）'
          % fm(max([x['eecu'] for x in ctx['run'] if x['land_frac'] == 1.0], default=None), 2))
    if ref:
        A('* 第一次对照来源：`%s`（只读复算，非文档抄录）——总 %.1f EECU·h、'
          '内陆 n=%d 均值 %s、沿海 n=%d 均值 %s、超带瓦 %s'
          % (ref['src']['board'], ref['total'], ref['s_in']['n'], fm(ref['s_in']['mean'], 1),
             ref['s_co']['n'], fm(ref['s_co']['mean'], 1),
             '，'.join(ref['over']) if ref['over'] else '无'))
        if ref['s_in']['mean'] and ctx['s_in']['mean']:
            A('* 偏差：本批内陆均值 − 第一次 = %+.2f EECU·h（同口径实花>0 均值差）%s'
              % (ctx['s_in']['mean'] - ref['s_in']['mean'],
                 '（**本批终态 n=%d<5，样本不足，此偏差仅供跟踪、不构成结论**）'
                 % ctx['s_in']['n'] if ctx['s_in']['n'] < 5 else ''))
    if ctx['smoke']:
        A('* （SMOKE：对照数据即本数据，偏差应为 0.00，用于自检对照通路。）')
    A('')

    # ---- 六、口径核对 ----
    A('## 六、口径核对与数据坑（自动检查）')
    A('')
    ok = lambda b: 'PASS' if b else '⚠'
    A('* %s plan vs board 瓦数：plan %d / board %d%s'
      % (ok(not ctx['plan_only'] and not ctx['board_only']), len(ctx['plan']), len(ctx['recs']),
         '' if not ctx['plan_only'] and not ctx['board_only'] else
         '；plan 独有 %s / board 独有 %s' % (ctx['plan_only'] or '无', ctx['board_only'] or '无')))
    got = {p['prov']: p['n'] for p in ctx['prov_tot']}
    exp = ctx['exp_prov']
    if exp:
        diff = {k: (exp[k], got.get(k, 0)) for k in exp if got.get(k, 0) != exp[k]}
        A('* %s 逐省计数 vs config.note 期望：%s%s'
          % (ok(not diff), '，'.join('%s=%d' % (k, v) for k, v in got.items()),
             '' if not diff else '；**不符 %s**（期望值,实得）' % diff))
    else:
        A('* — 逐省计数：%s（config.note 无期望值可对照）'
          % '，'.join('%s=%d' % (k, v) for k, v in got.items()))
    A('* %s board 状态值合法性：%s' % (ok(not ctx['n_by'].get('其他')),
      '全部属已知集合' if not ctx['n_by'].get('其他')
      else '**未知状态 %s**' % sorted(set(r['state'] for r in ctx['recs'] if r['state'] not in BOARD_KNOWN))))
    A('* %s QA json 与 board.qa_done 一致性：json %d / qa_done=1 %d'
      % (ok(ctx['qa_cnt'] == ctx['qa_done_cnt']), ctx['qa_cnt'], ctx['qa_done_cnt']))
    missf = []
    for r in ctx['recs']:
        if not r['qa']:
            continue
        for k in ('coverage_px_pct', 'coverage_km2_pct', 'sea_classified_pct', 'n_classes', 'total_km2', 'land_frac'):
            if r['qa'].get(k) is None:
                missf.append('%s.%s' % (r['tile'], k))
    A('* %s QA json 字段完整性：%s' % (ok(not missf), '核心字段齐全' if not missf else '缺 %s' % '，'.join(missf[:8])))
    tiles_in_ledger = set(ctx['led_stats']['tiles'])
    board_t = set(r['tile'] for r in ctx['recs'])
    A('* %s 台账瓦片 ⊆ board：台账 %d 瓦%s' % (ok(tiles_in_ledger <= board_t), len(tiles_in_ledger),
      '' if tiles_in_ledger <= board_t else '；越界 %s' % sorted(tiles_in_ledger - board_t)[:8]))
    A('* %s 双 schema 分支：5 列 upload %d / 8 列 prod %d / 其他 %d%s'
      % (ok(ctx['led_stats']['n_raw'] == 0), ctx['led_stats']['n_upload'], ctx['led_stats']['n_prod'],
         ctx['led_stats']['n_raw'],
         '' if ctx['led_stats']['n_raw'] == 0 else '；样例 %s' % ctx['led_anom'][:1]))
    A('* %s 台账表头：表头行 %d（%d 列；列名只描述其所在 schema 段，其余段无重复表头）'
      % (ok(ctx['led_stats'].get('n_head', 0) <= 1), ctx['led_stats'].get('n_head', 0),
         ctx['led_stats'].get('head_schema', 0)))
    A('* %s state.json 覆盖：tiles %d 键%s' % (ok(True), len(ctx['state'].get('tiles', {}) or {}),
      '；不含 winner/total_eecu/done_n（进行中正常，收官后才有）'
      if not ctx['finished'] else ''))
    A('* 已知坑（实测，非本节自动结论）：')
    A('  1. `eecu_ledger_2nd.csv` **表头只描述上传段**（首行 `time,acct,task_id,desc,eecu_h` 5 列，c1 写的），')
    A('     生产段是 prod2_driver.ledger() 追加的 8 列且**不再补表头**——用 DictReader 按 8 列名读会把')
    A('     task_id 当 tile、desc 当 state；只能按列数分支（本脚本做法）。')
    A('  2. board 无 prov/land_frac 列，省/zone 必须从 plan/tiles_secondrun.csv（或 QA json）取；')
    A('     zone 判定 = land_frac<1（与驱动 load_tiles() 同式）。')
    A('  3. 在跑瓦 board.eecu 是**进行中累计读数**（†），不是终值——成本带/超带判定必须只用终态瓦，')
    A('     否则均值被系统性压低（本脚本第五节口径）。')
    A('  4. 赛跑计数三口径（原始事件行/去重瓦/发布报告次）不相等，引用须注明口径（见第四节）。')
    A('  5. 5p 参照：`eecu_ledger_batch.csv` 为单一 8 列 schema（无上传段表头）、`batch_board.csv` 与 2nd 同构，')
    A('     故 --smoke 可直接复用同一条代码路径；5p 的 plan/tiles_frozen.csv 比其 board 多 T2713/T2102 两试产瓦。')
    A('')

    # ---- 七、产物 ----
    A('## 七、产物与复现')
    A('')
    A('* 本摘要：`%s`（脚本幂等，可反复重跑覆盖）' % s['out_md'])
    A('* 规范化台账（9 列）：`%s`（%d 行 = upload %d + prod %d）'
      % (s['out_csv'], ctx['led_stats']['n_total'], ctx['led_stats']['n_upload'], ctx['led_stats']['n_prod']))
    A('* 输入（只读）：`%s`、`%s`、`%s`、`%s`、`%s`'
      % (s['board'], s['ledger'], s['state'], s['qa'] + r'\{tile}_qa.json', s['plan']))
    A('* 命令：`python c8_batch_summary.py%s`' % (' --smoke' if ctx['smoke'] else ''))
    A('* 本脚本不调用 GEE、不动任何输入文件；唯一写点 = <reports>（含 --smoke 的 _smoke/ 子目录）。')
    A('')
    return '\n'.join(L) + '\n'


def normalize_ledger_rows(raw_rows):
    """纯函数版：行列表 → 9 列规范化行（供 --smoke 自检样例复用，不落盘）。"""
    out, _, _ = _norm_core(raw_rows)
    return out


NORM_HEADER = ['time', 'phase', 'acct', 'tile', 'task_id', 'desc', 'state', 'eecu_h', 'note']


def write_outputs(ctx, out_dir):
    """写 md + 规范化 csv；返回两个路径。"""
    os.makedirs(out_dir, exist_ok=True)
    src = ctx['src']
    src['out_md'] = os.path.join(out_dir, 'batch_summary_2nd.md')
    src['out_csv'] = os.path.join(out_dir, 'eecu_ledger_2nd_normalized.csv')
    ref = ref_stats(REF_SRC) if not ctx['smoke'] else ref_stats(src)   # 冒烟时对照即自身（自检通路）
    md = render(ctx, ref)
    with open(src['out_md'], 'w', encoding='utf-8', newline='\n') as f:
        f.write(md)
    with open(src['out_csv'], 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(NORM_HEADER)
        w.writerows(ctx['ledger_rows'])
    return src['out_md'], src['out_csv']


# ---------------- 源配置 ----------------
def build_src(smoke):
    """返回输入/输出路径字典。smoke=True → 第一次五省批（只读）+ _smoke 输出。"""
    if smoke:
        return dict(smoke=True,
                    title='第一次五省批（川/黑/粤/宁/闽 74 瓦，2023）',
                    root=ROOT1,
                    plan=os.path.join(ROOT1, 'plan', 'tiles_frozen.csv'),
                    board=os.path.join(ROOT1, 'ledger', 'batch_board.csv'),
                    ledger=os.path.join(ROOT1, 'ledger', 'eecu_ledger_batch.csv'),
                    state=os.path.join(ROOT1, 'pilot', 'batch', 'state.json'),
                    qa=os.path.join(ROOT1, 'qa'),
                    config=None,
                    budget=1600.0,   # 第一次五省批项目上限（其 batch_summary.md §二.4 裁决），非第二次的 950
                    log=LOG1,
                    driver='pilot/b1_batch.py（第一次五省批驱动，同构）',
                    pool_note='smp2023_merged（2,060,577 点原样）',
                    script='代码/9.第二次分类川粤宁/c8_batch_summary.py --smoke')
    return dict(smoke=False,
                title='第二次分类（川/粤/宁 40 瓦，2023）',
                root=ROOT2,
                plan=os.path.join(ROOT2, 'plan', 'tiles_secondrun.csv'),
                board=os.path.join(ROOT2, 'ledger', 'board_2nd.csv'),
                ledger=os.path.join(ROOT2, 'ledger', 'eecu_ledger_2nd.csv'),
                state=os.path.join(ROOT2, 'run', 'batch', 'state.json'),
                qa=os.path.join(ROOT2, 'qa'),
                config=os.path.join(ROOT2, 'config.json'),
                log=LOG2,
                driver='run/prod2_driver.py',
                pool_note='train2023_v2（1,764,020 点 / 23 类，remap=false）',
                script='代码/9.第二次分类川粤宁/c8_batch_summary.py')


REF_SRC = dict(plan=os.path.join(ROOT1, 'plan', 'tiles_frozen.csv'),
               board=os.path.join(ROOT1, 'ledger', 'batch_board.csv'))


def main(argv=None):
    ap = argparse.ArgumentParser(description='第二次分类（川/粤/宁 40 瓦，2023）生产收官摘要生成器')
    ap.add_argument('--reports-dir', default=DEFAULT_REPORTS,
                    help='输出根目录（默认 %s）；仅此目录可写' % DEFAULT_REPORTS)
    ap.add_argument('--smoke', action='store_true',
                    help='冒烟自检：改用第一次五省批台账（只读）在 <reports>/_smoke/ 出同版式样例')
    ap.add_argument('--log', default=None, help='覆盖驱动日志路径（默认 %s / %s）' % (LOG2, LOG1))
    ap.add_argument('--stdout', action='store_true', help='额外把摘要打印到 stdout')
    a = ap.parse_args(argv)

    rep = os.path.abspath(a.reports_dir)
    out_dir = os.path.join(rep, '_smoke') if a.smoke else rep
    assert os.path.abspath(out_dir).startswith(rep), '输出必须落在 reports 目录内'

    src = build_src(a.smoke)
    if a.log:
        src['log'] = a.log
    for k in ('plan', 'board', 'ledger', 'state'):
        if not os.path.exists(src[k]):
            print('[warn] 输入缺失：%s（%s）' % (src[k], k))
    ctx = analyze(src)
    md_fp, csv_fp = write_outputs(ctx, out_dir)
    ctx['src']['out_md'], ctx['src']['out_csv'] = md_fp, csv_fp

    print('—— c8_batch_summary%s ——' % ('（SMOKE）' if a.smoke else ''))
    print('数据时点：%s ｜ 状态：%s' % (tstr(ctx['data_ts']), '收官' if ctx['finished'] else '进行中'))
    print('瓦片 %d ｜ %s ｜ 总 EECU %.2f/%.0f'
          % (len(ctx['recs']),
             '，'.join('%s=%d' % (k, v) for k, v in sorted(ctx['n_by'].items())),
             ctx['total_eecu'], ctx['budget']))
    print('终态成本（实花>0）：内陆 n=%d 均值=%s ｜ 沿海 n=%d 均值=%s ｜ 超带(>%.0f) %d 瓦'
          % (ctx['s_in']['n'], fm(ctx['s_in']['mean'], 2), ctx['s_co']['n'],
             fm(ctx['s_co']['mean'], 2), OVER_LIMIT, len(ctx['over'])))
    if ctx['run']:
        print('在跑累计（非终值）：%d 瓦 %.2f EECU·h' % (len(ctx['run']),
                                                     sum(r['eecu'] for r in ctx['run'])))
    print('产出：%s' % md_fp)
    print('产出：%s（%d 行）' % (csv_fp, ctx['led_stats']['n_total']))
    if a.stdout:
        print('')
        print(render(ctx, ref_stats(REF_SRC) if not a.smoke else ref_stats(src)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
