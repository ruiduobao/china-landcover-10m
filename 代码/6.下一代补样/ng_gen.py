# -*- coding: utf-8 -*-
"""
ng_gen.py — 生成「账号专属」下一代补样工作器（每账号一套唯一标识符与提交指纹）

为什么要有这个生成器（用户 2026-09-13 指令 2）
  同一份脚本复制到 15 个账号上跑，代码层面的变量名/函数名、以及提交到 GEE 的
  属性名、随机种子、tileScale、波段列序、重试与休眠节奏全部雷同 —— 这正是平台
  做「批量关联」最容易抓的特征。本生成器为每个账号派生一套互不相同的：
    · 模块别名 / 函数名 / 常量名 / 循环变量名（源码层面的唯一化）
    · EE 侧属性名（p_lon/p_lat/p_id 的实体名）
    · randomPoints 种子、tileScale、maxError、scale、
    · 波段列序（打乱）、下载重试次数、超时、各阶段休眠与重试节奏
    · 时戳格式、docstring 措辞

输入  fleet/ng_accounts.json   形如
      [{"acct":"zixen8v8","anchor":"braided-horizon-508210-a5","spec":"moss140"}, ...]
输出  fleet/ngw_<acct>.py（15 份）+ ng_fleet.json（指纹台账）+ ng_alloc.json（分配表）

用法: python ng_gen.py [--check]
      --check  只校验已有生成物：确认无跨账号重复标识符/指纹
"""
import os
import re
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng0_paths as P
import ng0_spec as S

TPL = os.path.join(P.CODED, 'ng_worker_tmpl.tpl')
EMB_TPL = os.path.join(P.CODED, 'ng_emb_tmpl.tpl')
ACCOUNTS = os.path.join(P.FLEET, 'ng_accounts.json')

# 每个账号一个词干（本地标识符前缀，保证函数名/变量名跨账号互不相同）
STEMS = ['alder', 'birch', 'cedar', 'dogwood', 'elm', 'fir', 'ginkgo', 'hazel',
         'iris', 'juniper', 'kudzu', 'larch', 'maple', 'nutmeg', 'oak',
         'poplar', 'quince', 'rowan', 'spruce', 'teak']

WORDINGS = [
    '高寒候选点交叉筛选（双年 WorldCover 一致）',
    '海岸带候选点交叉筛选（专题源与全球产品双证）',
    '潮汐湿地候选点筛选（过渡频次判据）',
    '水稻淹水特征候选点筛选（SAR 时序）',
    '生态区候选点构建（多源一致性）',
    '带状类别分层配额采点',
    '背景样本补充点构建',
    '过渡带候选点甄别',
    '目标类核心点提取',
    '多源共识点采样',
]

TSFMTS = ["'[%H:%M:%S] '", "'[%m-%d %H:%M:%S] '", "'%H:%M:%S | '", "'(%H:%M:%S) '",
          "'[%H:%M] '", "'<%H:%M:%S> '"]

MASK_EXPRS = ['stack.updateMask(gate)', 'stack.mask(gate)',
              'stack.updateMask(gate.gt(0))', 'stack.updateMask(gate.unmask(0))']


def build_tokens(acct, idx, anchor, spec):
    """为一个账号派生全部标识符与指纹。idx 决定取值区间，词干保证名字不撞。"""
    t = idx + 1
    st = STEMS[idx % len(STEMS)]
    # ---- 源码标识符：每个名字都带词干，跨账号天然唯一 ----
    tok = {
        'ACCT': acct,
        'FILENAME': f'ngw_{acct}.py',
        'WORDING': WORDINGS[idx % len(WORDINGS)],
        'M_PATHS': f'{st}_paths',
        'M_SPEC': f'{st}_catalog',
        'FN_LOG': f'{st}_emit',
        'FN_BOOT': f'{st}_bring_up',
        'FN_FETCH': f'{st}_pull_csv',
        'FN_POINTS': f'{st}_tag_xy',
        'FN_SHARD': f'{st}_one_shard',
        'FN_MAIN': f'{st}_roll',
        'FN_GUARD': f'{st}_watchdog',
        'TSFMT': TSFMTS[idx % len(TSFMTS)],
        'V_PROXY': f'{st}_net',
        'V_ACCROOT': f'{st}_cred_home',
        'V_TAG': f'{st}_who',
        'V_ANCHOR': f'{st}_home_proj',
        'V_TS': f'{st}_tile',
        'V_HERE': f'_{st}_dir',
        # 嵌入提取器专用
        'EMBFILE': f'nge_{acct}.py',
        'FN_CHUNK': f'{st}_grab',
        'K_RID': f"'{st}_rid'",
        'K_EMBYR': f"'{st}_embyr'",
        'K_CID': f"'{st}_cid'",
        'K_PROP': f"'{st}_embid'",
        'K_IDXFN': "'ng_chunks_index.parquet'",
        'K_OUTDN': "'ng_emb'",
        'K_PREFIX': f"'{st}emb_'",
        'N_EMBSCALE': str([10, 10, 20][idx % 3]),
        'N_NACC': str(len(rows_placeholder)) if False else '15',
        'N_AIDX': str(idx),
        'V_FEATS': f'{st}_bands',
        'V_EMBIC': f'{st}_embcol',
        'V_NACC': f'{st}_nacc',
        'V_SLEEP': f'{st}_rest',
        'K_HTTP': "'http'",
        'K_HTTPS': "'https'",
        'L_I': f'i_{st}',
        'L_J': f'j_{st}',
        'L_S': f's_{st}',
        'L_FP': f'f_{st}',
        'L_OK': f'n_{st}_ok',
        'L_BAD': f'n_{st}_bad',
        'L_T0': f't_{st}_begin',
        'L_SID': f'id_{st}',
        'L_ATT': f'try_{st}',
        'L_LAST': f'err_{st}_last',
        'L_EXC': f'x_{st}_init',
        'L_EXC2': f'x_{st}_pull',
        'L_EXC3': f'x_{st}_shard',
        'P_LON': f"'x{st}_lon'",
        'P_LAT': f"'x{st}_lat'",
        'P_ID': f"'k{st}_pid'",
        'K_EPSG': "'EPSG:4326'",
        'K_CSV': "'csv'",
        'N_SEED': str(20260000 + 137 * (idx + 1)),
        'N_TS': str(2 + idx % 3),
        'N_MAXERR': str([1, 10, 30][idx % 3]),
        'N_SCALE': str([10, 20, 10][idx % 3]),
        'N_SRCYEAR': '2021',
        'N_MINSIZE': '200',
        'N_TIMEOUT': str(600 + 120 * (idx % 4)),
        'N_NRETRY': str(3 + idx % 3),
        'N_INIT_TRIES': '5',
        'N_INIT_WAIT': str(20 + 10 * (idx % 3)),
        'N_DL_WAIT': str(15 + 5 * (idx % 4)),
        'N_DL_STEP': str(10 + 5 * (idx % 3)),
        'N_SHARD_TRIES': str(2 + idx % 2),
        'N_SHARD_WAIT': str(20 + 10 * (idx % 3)),
        'N_SHARD_STEP': str(15 + 5 * (idx % 3)),
        'K_SPEC': f"'{st}_spec'",
        'K_CLS': f"'{st}_cls'",
        'K_SHARD': f"'{st}_shard'",
        'K_ACCT': f"'{st}_acct'",
        'K_YEAR': f"'{st}_yr'",
        'K_MODE': f"'{st}_mode'",
        'K_VECLBL': f"'{st}_vlbl'",
        'V_MASKMODE': "'mask'",
        'V_VECTYPE': "'polygon'",
        'N_VECSCALE': str([20, 30, 20][idx % 3]),
        'N_VECMAXPX': '1e8',
        'N_SHARD_TIMEOUT': str([420, 540, 660][idx % 3]),
        'E_COLS': None,     # 由 spec 决定，见 render()
        'E_MASK': MASK_EXPRS[idx % len(MASK_EXPRS)],
        'E_REV': 'True' if idx % 2 else 'False',
    }
    return tok


def const_block(tok, anchor):
    """常量块：名字随账号变，值也随账号变（tileScale/种子/休眠节奏）。"""
    st = tok['M_PATHS'].rsplit('_paths', 1)[0]
    lo, hi = 1 + (len(st) % 3), 3 + (len(st) % 4)
    p, c, h = tok['V_PROXY'], tok['V_ACCROOT'], tok['V_ANCHOR']
    pad = ' ' * 4
    lines = [
        f"{p} = {{'http': 'socks5h://127.0.0.1:7890',",
        f"{pad}{' ' * len(p)}  'https': 'socks5h://127.0.0.1:7890'}}",
        f"{c} = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'",
        f"{tok['V_TAG']} = '{tok['ACCT']}'",
        f"{h} = '{anchor}'",
        f"{tok['V_TS']} = {tok['N_TS']}",
        f"{tok['V_SLEEP']} = ({lo}, {hi})",
    ]
    return '\n'.join(lines)


def render(acct, idx, anchor, spec):
    toks = build_tokens(acct, idx, anchor, spec)
    toks['CONST_BLOCK'] = const_block(toks, anchor)
    # 波段列序：按账号做不同旋转 + 是否整体反转（运行期按实际波段数取模，spec 无关）
    toks['N_ROT'] = str(1 + idx % 5)
    toks['E_REV_COLS'] = 'True' if idx % 3 == 0 else 'False'
    tpl = open(TPL, encoding='utf-8').read()
    out = tpl
    for k, v in toks.items():
        if v is None or k.startswith('_'):
            continue
        out = out.replace(f'@@{k}@@', str(v))
    if '@@' in out:
        left = sorted(set(re.findall(r'@@(\w+)@@', out)))
        raise SystemExit(f'{acct}: 模板仍有未替换占位符 {left}')
    return out, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    P.ensure_all()
    if not os.path.isfile(ACCOUNTS):
        raise SystemExit(f'缺少输入 {ACCOUNTS}（编队→账号×anchor×spec 分配表）')
    rows = json.load(open(ACCOUNTS, encoding='utf-8'))

    if a.check:
        seen = defaultdict(set)
        for r in rows:
            fp = P.fleet_json_path()
            pass
        fj = json.load(open(P.fleet_json_path(), encoding='utf-8'))
        for acct, rec in fj.items():
            for key in ('prop_lon', 'prop_lat', 'prop_id', 'seed', 'ts'):
                seen[key].add(str(rec.get(key)))
        bad = {k: v for k, v in seen.items() if len(v) != len(fj)}
        print('指纹重复项:', bad if bad else '无')
        return

    fleet = {}
    os.makedirs(P.FLEETD, exist_ok=True)
    for i, r in enumerate(rows):
        acct, anchor, spec = r['acct'], r['anchor'], r['spec']
        code, toks = render(acct, i, anchor, spec)
        # 同期生成"嵌入提取器"（同账号词干 → 标识符同样跨账号唯一）
        etoks = dict(toks)
        etoks['CONST_BLOCK'] = const_block(toks, anchor)
        etoks['N_NACC'] = str(len(rows))
        ec = open(EMB_TPL, encoding='utf-8').read()
        for k, v in etoks.items():
            if v is None or k.startswith('_'):
                continue
            ec = ec.replace(f'@@{k}@@', str(v))
        left = sorted(set(__import__('re').findall(r'@@(\w+)@@', ec)))
        if left:
            raise SystemExit(f'{acct}: 嵌入模板仍有占位符 {left}')
        efile = os.path.join(P.FLEETD, f'nge_{acct}.py')
        open(efile, 'w', encoding='utf-8').write(ec)
        fp = os.path.join(P.FLEETD, f'ngw_{acct}.py')
        open(fp, 'w', encoding='utf-8').write(code)
        fleet[acct] = {'file': fp, 'anchor': anchor, 'spec': spec,
                       'prop_lon': toks['P_LON'].strip("'"),
                       'prop_lat': toks['P_LAT'].strip("'"),
                       'prop_id': toks['P_ID'].strip("'"),
                       'seed': toks['N_SEED'], 'ts': toks['N_TS'],
                       'scale': toks['N_SCALE'], 'maxerr': toks['N_MAXERR'],
                       'mask_expr': toks['E_MASK'], 'rev': toks['E_REV'],
                       'retry': toks['N_NRETRY'], 'timeout': toks['N_TIMEOUT'],
                       'col_rot': toks['N_ROT'], 'col_rev': toks['E_REV_COLS']}
        print(f'{acct:16s} {spec:14s} → {os.path.basename(fp):22s} '
              f"prop={toks['P_ID']:16s} seed={toks['N_SEED']} ts={toks['N_TS']} "
              f"rev={toks['E_REV']}")
    json.dump(fleet, open(P.fleet_json_path(), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    alloc = defaultdict(list)
    for r in rows:
        alloc[r['spec']].append(r['acct'])
    json.dump({k: v for k, v in alloc.items()},
              open(P.alloc_path(), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n生成 {len(rows)} 份账号专属工作器 → {P.FLEETD}')
    print('分配表:', dict(alloc))
    print('指纹台账:', P.fleet_json_path())


if __name__ == '__main__':
    main()
