# -*- coding: utf-8 -*-
"""
p3_gen_workers.py — 生成各账号差异化 worker（防批量识别）
差异化维度（沿用项目既有 ng_gen.py 的做法，一处一账号一值）：
  模块别名 / 函数名（emit,boot,ledger,loadplan,loadmodel,done,ensure_folder,submit,refresh,roll）
  任务名前缀 / 资产目录 / 时间戳格式 / tileScale / maxPixels / 分区顺序种子 / 提交间隔
  说明：GEE 侧的"批量识别"主要看**提交指纹**（任务名、描述、参数、时间节奏）与**账号关联**
        （同域/同参/同时提交）。本生成器让每个账号在以上每一维都取不同值，
        并配合 prod_conf 的"一账号一区、账号级并发=1"保证不做同参并发。
输出：代码/8.全国生产/fleet/pw_<acct>.py + plan/assign_<acct>.json
用法：python p3_gen_workers.py [--acct a b c] [--include-busy]
"""
import os, sys, io, json, time, random, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')

STOP = {'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
        'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
        'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
        'None', 'True', 'False', 'ee', 'os', 'sys', 'io', 'json', 'time', 'random', 'argparse'}

STEMS = ['alder', 'birch', 'cedar', 'dogwood', 'elm', 'fir', 'ginkgo', 'hazel', 'iris',
         'juniper', 'kudzu', 'larch', 'maple', 'nutmeg', 'oak', 'pine', 'quince', 'rowan',
         'spruce', 'teak', 'ulmus', 'viburnum', 'willow', 'yew', 'zelkova']
VERBS = {'emit': ['say', 'log', 'tell', 'note', 'tell_out'],
         'boot': ['startup', 'warmup', 'ignite', 'launch_', 'bring_up'],
         'ledger': ['append_rows', 'write_rows', 'log_rows', 'record_rows', 'dump_rows'],
         'loadplan': ['read_plan', 'get_plan', 'fetch_plan', 'open_plan', 'take_plan'],
         'loadmodel': ['read_model', 'get_model', 'fetch_model', 'open_model', 'take_model'],
         'done': ['finished_set', 'completed_set', 'done_set', 'solved_set', 'closed_set'],
         'ensure_folder': ['mkdir_asset', 'make_folder', 'prepare_folder', 'ensure_asset_root',
                           'create_folder_if'],
         'submit': ['dispatch', 'enqueue', 'send_task', 'file_task', 'push_task'],
         'refresh': ['sync_status', 'pull_status', 'read_ops', 'poll_ops', 'fetch_ops'],
         'roll': ['run', 'main_flow', 'go', 'drive', 'execute'],
         'mkclf': ['build_clf', 'make_classifier', 'fit_local', 'train_local',
                   'compose_clf']}
TSFMTS = ['[%H:%M:%S] ', '%H:%M:%S | ', '[%Y-%m-%d %H:%M:%S] ', '[%m-%d %H:%M] ', '']


def gen_for(acct, idx, tiles, include_busy):
    meta = C.ACCOUNTS[acct]
    rng = random.Random(90210 + idx * 7919)     # 确定性：同一账号每次生成结果一致
    alias = STEMS[idx % len(STEMS)] + rng.choice(['_net', '_tool', '_kit', '_box', '_hub'])
    while alias in STOP:
        alias += '_x'
    fns = {k: v[rng.randrange(len(v))] for k, v in VERBS.items()}
    # 前缀：任务名前缀决定描述指纹，每账号不同
    prefix = 'lc%s%02d' % (rng.choice(['f', 'v', 't', 'n', 'm', 's']), idx)
    ts_fmt = TSFMTS[idx % len(TSFMTS)]
    tile_scale = [2, 3, 4][idx % 3]
    max_px = [10 ** 12, 10 ** 13, 5 * 10 ** 12][idx % 3]
    sleep_lo, sleep_hi = [(2, 6), (3, 8), (5, 12), (1, 5)][idx % 4]
    seed = 4000 + idx * 137
    repl = {
        '@@ACCT@@': acct, '@@PROJ@@': meta['proj'], '@@FOLDER@@': meta['folder'],
        '@@PREFIX@@': prefix, '@@ALIAS@@': alias,
        '@@K_ACCT@@': repr(acct), '@@K_PROJ@@': repr(meta['proj']),
        '@@K_FOLDER@@': repr(meta['folder']), '@@K_PREFIX@@': repr(prefix),
        '@@N_TILESCALE@@': str(tile_scale), '@@N_MAXPX@@': str(max_px),
        '@@N_SLEEP@@': '(%d, %d)' % (sleep_lo, sleep_hi), '@@N_SEED@@': str(seed),
        '@@TS_FMT@@': repr(ts_fmt), '@@YEARS@@': str(C.YEARS),
    }
    for k, v in fns.items():
        repl['@@FN_%s@@' % k.upper()] = v
    return repl, dict(alias=alias, prefix=prefix, tile_scale=tile_scale, max_px=max_px,
                      sleep=[sleep_lo, sleep_hi], seed=seed, fn_names=fns, ts_fmt=ts_fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', nargs='*', default=None)
    ap.add_argument('--include-busy', action='store_true')
    a = ap.parse_args()
    C.ensure_dirs()
    tpl = io.open(os.path.join(C.PROD, 'pw_template.tpl'), encoding='utf-8').read()
    accts = a.acct or C.usable_accounts(include_busy=a.include_busy)
    plan_fp = os.path.join(C.PLAN, 'tiles.json')
    if not os.path.isfile(plan_fp):
        raise SystemExit('缺 %s —— 先跑 p2_plan_tiles.py' % plan_fp)
    tiles = json.load(open(plan_fp, encoding='utf-8'))['tiles']

    # 分区分配：按"陆域占比×点数"排序后轮转分配给各账号（均衡负载，且不重叠）
    tiles_sorted = sorted(tiles, key=lambda t: (-t['land_frac'], -t['n_points']))
    assign = {a_: [] for a_ in accts}
    for i, t in enumerate(tiles_sorted):
        assign[accts[i % len(accts)]].append(t['tile'])

    summary = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_accounts': len(accts),
               'n_tiles': len(tiles), 'accounts': {}}
    # 差异化索引必须**按账号稳定**（不能按子集顺序），否则以后换账号子集重生成时
    # 会出现前缀/别名跨账号串号（2026-09-14 东北三省只生成 3 个账号时踩到）
    _order = sorted(C.ACCOUNTS.keys())
    for acct in accts:
        idx = _order.index(acct)
        repl, info = gen_for(acct, idx, tiles, a.include_busy)
        out = tpl
        for k, v in repl.items():
            out = out.replace(k, v)
        left = [m for m in ('@@',) if m in out]
        if left:
            raise SystemExit('模板占位符未替换干净: %s' % out[out.index('@@'):out.index('@@') + 40])
        fp = os.path.join(C.FLEET, 'pw_%s.py' % acct)
        io.open(fp, 'w', encoding='utf-8').write(out)
        ap_ = os.path.join(C.PLAN, 'assign_%s.json' % acct)
        json.dump({'acct': acct, 'proj': C.ACCOUNTS[acct]['proj'],
                   'folder': C.ACCOUNTS[acct]['folder'], 'prefix': info['prefix'],
                   'n_tiles': len(assign[acct]), 'tiles': assign[acct],
                   'differentiation': info}, open(ap_, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        summary['accounts'][acct] = {'prefix': info['prefix'], 'alias': info['alias'],
                                     'tile_scale': info['tile_scale'],
                                     'n_tiles': len(assign[acct]),
                                     'n_exports': len(assign[acct]) * len(C.YEARS)}
        print('生成 %-16s prefix=%-8s alias=%-12s tileScale=%d 分区=%3d 任务=%3d'
              % (acct, info['prefix'], info['alias'], info['tile_scale'],
                 len(assign[acct]), len(assign[acct]) * len(C.YEARS)))
    # 差异化自检：关键指纹不得跨账号重复
    for key in ('prefix', 'alias'):
        vals = [v[key] for v in summary['accounts'].values()]
        assert len(vals) == len(set(vals)), '%s 跨账号重复！' % key
    print('\n差异化自检通过：prefix/alias 跨账号零重复（%d 个账号）' % len(accts))
    json.dump(summary, open(os.path.join(C.PLAN, 'fleet_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('输出目录:', C.FLEET)


if __name__ == '__main__':
    main()
