# -*- coding: utf-8 -*-
""">>> 模板文件，由 p3_gen_workers.py 生成具体账号 worker；不要手改本模板以外的副本 <<<

zixen8v8 生产 worker · 全国 10m 土地覆盖（v-final.2026-09-14）
本账号：zixen8v8  项目：braided-horizon-508210-a5  资产目录：lcf_a01  任务名前缀：lcf19

用法：
  python pw_zixen8v8.py --submit --max 1     # 提交 1 个任务（账号级并发=1，勿多提）
  python pw_zixen8v8.py --list                # 看本账号待跑清单
  python pw_zixen8v8.py --cancel <taskid>     # 仅同区重复任务才允许取消
"""
import os
import sys
import io
import json
import time
import random
import argparse

teak_tool_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(teak_tool_here))
import prod_conf as teak_tool          # 配置源（路径/账号/常量）

teak_tool.apply_account_env('zixen8v8')  # 凭证隔离必须在构造任何 ee 对象之前
import ee                              # noqa: E402

K_ACCT = 'zixen8v8'
K_PROJ = 'braided-horizon-508210-a5'
K_FOLDER = 'lcf_a01'
K_PREFIX = 'lcf19'                 # 差异化任务名前缀（防批量识别）
K_TILE_SCALE = 3          # 差异化 tileScale
K_MAX_PX = 10000000000000                  # 差异化 maxPixels
K_SLEEP = (1, 5)                   # 差异化提交间隔（秒）
K_SHUFFLE = 6603                  # 差异化分区顺序种子
K_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


def log(msg):
    print(time.strftime('') + str(msg), flush=True)


def bring_up(tries=5):
    for i in range(tries):
        try:
            ee.Initialize(project=K_PROJ)
            log('[ee] %s / %s ready' % (K_ACCT, K_PROJ))
            return
        except Exception as exc:
            if i == tries - 1:
                raise
            log('initialize retry %d: %s' % (i + 1, str(exc)[:80]))
            time.sleep(20 + 10 * i)


def dump_rows(rows):
    """追加写台账（每个 worker 只碰自己的行；不同进程不共写同一文件）"""
    os.makedirs(teak_tool.LEDGER, exist_ok=True)
    fp = os.path.join(teak_tool.LEDGER, 'ledger_%s.jsonl' % K_ACCT)
    with io.open(fp, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return fp


def open_plan(plan_file='tiles.json'):
    fp = os.path.join(teak_tool.PLAN, plan_file)
    plan = json.load(open(fp, encoding='utf-8'))
    tiles = plan['tiles']
    random.Random(K_SHUFFLE).shuffle(tiles)     # 各账号分区顺序不同
    return plan, tiles


def get_model():
    """部署件：{'mode':'inline','trees':[...]} 或 {'mode':'asset','asset':'projects/...'}"""
    fp = os.path.join(teak_tool.PLAN, 'deploy_%s.json' % K_ACCT)
    if not os.path.isfile(fp):
        raise SystemExit('缺少部署件 %s —— 先跑 p1_deploy_model.py --acct %s' % (fp, K_ACCT))
    d = json.load(open(fp, encoding='utf-8'))
    if d.get('mode') == 'inline':
        clf = ee.Classifier.decisionTreeEnsemble(d['trees'])
        log('模型 inline：%d 树 / %.1f MB' % (len(d['trees']), d.get('size_mb', 0)))
    elif d.get('mode') == 'asset':
        # 样本资产可能是多片（列表）→ flatten 合并；跨账号共享时用 reader_email 对应的授权
        ids = d.get('assets') or [d['asset']]
        # 多片样本必须用链式 merge；ee.FeatureCollection(字符串列表) 会掉进
        # "把字符串当 Feature 列表"的分支 → Invalid GeoJSON geometry（实测）
        fc = ee.FeatureCollection(ids[0])
        for _a in ids[1:]:
            fc = fc.merge(ee.FeatureCollection(_a))
        # GEE 分类器训练的样本量上限实测 ≈2 万点（5 万即报 Computed value is too large）
        # → 训练前做随机子抽样（比例写进部署件），分类图仍逐像元全量计算
        if d.get('train_frac') and d['train_frac'] < 1:
            fc = fc.randomColumn('rr', d.get('train_seed', 7)).filter(
                'rr < %s' % d['train_frac'])
            log('训练子抽样：取 %.0f%%' % (d['train_frac'] * 100))
        # 注意：variablesPerSplit 传 0 会报 'Invalid number of random selected features
        # for splitting: 0'（ee 默认是 None=sqrt(n)），所以**只在给出正整数时才传**
        kw = {'numberOfTrees': d.get('n_trees', 100),
              'minLeafPopulation': d.get('min_leaf', 2),
              'maxNodes': d.get('max_nodes', 0) or None,
              'seed': d.get('seed', 42)}
        if d.get('vars_per_split'):
            kw['variablesPerSplit'] = int(d['vars_per_split'])
        clf = ee.Classifier.smileRandomForest(**kw).train(fc, d.get('label_col', 'cl'),
                                                          teak_tool.FEATS)
        log('模型参数: %s' % kw)
        log('模型 asset-trained：%s（%d 片）' % (ids[0] if len(ids) == 1 else ids[0] + ' ...', len(ids)))
    else:
        raise SystemExit('未知部署模式: %r' % d.get('mode'))
    return clf, d


def solved_set():
    """已完成集合：读本账号台账里 state=DONE 的 (tile,year)"""
    done = set()
    fp = os.path.join(teak_tool.LEDGER, 'ledger_%s.jsonl' % K_ACCT)
    if os.path.isfile(fp):
        rows, task_state = [], {}
        for line in io.open(fp, encoding='utf-8', errors='replace'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
            if r.get('task'):
                task_state[r['task']] = r.get('state', '')
        # 🔴 换仓后旧仓任务的**最终状态**不在新仓 operations 里（台账停在 RUNNING），
        #    必须回查旧仓，否则已完成瓦片会被重复提交（2026-09-15 N0306 实测踩中）
        for _lp in getattr(teak_tool, 'LEGACY_PROJS', {}).get(K_ACCT, []):
            for r_ in poll_ops(_lp):
                if r_.get('task'):
                    task_state[r_['task']] = r_['state']
        for r in rows:      # 两遍扫描：refresh 行不带 tile/year，必须按 task id 回联
            if r.get('tile') and r.get('year') is not None:
                s = task_state.get(r.get('task'), r.get('state', ''))
                if s in ('SUCCEEDED', 'DONE_FETCHED'):
                    done.add((r['tile'], int(r['year'])))
    return done


def ensure_asset_root(path):
    """资产根不存在会让导出直接 FAILED；存在性检查用 read-only，创建仅在本函数内发生"""
    try:
        ee.data.getAsset(path)
        return 'exists'
    except Exception:
        pass
    try:
        ee.data.createAsset({'type': 'Folder'}, path)
        return 'created'
    except Exception as exc:
        s = str(exc)
        return 'exists' if 'already exists' in s else ('error: ' + s[:90])


def compose_clf(tile, d, gclf):
    """按部署件配置返回**该瓦片的分类器列表**（技术文档 33 阶段 C）
       · d['local_train'] 未设/假 → [全局模型]，行为与旧版完全一致（向后兼容）
       · d['local_train'] 为真    → 取「本瓦片 + local_margin_deg 圈」的样本独立训练（局部自适应）
       · d['k_vote'] > 1          → 训练 K 个模型，导出算子里 mode() 投票（绕开单次训练点数上限）
       依据：局部自适应建模在 25km 块留出池上 +6.56pp（3 种子配对，每模型同为 2 万点）。
    """
    if not d.get('local_train'):
        return [gclf]
    ids = d.get('assets') or [d['asset']]
    mg = float(d.get('local_margin_deg', 2.0))      # 2° = 一圈邻域（本地模拟里 3×3 窗最优）
    roi = ee.Geometry.Rectangle([tile['x0'] - mg, tile['y0'] - mg,
                                 tile['x1'] + mg, tile['y1'] + mg])
    fc = ee.FeatureCollection(ids[0])
    for _a in ids[1:]:
        fc = fc.merge(ee.FeatureCollection(_a))     # 多片必须链式 merge（见 get_model 注释）
    fc = fc.filterBounds(roi)
    kw = {'numberOfTrees': d.get('n_trees', 100),
          'minLeafPopulation': d.get('min_leaf', 2),
          'maxNodes': d.get('max_nodes', 0) or None}
    if d.get('vars_per_split'):
        kw['variablesPerSplit'] = int(d['vars_per_split'])
    k = max(1, int(d.get('k_vote', 1)))
    frac = d.get('train_frac')
    base_seed = int(d.get('seed', 42))
    clfs = []
    for _i in range(k):
        sub = fc
        if frac and frac < 1:
            # 每个成员的子样本不同 → 投票才有增益（单次训练点数上限不变）
            sub = sub.randomColumn('rr', base_seed + _i * 131).filter('rr < %s' % frac)
        kw2 = dict(kw)
        kw2['seed'] = base_seed + _i * 17
        clfs.append(ee.Classifier.smileRandomForest(**kw2)
                    .train(sub, d.get('label_col', 'cl'), teak_tool.FEATS))
    log('  局部建模：窗 %s ｜ 成员 %d ｜ 训练子样本 %.0f%%'
                % (roi.bounds().coordinates().getInfo()[:1] if False else '±%.1f°' % mg,
                   k, 100 * (frac if frac and frac < 1 else 1)))
    return clfs


def push_task(tile, year, clfs):
    box = ee.Geometry.Rectangle([tile['x0'], tile['y0'], tile['x1'], tile['y1']])
    aef = (ee.ImageCollection(teak_tool.AEF)
           .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
           .filterBounds(box).mosaic().select(teak_tool.FEATS))
    # clfs 是列表：>1 个成员时做众数投票（K 模型集成，见技术文档 30 §三 / 33 阶段 C）
    if isinstance(clfs, (list, tuple)):
        cls = (aef.classify(clfs[0]) if len(clfs) == 1
               else ee.ImageCollection([aef.classify(_c) for _c in clfs]).mode())
    else:
        cls = aef.classify(clfs)          # 兼容旧的单模型调用
    cls = cls.rename('class').uint8()
    name = '%s_%s_%d' % (K_PREFIX, tile['tile'], year)
    asset_id = 'projects/%s/assets/%s/%s' % (K_PROJ, K_FOLDER, name)
    task = ee.batch.Export.image.toAsset(
        image=cls.clip(box), description=name, assetId=asset_id,
        scale=10, crs='EPSG:4326', region=box,
        maxPixels=K_MAX_PX, shardSize=K_TILE_SCALE * 8)
    # 注：Export.image.toAsset 不接受 tileScale（那是 toDrive 的参数）；
    #     这里用差异化 shardSize 承担同等作用（分片粒度不同 → 提交指纹不同）
    task.start()
    return task, asset_id, name


def poll_ops(proj=None):
    """回读本账号 operations 状态并补写台账（GET，路径不带 :list）"""
    import requests
    import google.oauth2.credentials
    from google.auth.transport.requests import AuthorizedSession
    from ee import oauth as teak_tool_oauth
    cred = os.path.join(teak_tool.cred_home(K_ACCT), '.config', 'earthengine', 'credentials')
    with io.open(cred, encoding='utf-8') as f:
        cj = json.load(f)
    creds = google.oauth2.credentials.Credentials(
        token=None, refresh_token=cj['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=teak_tool_oauth.CLIENT_ID, client_secret=teak_tool_oauth.CLIENT_SECRET,
        scopes=cj.get('scopes', ['https://www.googleapis.com/auth/earthengine']))
    sess = AuthorizedSession(creds)
    sess.proxies = teak_tool.PROXY
    _p = proj or K_PROJ
    url = 'https://earthengine.googleapis.com/v1/projects/%s/operations' % _p
    r = sess.get(url, params={'pageSize': 200},
                 headers={'X-Goog-User-Project': K_PROJ}, timeout=60)
    r.raise_for_status()
    rows = []
    for op in r.json().get('operations', []):
        md = op.get('metadata', {})
        st = md.get('state') or ('DONE' if op.get('done') else 'UNKNOWN')
        ec = 0.0
        for k, v in md.items():
            if 'ecu' in k.lower():
                try:
                    ec = float(v)
                except (TypeError, ValueError):
                    pass
        rows.append({'acct': K_ACCT, 'task': op.get('name', '').split('/')[-1],
                     'desc': md.get('description', ''), 'state': st,
                     'eecu_s': ec, 'begin': md.get('startTime'), 'end': md.get('endTime'),
                     'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
    if rows:
        dump_rows(rows)
    return rows


def go():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--max', type=int, default=1)
    ap.add_argument('--years', type=int, nargs='*', default=K_YEARS)
    ap.add_argument('--tile', default='')
    ap.add_argument('--only-tile', default='')
    ap.add_argument('--plan', default='tiles.json')
    a = ap.parse_args()
    teak_tool.ensure_dirs()

    if a.refresh:
        rows = poll_ops()
        live = [r for r in rows if r['state'] in ('RUNNING', 'PENDING')]
        log('[%s] %d 条 operation，其中活动 %d' % (K_ACCT, len(rows), len(live)))
        return 0

    plan, tiles = open_plan(a.plan)
    done = solved_set()
    todo = []
    for t in tiles:
        if a.only_tile and t['tile'] != a.only_tile:
            continue
        for y in a.years:
            if (t['tile'], int(y)) not in done:
                todo.append((t, int(y)))
    log('[%s] 分区 %d / 待跑 %d（已完成 %d）' % (K_ACCT, len(tiles), len(todo), len(done)))
    if a.list or not a.submit:
        for t, y in todo[:20]:
            print('   %s %d  land=%.2f pts=%s' % (t['tile'], y, t.get('land_frac', 0),
                                                  t.get('n_points', '-')))
        if len(todo) > 20:
            print('   ... 共 %d' % len(todo))
        return 0

    # 提交前先确认没有活动任务（账号级并发=1；同账号多仓也不并行）
    rows = poll_ops()
    # 🔴 换仓后新仓的 operations 看不到旧仓在跑的任务 → 必须把旧仓也查一遍，
    #    否则会给同一瓦片重复提交（2026-09-15 实测踩中，浪费 17 EECU·h 一次）
    live = [r for r in rows if r['state'] in ('RUNNING', 'PENDING')]
    for _lp in getattr(teak_tool, 'LEGACY_PROJS', {}).get(K_ACCT, []):
        for r_ in poll_ops(_lp):
            if r_['state'] in ('RUNNING', 'PENDING'):
                live.append(r_)
    if live:
        log('[%s] 已有 %d 个活动任务 → 不提交（等其完成）: %s'
                    % (K_ACCT, len(live), live[0]['desc'][:60]))
        return 0

    bring_up()
    root = ensure_asset_root('projects/%s/assets/%s' % (K_PROJ, K_FOLDER))
    log('资产根 %s/%s → %s' % (K_PROJ, K_FOLDER, root))
    gclf, meta = get_model()
    if a.tile:
        todo = [(t, y) for t, y in todo if t['tile'] == a.tile]
    n = 0
    for t, y in todo[:a.max]:
        for att in range(3):
            try:
                # 局部建模时**每个瓦片训练自己的模型**（技术文档 33 阶段 C）；
                # 未开启 local_train 时 MKCLF 原样返回全局模型，行为与旧版一致。
                _clfs = compose_clf(t, meta, gclf)
                task, asset_id, name = push_task(t, y, _clfs)
                dump_rows([{'acct': K_ACCT, 'tile': t['tile'], 'year': y,
                                'desc': name, 'asset': asset_id, 'task': task.id,
                                'state': 'SUBMITTED', 'eecu_s': 0.0,
                                'ts': time.strftime('%Y-%m-%d %H:%M:%S')}])
                log('提交 %s → task %s' % (name, task.id))
                n += 1
                break
            except Exception as exc:
                log('提交重试 %d/%d: %s' % (att + 1, 3, str(exc)[:110]))
                time.sleep(random.uniform(*K_SLEEP))
        else:
            log('提交失败 %s %d' % (t['tile'], y))
        time.sleep(random.uniform(*K_SLEEP))
    log('[%s] 本轮提交 %d 个' % (K_ACCT, n))
    return 0


if __name__ == '__main__':
    raise SystemExit(go())
