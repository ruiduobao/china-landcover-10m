# -*- coding: utf-8 -*-
"""advance2.py — 第二阶段（瓦片试产）幂等推进器：t_upload → t_submit → t_qa → done
每轮：单实例锁 → 备份 → 执行当前阶段一步 → 落 STATE2.json
"""
import os, sys, time, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC

STATE2 = os.path.join(VC.ROOT, 'STATE2.json')


def st_t_upload(st, pool):
    import t_all
    t_all.upload_pools()
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'), {})
    n = sum(1 for t in t_all.TILES if reg.get('tile_pool_%s' % t))
    VC.emit('池就绪 %d/%d' % (n, len(t_all.TILES)))
    if n == len(t_all.TILES):
        return 't_submit'
    return None


def st_t_submit(st, pool):
    import t_all
    t_all.submit()
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'), {})
    n = sum(1 for t in t_all.TILES if isinstance(reg.get('tile_raster_%s' % t), dict))
    VC.emit('栅格任务已提交 %d/%d' % (n, len(t_all.TILES)))
    if n == len(t_all.TILES):
        return 't_qa'
    return None


def st_t_qa(st, pool):
    import t_all
    fp = t_all.qa()
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'), {})
    done = 0
    for t in t_all.TILES:
        try:
            win = t_all.winner_of(t)
        except Exception:
            continue
        if win['state'] == 'COMPLETED':
            done += 1
        elif win['state'] in ('FAILED', 'CANCELLED'):
            raise SystemExit('NEED_MANUAL: 瓦片 %s 导出 %s' % (t, win['state']))
    VC.emit('瓦片完成 %d/%d' % (done, len(t_all.TILES)))
    if done == len(t_all.TILES):
        st['tile_report'] = fp
        VC.save_state2(st)
        return 'done'
    return None


STAGES = {'t_upload': st_t_upload, 't_submit': st_t_submit, 't_qa': st_t_qa}


def main():
    t0 = time.time()
    lock = VC.acquire_lock()
    if lock is None:
        VC.emit('已有 advance 在跑，本轮退出'); return
    st = VC.jload(STATE2, {})
    stage = st.get('stage', 't_upload')
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    VC.emit('===== advance2 阶段=%s =====' % stage)
    try:
        import backup
        backup.main()
    except Exception as e:
        VC.emit('备份失败（不阻断）: %s' % str(e)[:80])
    try:
        if stage == 'done':
            VC.emit('瓦片试产阶段已完成（详见 reports/瓦片试产QA_*.md）'); return
        nxt = STAGES[stage](st, pool)
        if nxt:
            st = VC.jload(STATE2, {})       # 重读防覆盖
            st['stage'] = nxt
            VC.save_state2(st)
            VC.emit('阶段 %s → %s' % (stage, nxt))
        else:
            VC.emit('阶段 %s 未完成，下一轮继续' % stage)
    except SystemExit as e:
        VC.emit('NEED_MANUAL: %s' % e)
        st['need_manual'] = str(e); VC.save_state2(st)
        raise
    except Exception:
        st['last_error'] = traceback.format_exc()[-1200:]
        VC.save_state2(st)
        VC.emit('异常（下轮重试）:\n%s' % st['last_error'])
    finally:
        VC.release_lock(lock)
        VC.emit('===== advance2 结束（%.1f 分钟）=====' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
