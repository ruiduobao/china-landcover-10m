# -*- coding: utf-8 -*-
"""advance.py — v3.1 实验总控状态机（幂等，供定时任务每轮调用）
每轮：加锁 → backup.py 快照 → 执行当前阶段一步 → 落 STATE.json → 退出。
阶段链：sweep → upload_eval → submit_abd → poll(A/B/D) → report_abd → w5 → submit_w5
        → poll_w5 → carm_sample → carm_submit → poll_c → summary → done
异常约定：SystemExit(NEED_MANUAL...) = 需人工，定时轮只汇报不重试；其他异常记 STATE 下轮重试。
"""
import os, sys, time, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC


def up_root_of(pool):
    return 'projects/%s/assets' % VC.pid_of(pool['uploader'])


def open_jobs():
    jobs = VC.jload(VC.JOBS_FP, [])
    return [j for j in jobs if j.get('attempts') and
            j['attempts'][-1].get('state') in ('SUBMITTED', 'RUNNING', 'DOWNLOAD_FAIL')]


def st_sweep(st, pool):
    import s0_sweep
    s0_sweep.main()
    return 'upload_eval'


def st_upload_eval(st, pool):
    import s1_upload_eval
    s1_upload_eval.main()
    ur = up_root_of(pool)
    shared = VC.jload(os.path.join(VC.DATA, 'shared_done.json'), {})
    ready = [w for w in ['w1', 'w2', 'w3', 'w4']
             if shared.get(VC.reg_get('eval_%s' % w) or '%s/v31e_%s_merged' % (ur, w))]
    if len(ready) == 4:
        VC.emit('w1-w4 评估资产就绪且已共享 → 冒烟+提交 ABD')
        return 'submit_abd'
    return None


def st_submit_abd(st, pool):
    import s2_submit
    s2_submit.main()
    return 'poll'


def st_poll(st, pool):
    import s3_poll_download, s4_eval
    s3_poll_download.main()
    s4_eval.main()
    jobs = VC.jload(VC.JOBS_FP, [])
    pending = [j for j in jobs if not j.get('done')]
    if pending:
        in_flight = any(j.get('attempts') and j['attempts'][-1].get('state')
                        in ('SUBMITTED', 'RUNNING', 'PENDING') for j in pending)
        retryable = [j for j in pending if j.get('attempts') and
                     j['attempts'][-1].get('state') in ('FAILED', 'CANCELLED', 'DOWNLOAD_FAIL')
                     and len(j['attempts']) < 2]
        unsubmitted = [j for j in pending if not j.get('attempts')]
        if in_flight or retryable or unsubmitted:
            import s2_submit
            s2_submit.main()  # 提交未交的 + 重试失败的（内部跳过在飞/超限）
            return None
        raise SystemExit('NEED_MANUAL: 2 次尝试仍未完成的任务: %s' %
                         [j['job_id'] for j in pending][:6])
    s4_eval.aggregate()
    nxt = dict(poll='report_abd', poll_w5='carm_sample', poll_c='summary')[st['stage']]
    VC.emit('%s 全部任务落地 → %s' % (st['stage'], nxt))
    return nxt


def st_report_abd(st, pool):
    import s5_report
    v = s5_report.main()
    if v is None:
        return None
    return 'w5'


def st_w5(st, pool):
    import s6_w5_build, s1_upload_eval
    s6_w5_build.main()
    s1_upload_eval.main()  # w5 评估点上传（若 eval_w5.parquet 已生成）
    ur = up_root_of(pool)
    tr_ready = VC.reg_get('train_w5') or VC.asset_exists(pool['uploader'], '%s/v31s_w5_merged' % ur)
    ev_ready = VC.reg_get('eval_w5') or VC.asset_exists(pool['uploader'], '%s/v31e_w5_merged' % ur)
    if tr_ready and ev_ready:
        shared = VC.jload(os.path.join(VC.DATA, 'shared_done.json'), {})
        full = VC.reg_get('train_w5') or '%s/v31s_w5_merged' % ur
        if not shared.get(full):
            emails = ['user:' + pool['emails'][a] for a in pool['workers'] if a in pool['emails']]
            try:
                e, _ = VC.ctx(pool['uploader'])
                e.data.setAssetAcl(full, {'readers': emails})
                shared[full] = time.strftime('%Y-%m-%d %H:%M')
                VC.jsave(shared, os.path.join(VC.DATA, 'shared_done.json'))
            except Exception as e2:
                VC.emit('w5 训练共享失败: %s' % str(e2)[:90])
        VC.emit('w5 就绪 → 提交 w5 任务')
        return 'submit_w5'
    return None


def st_submit_w5(st, pool):
    import s2_submit
    s2_submit.main()
    return 'poll_w5'


def st_carm_sample(st, pool):
    import s7_carm
    s7_carm.main()
    done = VC.state().get('carm_sampled', {})   # 重读（s7 本轮可能刚写入）
    st['carm_sampled'] = done
    if all(w in done for w in ('w2', 'w5')):
        return 'carm_submit'
    return None


def st_carm_submit(st, pool):
    import s7_carm, s2_submit
    s7_carm.stage_submit(pool)   # 追加 C 臂任务（直接调 submit 阶段，不走 main 的默认 sample）
    s2_submit.main()             # 提交追加的 C 任务
    jobs = VC.jload(VC.JOBS_FP, [])
    n_c = sum(1 for j in jobs if j.get('arm') == 'C')
    if n_c == 0:
        raise SystemExit('NEED_MANUAL: C 臂任务未追加（检查 s7 stage_submit）')
    st['carm_submitted'] = time.strftime('%Y-%m-%d %H:%M')
    VC.save_state(st)
    return 'poll_c'


def st_summary(st, pool):
    import s8_summary
    s8_summary.main()
    return 'done'


STAGES = {
    'sweep': st_sweep,
    'upload_eval': st_upload_eval,
    'submit_abd': st_submit_abd,
    'poll': st_poll,
    'report_abd': st_report_abd,
    'w5': st_w5,
    'submit_w5': st_submit_w5,
    'poll_w5': st_poll,
    'carm_sample': st_carm_sample,
    'carm_submit': st_carm_submit,
    'poll_c': st_poll,
    'summary': st_summary,
}
NEXT_PRE = {'poll': 'report_abd', 'poll_w5': 'carm_sample', 'poll_c': 'summary'}


def main():
    t0 = time.time()
    lock = VC.acquire_lock()
    if lock is None:
        VC.emit('已有 advance 在跑，本轮退出')
        return
    st = VC.state()
    stage = st.get('stage', 'sweep')
    pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
    VC.emit('===== advance 阶段=%s（PID %d）=====' % (stage, os.getpid()))
    try:
        import backup
        backup.main()
    except Exception as e:
        VC.emit('备份失败（继续，不阻断）: %s' % str(e)[:100])
    try:
        if stage == 'done':
            VC.emit('实验已全部完成（详见 reports/）')
            return
        if not pool:
            import s0_sweep
            s0_sweep.main()
            pool = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'))
        nxt = STAGES[stage](st, pool)
        if nxt:
            st = VC.state()      # 重读（阶段函数可能已写入结果，避免旧快照覆盖）
            st['stage'] = nxt
            VC.save_state(st)
            VC.emit('阶段 %s → %s' % (stage, nxt))
        else:
            VC.emit('阶段 %s 未完成，下一轮继续' % stage)
    except SystemExit as e:
        VC.emit('NEED_MANUAL: %s' % e)
        st['need_manual'] = str(e)
        VC.save_state(st)
        raise
    except Exception:
        st['last_error'] = traceback.format_exc()[-1500:]
        st['last_error_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        VC.save_state(st)
        VC.emit('异常（下轮重试）:\n%s' % st['last_error'])
    finally:
        VC.release_lock(lock)
        VC.emit('===== advance 结束（%.1f 分钟）=====' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
