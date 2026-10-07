# -*- coding: utf-8 -*-
"""p1d_set_deploy_opts.py — 给部署件打开/关闭「局部自适应建模 + K 模型投票」

技术文档 33 阶段 C 的落地开关。默认**不改任何部署件**（保持向后兼容），
只有显式运行本脚本才会写入。

用法：
  python p1d_set_deploy_opts.py --show                      # 查看当前各账号设置
  python p1d_set_deploy_opts.py --acct nhqz5uj --on --k 5   # 打开（局部建模 + 5 模型投票）
  python p1d_set_deploy_opts.py --acct nhqz5uj --off        # 关闭（回到旧的全局单模型）

写入的字段（被 pw_<acct>.py 的 build_clf 读取）：
  local_train      : true  开启局部自适应建模（每瓦片用「本瓦片 ± local_margin_deg」的样本独立训练）
  local_margin_deg : 2.0   邻域外扩度数（2° = 一圈；本地模拟里 3×3=6° 窗最优，5×5 更差）
  k_vote           : 5     训练 K 个模型，导出算子里 ee.ImageCollection(...).mode() 投票
                           （绕开 GEE「单次训练 ≈2 万点」上限，见 doc 30 §三）
  train_frac       : 0.10  每个成员各自的训练子抽样比例（成员之间子样本不同才有投票增益）

实测依据（25km 块留出验证池，多种子配对）：
  全局单模型            OA 0.6453
  + maxNodes 2万        +0.81 ± 0.13 pp（5 种子全正）
  + K=5 投票            +1.09 ± 0.14 pp（5 种子全正）
  + 训练端合并          +2.46 ± 0.35 pp
  —— doc30 全套          0.6907（+4.54 ± 0.36 pp）
  局部建模（K=1）        +6.56 pp（3 种子全正）
  局部 + K5 + 合并       0.7350（+8.80 pp）
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

DEFAULTS = dict(local_train=True, local_margin_deg=2.0, k_vote=5, train_frac=0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', nargs='*', default=None)
    ap.add_argument('--on', action='store_true')
    ap.add_argument('--off', action='store_true')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--margin', type=float, default=2.0)
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--all', action='store_true', help='作用到 plan/deploy_*.json 全部账号')
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(C.PLAN) if f.startswith('deploy_') and f.endswith('.json'))
    if a.acct:
        files = [f for f in files if f[len('deploy_'):-len('.json')] in set(a.acct)]
    if not files:
        raise SystemExit('没有匹配的部署件（%s）' % C.PLAN)

    if a.show or not (a.on or a.off):
        print('%-14s %-12s %-8s %-8s %s' % ('账号', 'local_train', 'margin', 'k_vote', 'train_frac'))
        print('-' * 62)
        for f in files:
            d = json.load(open(os.path.join(C.PLAN, f), encoding='utf-8'))
            print('%-14s %-12s %-8s %-8s %s' % (
                f[len('deploy_'):-len('.json')], d.get('local_train', '(未设→关闭)'),
                d.get('local_margin_deg', '-'), d.get('k_vote', 1),
                d.get('train_frac', '-')))
        if not (a.on or a.off):
            return

    for f in files:
        fp = os.path.join(C.PLAN, f)
        d = json.load(open(fp, encoding='utf-8'))
        if a.on:
            d.update(local_train=True, local_margin_deg=a.margin,
                     k_vote=max(1, a.k), train_frac=a.frac)
            act = '打开'
        else:
            for k in ('local_train', 'local_margin_deg', 'k_vote'):
                d.pop(k, None)
            act = '关闭'
        # 备份原部署件（只备份一次）
        bak = fp + '.bak'
        if not os.path.isfile(bak):
            json.dump(json.load(open(fp, encoding='utf-8')), open(bak, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
        json.dump(d, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('%s %s → %s' % (act, f, {k: d.get(k) for k in
              ('local_train', 'local_margin_deg', 'k_vote', 'train_frac')}))
    print('\n⚠️ 改完后需重新生成 worker：python p3_gen_workers.py --acct <...> --include-busy')
    print('⚠️ 已提交/运行中的任务不受影响（worker 在启动时读部署件）')


if __name__ == '__main__':
    main()
