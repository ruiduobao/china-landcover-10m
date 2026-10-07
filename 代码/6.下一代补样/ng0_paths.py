# -*- coding: utf-8 -*-
"""
ng0_paths.py — 下一代补样：唯一路径/存储事实来源（2026-09-13）

存储分层原则（沿用本机磁盘规则）
  · F 盘 = 代码 / 文档 / 小索引 / 报告（< 数 GB）
  · K 盘 = 重产物（GEE 拉回的候选点、分片、日志归档），K 是项目权威副本所在卷
  · 任何脚本不得硬编码路径，一律 import 本模块

目录结构（K 盘）
  下一代补样/
    ng_idx/                       本地生成：区域定义、分片清单（零 EECU）
    ng_raw/<spec>/                原始候选点（GEE 拉回，按账号×分片落盘，可断点续跑）
    ng_gate/<spec>/               过门槛后的高置信样本
    ng_qa/                        质检报告（硬门槛 json + 分布统计）
    ng_log/                       工作器日志（按账号）
    ng_fleet/                     编队配置、账号指纹、账号×spec 分配表
"""
import os

# ---------- 项目根 ----------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'

# ---------- 代码 / 文档（F 盘）----------
CODED = os.path.join(PROJ, '代码', '6.下一代补样')
DOCD = os.path.join(PROJ, '技术文档')
FLEETD = os.path.join(CODED, 'fleet')                 # 生成的账号专属工作器
SPEC_JSON = os.path.join(CODED, 'ng_spec.json')        # 规格快照（供审计）

# ---------- 数据（K 盘）----------
NG = os.path.join(KB, '数据', '本地处理', '下一代补样')
IDX = os.path.join(NG, 'ng_idx')
RAW = os.path.join(NG, 'ng_raw')
GATE = os.path.join(NG, 'ng_gate')
QA = os.path.join(NG, 'ng_qa')
LOG = os.path.join(NG, 'ng_log')
FLEET = os.path.join(NG, 'ng_fleet')

# ---------- 产物文件名约定 ----------
def raw_path(spec, acct, shard):
    """原始候选点：一账号一 spec 一分片一文件（可断点续跑）"""
    return os.path.join(RAW, spec, f'raw_{spec}_{acct}_s{shard:03d}.parquet')


def gate_path(spec):
    """合并 + 过门槛后的样本"""
    return os.path.join(GATE, f'ng_{spec}.parquet')


def qa_path(spec):
    return os.path.join(QA, f'qa_{spec}.json')


def log_path(acct, tag='run'):
    return os.path.join(LOG, f'{acct}_{tag}.log')


def shard_plan_path(spec):
    return os.path.join(IDX, f'plan_{spec}.json')


def fleet_json_path():
    return os.path.join(FLEET, 'ng_fleet.json')


def alloc_path():
    """账号 × spec 分配表（防同类任务跨账号；同自定域账号对强制不同 spec）"""
    return os.path.join(FLEET, 'ng_alloc.json')


def ensure_all(specs=None):
    """建目录。传 specs 时同时建每个 spec 的子目录 —— 工作器落盘前必须已存在，
    否则 pyarrow 会报 'Cannot save file into a non-existent directory'（本次试点踩到）。"""
    for d in (CODED, FLEETD, NG, IDX, RAW, GATE, QA, LOG, FLEET):
        os.makedirs(d, exist_ok=True)
    for s in (specs or []):
        os.makedirs(os.path.join(RAW, s), exist_ok=True)
        os.makedirs(os.path.join(GATE, s), exist_ok=True)


if __name__ == '__main__':
    ensure_all()
    for k in ('PROJ', 'KB', 'CODED', 'NG', 'IDX', 'RAW', 'GATE', 'QA', 'LOG', 'FLEET'):
        print(f'{k:8s} {globals()[k]}')
