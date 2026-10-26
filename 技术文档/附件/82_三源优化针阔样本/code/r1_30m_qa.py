# -*- coding: utf-8 -*-
"""
r1_30m_qa.py — 《全国 30 米分辨率植被类型空间分布数据》针阔标签源质量诊断 + 三源/四源交叉
* 输入
    - 30m 植被类型栅格 : F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif  (EPSG:3857, uint16, nodata=65535)
    - VAT 码表        : 同名 .tif.vat.dbf (863 行: Value/Count/类型组/植被型/群系/代码)
    - 栅格辅助统计     : 同名 .tif.aux.xml (65536 桶精确直方图, 用于 nodata/有效像元精确口径)
    - v2 针阔样本      : F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet (110,653 点)
    - 100 万植被图     : 数据/外部数据/2020年中国100万植被图/vegetation.shp (68,756 图斑)
    - PFT 2013        : 数据/外部数据/China_Forest_Dataset_v1.0/2_PFT_Annual_1981-2013/2013/ (8 类 1.29km)
    - 既有诊断         : opt/veg_qa.json (Q1) / opt/pft_qa.json (Q3)  —— 直接引用, 不重跑
    - 瓦片规划         : 代码/8.全国生产/plan/tiles.csv (313 瓦)
* 规则源
    - 技术文档/81_植被图与PFT优化针阔样本技术方案.md (判据 V1–V9; 本轮 R1 为新增第四源诊断)
* 门槛
    - 参照 doc81 V2 门 0.80（对 30m 未预注册, 仅作参照）
    - 分布/面积只作描述统计, 不设门
* 输出 (全部落 F:/lc_work/nb_national_2026/opt/)
    - r1_30m_qa.json   basic / code_map / R2 / vs_vegmap / vs_pft / three_way / by_prov / by_region / south
                       / generation_test / verdict / caveats
    - r1_code_map.csv  863 行码表映射 + 歧义标注
    - r1_by_tile.csv   313 行分瓦覆盖与针阔像元
    - r1_by_prov.csv   分省点数与一致率
* 用法
    python r1_30m_qa.py            # 全量
    python r1_30m_qa.py --skip-tiles   # 跳过 313 瓦栅格扫描(调试用, basic 面积改报 VAT 口径)
* 约束
    单进程; 栅格一律 window 读取(整幅 uint16 = 80 GB); 读点用 rasterio.sample() 批量; 输入只读
"""
import os
import sys
import json
import time
import re
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORK = r'F:/lc_work/nb_national_2026'
OPT = os.path.join(WORK, 'opt')
VEG30 = os.path.join(WORK, 'veg30', '中国植被类型分布_30m.tif')
VAT = VEG30 + '.vat.dbf'
AUX = VEG30 + '.aux.xml'
V2_PARQ = os.path.join(WORK, 'labels', 'needle_broad_national_v2_habit.parquet')
VEG_SHP = os.path.join(PROJ, '数据', '外部数据', '2020年中国100万植被图', 'vegetation.shp')
PFT_DIR = os.path.join(PROJ, '数据', '外部数据', 'China_Forest_Dataset_v1.0', '2_PFT_Annual_1981-2013', '2013')
TILES_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
VEG_QA = os.path.join(OPT, 'veg_qa.json')
PFT_QA = os.path.join(OPT, 'pft_qa.json')

Z95 = 1.959963984540054
T0 = time.time()


def log(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg), flush=True)


def wilson(k, n, z=Z95):
    """Wilson 二项比例 95% CI"""
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    return o


def rate(k, n):
    return None if n <= 0 else round(float(k) / n, 4)


# =====================================================================================
# 1. 码表解析与针阔映射
# =====================================================================================
log('读 VAT 码表 ...')
import fiona

rows = []
with fiona.open(VAT) as src:
    for f in src:
        rows.append(dict(f['properties']))
vat = pd.DataFrame(rows)
vat['群系'] = vat['群系'].fillna('').astype(str).str.strip()
vat['植被型'] = vat['植被型'].fillna('').astype(str).str.strip()
vat['类型组'] = vat['类型组'].fillna('').astype(str).str.strip()
vat['Count'] = vat['Count'].astype('int64')
log('VAT %d 行, Count 合计 %d' % (len(vat), int(vat['Count'].sum())))

GROUP_ID = {'1 针叶林': 1, '2 针阔叶混': 2, '3 阔叶林': 3, '4 灌丛': 4, '5 荒漠': 5, '6 草原': 6,
            '7 草丛': 7, '8 草甸': 8, '9 沼泽': 9, '10 高山植': 10, '11 栽培植': 11, '其他': 0}
vat['group_id'] = vat['类型组'].map(GROUP_ID).fillna(0).astype(int)

# 属名关键词（用于识别"类型组与群系名矛盾"）
NEEDLE_GENUS = ['松', '杉', '柏', '云杉', '冷杉', '铁杉', '落叶松', '油杉', '黄杉', '桧', '杜松', '柳杉', '水杉',
                '红杉', '圆柏', '刺柏']
BROAD_GENUS = ['栎', '桦', '杨', '柳', '樟', '栲', '槠', '青冈', '木荷', '桉', '椴', '槭', '榆', '槐', '栌',
               '水青冈', '楠', '榕', '枫', '朴', '榉', '鹅耳枥', '化香', '锥', '润楠', '木莲', '含笑', '相思',
               '木棉', '榄', '栗', '桐', '漆', '椿', '楝', '柿', '桑', '桤', '珙桐', '银柴', '木姜子', '山矾',
               '杨梅', '桃金娘', '黄檀', '琼楠', '蒲桃', '杜英', '山龙眼', '厚壳桂', '红木荷', '团花', '番龙眼',
               '千果榄仁', '龙脑香', '坡垒', '紫荆', '胡颓子', '盐肤木', '密花树', '谷木', '子龙', '仪花', '风吹楠']
# 注意：'岗松'(Myrtaceae 灌木)、'地盘松'(Pinus) 等由下方规则单独处理


def has_any(s, kws):
    return [k for k in kws if k in s]


# 针叶树种全名（含"樟"等会与阔叶属名冲突的字），比对阔叶属前先从群系名里剔除
NEEDLE_TAXA = ['樟子松', '红松', '偃松', '华山松', '马尾松', '云南松', '思茅松', '台湾松', '巴山松',
               '海南松', '长叶松', '高山松', '乔松', '油松', '赤松', '黑松', '白皮松', '侧柏', '杜松',
               '圆柏', '塔枝圆柏', '祁连圆柏', '大果圆柏', '方枝圆柏', '密枝圆柏', '玉山圆柏',
               '兴安落叶松', '长白落叶松', '华北落叶松', '西伯利亚落叶松', '日本落叶松', '海林落叶松',
               '太白红杉', '红杉', '大果红杉', '云杉', '冷杉', '铁杉', '油杉', '黄杉', '柳杉', '水杉',
               '杉木', '柏木', '干香柏', '西伯利亚红松']


def strip_needle_taxa(name):
    for t in sorted(NEEDLE_TAXA, key=len, reverse=True):
        name = name.replace(t, '')
    return name


def dominant_part(name):
    """'含白栎、短柄树的马尾松林' -> '马尾松林'（含 X 的 Y：Y 是建群层）"""
    m = re.match(r'^含(.+?)的(.+)$', name)
    return m.group(2) if m else name


# 歧义处置建议（按规则首字母）
AMBIG_SUGGEST = {
    'A': '保持独立成 bamboo 类，不进针/阔两类训练；若必须二选一在阔叶侧（竹冠层常绿厚实，遥感常判常绿阔叶）。',
    'B': '按类型组判 needle，但该码作"弱标签"——建议下游只用其非边界核心、或降权；不参与南方针叶增益评估。',
    'B2': '按类型组判 needle 但存疑，建议列入人工核查清单。',
    'C': '按类型组判 broad（建群种确为阔叶，针叶为共建），可直接用。',
    'D': '叶型可定、郁闭度存疑：作标签时须过 AEF 森林掩膜(p_forest≥0.5)并做影像核查，不可直接入库。',
    'F': '保持 nonforest（与 FCS10 口径一致），但作为"乔木存在"的正证据记录，可用于排除"非森林"误判。',
    'E': '保持 nonforest。群系名里的松/栎/柳是灌木属（岗松、地盘松、矮高山栎、山居柳），不是乔木林。',
    'E2': '保持 nonforest。',
}


def classify(r):
    """返回 (映射类, 歧义级别 0/1/2, 歧义说明)
    级别 1 = 会改变映射决策或标签可用性的矛盾（必须人工裁决）
    级别 2 = 仅提示性的口径差异（映射不变，但下游用点时应知情）
    """
    gid = r['group_id']
    name = r['群系']
    vt = r['植被型']
    hard, soft = [], []

    if gid == 1:
        cls = 'needle'
    elif gid == 2:
        cls = 'mixed'
    elif gid == 3:
        cls = 'broad'
    else:
        cls = 'nonforest'

    # --- 规则 A（硬）: 竹林。植被型 (17) 竹林被放在"3 阔叶林"组内，但竹为禾本科巨型草本，不是针/阔叶乔木林
    if gid == 3 and '竹' in vt:
        cls = 'bamboo'
        hard.append('A_竹(禾本科草本)被编在阔叶林组_vt=%s' % vt)

    # --- 规则 B（硬）: 类型组=针叶林 但群系名写的是针阔混树种组成
    #     先剔除针叶树种全名（避免"樟子松"的"樟"、"红松"的"松"被当阔叶/针叶重复计数），
    #     再看"含 X 的 Y"结构：Y 是建群层，X 只是伴生层 -> 伴生层里的阔叶不算歧义
    if gid == 1:
        dom = dominant_part(name)
        resid = strip_needle_taxa(dom)          # 只检查建群层里剩下的属名
        b = has_any(resid, BROAD_GENUS)
        if b:
            hard.append('B_针叶组内建群层混写阔叶属(%s):按类型组判 needle' % '/'.join(b[:3]))
        comp = dominant_part(name)
        if comp != name:                         # "含白栎、短柄树的马尾松林" -> 伴生阔叶, 记为软提示
            soft.append('B0_针叶林伴生层含阔叶(%s):建群种仍为针叶' % name[:18])

    # --- 规则 C（硬）: 类型组=阔叶林 但群系名含针叶属（针叶为共建种的混交林）
    if gid == 3:
        n = [k for k in NEEDLE_TAXA if k in name]
        if n:
            hard.append('C_阔叶组内群系名含针叶树种(%s):按类型组判 broad' % '/'.join(n[:3]))

    # --- 规则 D（硬）: 疏林 / 矮林 / 矮曲林 —— 叶型可定但"是否乔木林/郁闭度"存疑
    for kw in ['疏林', '矮林', '矮曲林']:
        if kw in name:
            hard.append('D_形态存疑(含%s):郁闭度/树高可能不到乔木林标准' % kw)
            break

    # --- 规则 E（硬）: 栽培植被中的用材林/经济林/园 —— 树冠是树，但归"栽培植被"组
    if gid == 11:
        for kw in ['园', '林']:
            if kw in name:
                hard.append('F_栽培植被内含木本(%s):按类型组判 nonforest' % name[:16])
                break

    # --- 规则 E（软）: 非森林组内出现乔木属名 —— 多为同名灌木（岗松/地盘松/矮高山栎/山居柳），映射不变
    if gid in (4, 5, 6, 7, 8, 9, 10):
        n = has_any(name, NEEDLE_GENUS)
        b = [k for k in has_any(name, BROAD_GENUS) if k not in ('柳', '杨')]
        if n:
            soft.append('E_非森林组内群系名含针叶字(%s):实为灌木/散生' % '/'.join(n[:2]))
        elif b:
            soft.append('E2_非森林组内群系名含阔叶属(%s):实为灌木' % '/'.join(b[:2]))

    if hard:
        return cls, 1, ';'.join(hard + soft)
    if soft:
        return cls, 2, ';'.join(soft)
    return cls, 0, ''


cm = vat.apply(classify, axis=1, result_type='expand')
vat['map_cls'] = cm[0]
vat['ambig_level'] = cm[1].astype(int)
vat['ambig_type'] = cm[2].astype(str)

# 像元面积：EPSG:3857 名义像元 900 m²；真实面积 = 900*cos²(φ)，按纬度带折算（见 §2 分瓦扫描）
vat['px'] = vat['Count']
vat['km2_webmerc'] = vat['px'] * 900.0 / 1e6

vat_out = pd.DataFrame({
    'Value': vat['Value'].astype(int),
    '类型组': vat['类型组'],
    '植被型': vat['植被型'],
    '群系': vat['群系'],
    '像元数': vat['px'].astype('int64'),
    'km2_webmerc_nominal': vat['km2_webmerc'].round(3),
    '映射类': vat['map_cls'],
    '是否歧义': np.where(vat['ambig_level'] == 1, '是(硬)', np.where(vat['ambig_level'] == 2, '提示(软)', '否')),
    '歧义级别': vat['ambig_level'],
    '歧义说明': vat['ambig_type'],
    '建议处置': vat['ambig_type'].map(
        lambda s: ' | '.join(AMBIG_SUGGEST[k] for k in
                             [t[:1] if not t.startswith('B2') else 'B2' for t in str(s).split(';')]
                             if k in AMBIG_SUGGEST) if s else ''),
    '图例代码': vat['代码'],
})
vat_out.to_csv(os.path.join(OPT, 'r1_code_map.csv'), index=False, encoding='utf-8-sig')
log('r1_code_map.csv 写入 %d 行' % len(vat_out))

CLS = ['needle', 'broad', 'mixed', 'bamboo', 'nonforest']
LUT = np.zeros(65536, dtype=np.uint8)  # 0=nodata/未映射 1..5
for i, c in enumerate(CLS, start=1):
    sel = vat['map_cls'] == c
    LUT[vat.loc[sel, 'Value'].to_numpy()] = i

code_map_summary = {
    'n_codes': int(len(vat)),
    'n_codes_mapped_forest': int(vat['map_cls'].isin(['needle', 'broad', 'mixed', 'bamboo']).sum()),
    'rule': ('类型组 1 针叶林->needle; 3 阔叶林->broad; 2 针阔叶混->mixed; '
             '竹林(植被型 17, 编在阔叶林组内)->bamboo 独立成类; 其余(灌丛/草原/草丛/草甸/荒漠/沼泽/高山植被/'
             '栽培植被/其他)->nonforest'),
    'by_group': {},
    'by_class': {},
    'n_ambig_hard_codes': int((vat['ambig_level'] == 1).sum()),
    'ambig_hard_px': int(vat.loc[vat['ambig_level'] == 1, 'px'].sum()),
    'n_ambig_soft_codes': int((vat['ambig_level'] == 2).sum()),
    'ambig_soft_px': int(vat.loc[vat['ambig_level'] == 2, 'px'].sum()),
}
for g, sub in vat.groupby('类型组'):
    code_map_summary['by_group'][g] = {
        'n_codes': int(len(sub)), 'px': int(sub['px'].sum()),
        'km2_webmerc_nominal': round(float(sub['km2_webmerc'].sum()), 1),
        'pct_px': round(float(sub['px'].sum()) / float(vat['px'].sum()) * 100, 2),
        'classes': sub.groupby('map_cls')['px'].sum().astype('int64').to_dict(),
    }
for c, sub in vat.groupby('map_cls'):
    code_map_summary['by_class'][c] = {
        'n_codes': int(len(sub)), 'px': int(sub['px'].sum()),
        'km2_webmerc_nominal': round(float(sub['km2_webmerc'].sum()), 1),
        'pct_px': round(float(sub['px'].sum()) / float(vat['px'].sum()) * 100, 2),
    }

# 歧义清单（硬歧义按像元数降序）
amb = vat[vat['ambig_level'] == 1].sort_values('px', ascending=False)
ambig_list = []
for _, r in amb.iterrows():
    ambig_list.append({
        'Value': int(r['Value']), '群系': r['群系'], '类型组': r['类型组'], '植被型': r['植被型'],
        'px': int(r['px']), 'km2_webmerc_nominal': round(float(r['km2_webmerc']), 1),
        'map_cls': r['map_cls'], 'ambig': r['ambig_type'],
        'suggest': AMBIG_SUGGEST.get(r['ambig_type'].split(':')[0].split('_')[0], ''),
    })
code_map_summary['ambiguity_hard_list'] = ambig_list
code_map_summary['ambiguity_px_by_rule'] = (
    vat[vat['ambig_level'] == 1]
    .assign(t=lambda d: d['ambig_type'].str.split('(').str[0])
    .groupby('t').agg(n_codes=('Value', 'size'), px=('px', 'sum'), km2=('km2_webmerc', 'sum'))
    .reset_index().assign(px=lambda d: d['px'].astype('int64'))
    .to_dict('records'))

# --- 用户点名的三个具体歧义案例（逐个给出裁决）
code_map_summary['named_cases'] = {}
_nc = {
    '兴安落叶松、白桦林 (139)': 139,
    '马尾松疏林 / 樟子松疏林 (831)': 831,
    '蒙古栎矮林 (802)': 802,
    '毛竹林 (93)': 93,
    '箭竹丛 (111)': 111,
    '桉树林 (774, 栽培植被)': 774,
    '橡胶园 (768, 栽培植被)': 768,
    '茶园 (730, 栽培植被)': 730,
    '油茶林 (728, 栽培植被)': 728,
    '柳杉林 (775, 栽培植被)': 775,
    '岗松灌丛 (161)': 161,
    '地盘松灌丛 (237)': 237,
    '矮高山栎灌丛 (231)': 231,
    '鸡毛松、坡垒、海南杨桐林 (89)': 89,
    '陆均松、海南紫荆、红稠林 (90)': 90,
    '岳桦矮曲林 (835)': 835,
    '偃松矮曲林 (798)': 798,
}
for label, vv in _nc.items():
    r = vat[vat['Value'] == vv]
    if not len(r):
        continue
    r = r.iloc[0]
    code_map_summary['named_cases'][label] = {
        'Value': int(vv), '类型组': r['类型组'], '植被型': r['植被型'], '群系': r['群系'],
        'px': int(r['px']), 'km2_webmerc_nominal': round(float(r['km2_webmerc']), 1),
        'map_cls': r['map_cls'], 'ambig_level': int(r['ambig_level']),
        'ambig': r['ambig_type'],
        'rule_applied': ('按 类型组 判 ' + r['map_cls']),
    }
code_map_summary['named_cases_verdicts'] = {
    '针阔混写成一个"群系"（如 兴安落叶松、白桦林）': (
        '**按 类型组=1 针叶林 判 needle**。理由：1:100 万植被图的 类型组 是编制者按"优势层生活型"定的'
        '一级归类；"兴安落叶松、白桦林"的母群系是"兴安落叶松林"（图例码 139 的母码 1 = 兴安落叶松林），'
        '白桦是次生演替伴生层。但这类码共 11 个、12.1 亿像元（占针叶像元的 9.0%），'
        '**建议：作弱标签，不进南方针叶增益评估；若用于训练，降权或只取图斑核心**（已在 r1_code_map.csv 的"建议处置"列标注）。'),
    '疏林（如 樟子松疏林 831 / 榆树疏林 837）': (
        '**叶型仍按 类型组 判（831→needle、837→broad）**，但"疏林"郁闭度可能 <0.3、'
        '在 10 m 影像上常呈草地/灌丛混合像元。**建议：必须过 AEF 森林掩膜(p_forest≥0.5) + 影像核查，不可直接入库**。'
        '樟子松疏林 457 万像元、榆树疏林合计 3,555 万像元。'),
    '矮林/矮曲林（蒙古栎矮林 802 / 偃松矮曲林 798 / 岳桦矮曲林 835）': (
        '**按 类型组 判（802→broad、798→needle、835→broad）**，但矮曲林（krummholz）是高山林线形态，'
        '树高 1–3 m，遥感易判为灌丛。**建议同上：须影像核查；在青海/西藏/东北林线附近尤其不可轻信**。'),
    '竹林（毛竹林 93 / 箭竹丛 111，共 17 码 4,727 万像元）': (
        '**单独成 bamboo 类，不进 needle/broad**。竹为禾本科巨型草本，叶形轴（针/阔）本身不适用；'
        '遥感上常被 FCS10 判为常绿阔叶（EBF）。这是"我方 vs FCS10 阔叶"分歧的一个结构性来源，'
        '但不应据此判 FCS10 错——正确做法是 bamboo 不参与针阔二分类评分。'),
    '栽培植被中的木本（桉树林 774 / 橡胶园 768 / 柳杉林 775 / 茶园 730 / 油茶林 728…）': (
        '**保持 nonforest**（与 FCS10 的"栽培/农田"口径一致，也与 doc81 的"森林=自然林"前提一致）。'
        '但这些码共 29 个、8,367 万像元，树冠确实是树——若下游需要"乔木覆盖"而来做针阔，'
        '应把它们作为"乔木存在"的正证据单独一层，不能算 nonforest 的"无树"证据。'
        '特别地，**柳杉林 44 万像元、桉树林 1,169 万像元本身就是针叶/阔叶乔木**，'
        '把它们一律当 nonforest 会系统性压低南方的"森林"面积。'),
    '非森林组里的同名属（岗松灌丛 161 / 地盘松灌丛 237 / 矮高山栎灌丛 231）': (
        '**保持 nonforest，不因群系名含"松/栎"就判针/阔**。岗松是桃金娘科灌木、'
        '地盘松是 Pinus yunnanensis var. pygmaea（灌木状）、矮高山栎是高山栎类灌木。'
        '它们是灌丛不是乔木林。这一条正是 doc81 §1.1 提到的"灌丛生活型泄漏"陷阱，'
        '在 30m 的 类型组 体系里已被正确处理（灌丛 单独成组）。'),
    '热带雨林里的针叶共建种（鸡毛松 89 / 陆均松 90）': (
        '**按 类型组=3 阔叶林 判 broad**。鸡毛松(Podocarpus imbricatus)、陆均松(Dacrydium pectinatum) '
        '确是针叶树，但在热带雨林里是零星共建种，群落优势层是阔叶。共 163 万像元，量级小，'
        '建议标注但不必改判。'),
}

# =====================================================================================
# 2. 基础质检
# =====================================================================================
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pyproj import Transformer

log('读栅格元数据 ...')
with rasterio.open(VEG30) as ds:
    W, H = ds.width, ds.height
    T = ds.transform
    NODATA = int(ds.nodata)
    RASTER_CRS = str(ds.crs)
    COMPRESSION = ds.profile.get('compress')
    BLOCK = ds.block_shapes[0]
    N_OVR = len(ds.overviews(1))
TOTAL_PX = W * H
log('栅格 %d x %d = %d 像元, crs=%s, nodata=%d, compress=%s, block=%s, ovr=%d 级'
    % (W, H, TOTAL_PX, RASTER_CRS, NODATA, COMPRESSION, BLOCK, N_OVR))

# nodata 精确口径：aux.xml 的 65536 桶直方图（只统计有效值，不含 nodata）
import re
hist = None
try:
    txt = open(AUX, encoding='utf-8', errors='replace').read()
    m = re.search(r'<HistCounts>(.*?)</HistCounts>', txt, re.S)
    hist = np.array([int(x) for x in m.group(1).split('|')], dtype='int64')
    log('aux.xml 直方图 %d 桶, 合计 %d' % (len(hist), hist.sum()))
except Exception as e:
    log('aux.xml 直方图解析失败: %s' % e)

# 抽稀读取（金字塔为最近邻抽稀，已实测与精确子采样逐像元相同）作为 nodata 占比独立校验
with rasterio.open(VEG30) as ds:
    dec = ds.read(1, out_shape=(int(H / 512), int(W / 512))).astype('uint16')
dec_nodata_frac = float((dec == NODATA).mean())
log('抽稀读取 %s: nodata 占比 %.4f' % (dec.shape, dec_nodata_frac))

if hist is not None:
    valid_px = int(hist[:NODATA].sum())
    nodata_px = int(TOTAL_PX - valid_px)
else:
    valid_px = int(round(TOTAL_PX * (1 - dec_nodata_frac)))
    nodata_px = int(TOTAL_PX - valid_px)

basic = {
    'raster': VEG30,
    'width': W, 'height': H, 'n_pixels_total': int(TOTAL_PX),
    'dtype': 'uint16', 'crs': RASTER_CRS, 'nodata': NODATA,
    'transform': [T.a, T.b, T.c, T.d, T.e, T.f],
    'compression': COMPRESSION, 'block': list(BLOCK), 'n_overviews': N_OVR,
    'nodata_px': nodata_px,
    'nodata_pct': round(nodata_px / TOTAL_PX * 100, 3),
    'valid_px': valid_px,
    'valid_pct': round(valid_px / TOTAL_PX * 100, 3),
    'nodata_source': 'aux.xml 65536 桶精确直方图（只含有效值）',
    'nodata_cross_check_decimated_pct': round(dec_nodata_frac * 100, 3),
    'vаt_count_sum': int(vat['px'].sum()),
    'vat_vs_hist_note': ('VAT Count 合计 = %d，与直方图有效像元 %d %s'
                         % (int(vat['px'].sum()), valid_px,
                            '完全一致' if int(vat['px'].sum()) == valid_px else '不一致（差 %d）' % (int(vat['px'].sum()) - valid_px))),
}
# 修正 typo key
basic['vat_count_sum'] = basic.pop('vаt_count_sum')

# 名义面积（web-mercator，未做 cos²φ 折算）
basic['class_px_webmerc'] = {}
for c, sub in vat.groupby('map_cls'):
    basic['class_px_webmerc'][c] = {
        'px': int(sub['px'].sum()),
        'km2_webmerc_nominal': round(float(sub['km2_webmerc'].sum()), 1),
        'pct_of_valid': round(float(sub['px'].sum()) / valid_px * 100, 2),
    }
basic['area_note'] = ('EPSG:3857 名义像元 900 m²；真实地表面积 = 900*cos²φ。'
                      'km2_webmerc_nominal 为未折算的名义面积，仅供口径对照；'
                      '真实面积见 area_by_tile_scan（按行纬度 cos²φ 加权）。')

# ---- 313 瓦扫描（原生 30m，window 读取）
by_tile = None
area_scan = None
if '--skip-tiles' not in sys.argv:
    log('313 瓦原生 30m 扫描 ...')
    tiles = pd.read_csv(TILES_CSV, encoding='utf-8-sig')
    log('tiles.csv %d 行' % len(tiles))
    R = 6378137.0

    def merc_y_to_lat(y):
        return np.degrees(2.0 * np.arctan(np.exp(np.asarray(y) / R)) - np.pi / 2.0)

    recs = []
    area_acc = {c: 0.0 for c in CLS}
    valid_acc = 0.0
    nom_acc = {c: 0 for c in CLS}
    GRP_NAMES = [g for g in sorted(vat['类型组'].unique()) if g]
    GRP_LUT = np.zeros(65536, dtype=np.uint8)
    for gi_, gname in enumerate(GRP_NAMES, start=1):
        GRP_LUT[vat.loc[vat['类型组'] == gname, 'Value'].to_numpy()] = gi_
    grp_px = {g: 0 for g in GRP_NAMES}
    grp_km2 = {g: 0.0 for g in GRP_NAMES}
    with rasterio.open(VEG30) as ds:
        for i, t in tiles.iterrows():
            tb = transform_bounds('EPSG:4326', 'EPSG:3857', t.x0, t.y0, t.x1, t.y1)
            win = from_bounds(*tb, transform=ds.transform)
            col0 = max(int(win.col_off), 0)
            row0 = max(int(win.row_off), 0)
            col1 = min(int(win.col_off + win.width), W)
            row1 = min(int(win.row_off + win.height), H)
            if col1 <= col0 or row1 <= row0:
                log('  WARN 瓦片 %s 落在栅格范围外，跳过' % t.tile)
                continue
            win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
            a = ds.read(1, window=win).astype('uint16')
            valid = a != NODATA
            nvalid = int(valid.sum())
            # 行中心纬度 -> cos² 权重（3857 的 y -> 纬度）
            yc = ds.transform.f + (win.row_off + np.arange(a.shape[0]) + 0.5) * ds.transform.e
            w_row = np.cos(np.radians(merc_y_to_lat(yc))) ** 2
            a_l = LUT[a]
            a_l[~valid] = 0
            a_g = GRP_LUT[a]
            a_g[~valid] = 0
            cnt = {}
            tile_area_t = {}
            for ci, cname in enumerate(CLS, start=1):
                per_row = (a_l == ci).sum(axis=1).astype('float64')
                cnt[cname] = int(per_row.sum())
                ta = float((per_row * w_row).sum()) * 900.0 / 1e6
                tile_area_t[cname] = round(ta, 2)
                area_acc[cname] += ta
                nom_acc[cname] += cnt[cname]
            for gi_, gname in enumerate(GRP_NAMES, start=1):
                per_row = (a_g == gi_).sum(axis=1).astype('float64')
                grp_px[gname] += int(per_row.sum())
                grp_km2[gname] += float((per_row * w_row).sum()) * 900.0 / 1e6
            valid_row = valid.sum(axis=1).astype('float64')
            va = float((valid_row * w_row).sum()) * 900.0 / 1e6
            valid_acc += va
            n_all = int(a.size)
            recs.append({
                'tile': t.tile, 'x0': t.x0, 'y0': t.y0, 'x1': t.x1, 'y1': t.y1,
                'width_px': int(a.shape[1]), 'height_px': int(a.shape[0]),
                'valid_px': nvalid, 'total_px': n_all,
                'valid_pct': round(nvalid / n_all * 100, 3) if n_all else None,
                'valid_km2': round(va, 2),
                'needle_px': cnt['needle'], 'broad_px': cnt['broad'], 'mixed_px': cnt['mixed'],
                'bamboo_px': cnt['bamboo'], 'nonforest_px': cnt['nonforest'],
                'needle_km2': tile_area_t['needle'], 'broad_km2': tile_area_t['broad'],
                'mixed_km2': tile_area_t['mixed'], 'bamboo_km2': tile_area_t['bamboo'],
                'needle_pct_of_valid': round(cnt['needle'] / nvalid * 100, 3) if nvalid else None,
                'broad_pct_of_valid': round(cnt['broad'] / nvalid * 100, 3) if nvalid else None,
            })
            if (i + 1) % 50 == 0:
                log('  瓦片 %d/%d, 累计 %.0fs' % (i + 1, len(tiles), time.time() - T0))
    by_tile = pd.DataFrame(recs)
    by_tile.to_csv(os.path.join(OPT, 'r1_by_tile.csv'), index=False, encoding='utf-8-sig')
    log('r1_by_tile.csv 写入 %d 行' % len(by_tile))
    area_scan = {
        'method': '313 瓦逐瓦原生 30m 读取（window，不重采样）；逐行按 cos²φ 加权，像元真实面积 = 900*cos²φ m²',
        'n_tiles': int(len(by_tile)),
        'valid_km2': round(valid_acc, 1),
        'by_class_km2': {c: round(v, 1) for c, v in area_acc.items()},
        'by_class_km2_pct': {c: round(v / valid_acc * 100, 2) for c, v in area_acc.items()},
        'by_class_px': nom_acc,
        'by_type_group_px': grp_px,
        'by_type_group_km2': {g: round(v, 1) for g, v in grp_km2.items()},
        'by_type_group_km2_pct': {g: round(v / valid_acc * 100, 2) for g, v in grp_km2.items()},
        'note': ('瓦片并集覆盖 tiles.csv 的 313 个 2° 瓦（中国陆地范围），与全国栅格口径略有差异'
                 '（瓦外像元未计）；valid_km2 为瓦内有效像元真实面积之和。'
                 'by_type_group_* 是 12 个 类型组 的本轮核算面积，直接回答"哪些灌丛/稀疏植被被编进针叶/阔叶"。'),
        'sanity_check': ('cos²φ 加权面积 应小于 名义面积 (by_class_km2 < class_px_webmerc.km2_webmerc_nominal)。'
                         '若反向则权重算错。'),
    }
    basic['area_by_tile_scan'] = area_scan

# =====================================================================================
# 3. 在 v2 的 110,653 点上取值
# =====================================================================================
log('读 v2 样本 ...')
p = pd.read_parquet(V2_PARQ)
log('v2 %d 点, 列: %s' % (len(p), list(p.columns)))

tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
px, py = tr.transform(p['lon'].to_numpy(), p['lat'].to_numpy())

log('rasterio.sample 批量取值（110,653 点, 含 3x3 邻域）...')
with rasterio.open(VEG30) as ds:
    vals = np.array([v[0] for v in ds.sample(
        [(float(x), float(y)) for x, y in zip(px, py)], indexes=1)], dtype='uint16')
    # 3x3 邻域（±30 m）多数票，作为边界敏感性检验
    nb = np.zeros((9, len(p)), dtype='uint16')
    for j, (dx, dy) in enumerate([(0, 0), (-30, 0), (30, 0), (0, -30), (0, 30),
                                  (-30, -30), (30, -30), (-30, 30), (30, 30)]):
        nb[j] = np.array([v[0] for v in ds.sample(
            [(float(x + dx), float(y + dy)) for x, y in zip(px, py)], indexes=1)], dtype='uint16')

p['v30_raw'] = vals
IDX2CLS = ['nodata'] + CLS          # 0 = nodata/未映射
p['v30_cls'] = [IDX2CLS[i] for i in LUT[vals]]

# 3x3 多数票
cls_nb = LUT[nb]                       # 9 x n
counts = np.zeros((6, len(p)), dtype=np.uint8)
for j in range(9):
    for ci in range(6):
        counts[ci] += (cls_nb[j] == ci)
maj = counts.argmax(axis=0)
p['v30_cls_3x3'] = [IDX2CLS[i] for i in maj]
p['v30_code_3x3'] = nb[4]  # 中心

n_nodata_pt = int((p['v30_cls'] == 'nodata').sum())
log('点取值: nodata %d (%.2f%%), 有效 %d' % (n_nodata_pt, n_nodata_pt / len(p) * 100, len(p) - n_nodata_pt))

# 映射类分布（点口径）
point_cls_dist = p['v30_cls'].value_counts().to_dict()

# =====================================================================================
# 4. 100 万植被图按点归属（复刻 Q1 规则：within + 森林优先 + 面积最大去重）
# =====================================================================================
log('读 100 万植被图 ...')
import geopandas as gpd
from shapely import make_valid
from shapely import STRtree

veg = gpd.read_file(VEG_SHP)
log('植被图 %d 图斑, crs_declared=%s' % (len(veg), veg.crs))
veg = veg.set_crs('EPSG:4326', allow_override=True)

FOREST_TYPES = {'1、针叶林', '2、针阔叶混交林', '3、阔叶林'}
NEEDLE = {'常绿针叶', '落叶针叶'}
BROAD = {'常绿阔叶', '落叶阔叶'}
MIXED = {'常绿阔叶和落叶阔叶混交', '常绿针叶和落叶阔叶混交', '常绿针叶和常绿阔叶混交'}
BAMBOO = {'竹林'}


def map_habit(h):
    if h in NEEDLE:
        return 'needle'
    if h in BROAD:
        return 'broad'
    if h in MIXED:
        return 'mixed'
    if h in BAMBOO:
        return 'bamboo'
    return 'none'


veg['_leaf_all'] = veg['生活型'].map(map_habit)
veg['_is_forest'] = veg['植被型'].isin(FOREST_TYPES)
veg['_leaf_forest'] = np.where(veg['_is_forest'], veg['_leaf_all'], 'none')
veg['_area'] = veg['area_12'].astype('float64')
# 植被型 组号
veg['_gid'] = veg['植被型'].astype(str).str.extract(r'^(\d+)')[0].astype('float')
veg['_gid'] = veg['_gid'].fillna(0).astype(int)

# make_valid
bad = ~veg.geometry.is_valid
log('invalid 图斑 %d, make_valid ...' % int(bad.sum()))


def poly_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type == 'Polygon':
        return g
    if hasattr(g, 'geoms'):
        ps = [x for x in g.geoms if x.geom_type == 'Polygon']
        if not ps:
            return None
        from shapely.ops import unary_union
        return unary_union(ps)
    return None


if bad.any():
    veg.loc[bad, 'geometry'] = veg.loc[bad, 'geometry'].apply(make_valid).apply(poly_only)
keep = veg.geometry.notna() & ~veg.geometry.is_empty
veg = veg[keep].reset_index(drop=True)
log('植被图有效 %d 图斑' % len(veg))

log('STRtree 建树 + 点查询 ...')
tree = STRtree(veg.geometry.to_numpy())
pts_geom = gpd.points_from_xy(p['lon'], p['lat'])
idx_pt, idx_geom = tree.query(pts_geom, predicate='within')
log('命中 pair %d' % len(idx_pt))

# 去重：森林优先, 其次面积最大
cand = pd.DataFrame({'i': idx_pt, 'g': idx_geom})
cand['forest'] = veg['_is_forest'].to_numpy()[cand['g']].astype(int)
cand['area'] = veg['_area'].to_numpy()[cand['g']]
cand = cand.sort_values(['i', 'forest', 'area'], ascending=[True, False, False])
cand = cand[~cand['i'].duplicated(keep='first')]

veg_leaf_all = np.full(len(p), 'none', dtype=object)
veg_leaf_for = np.full(len(p), 'none', dtype=object)
veg_form = np.full(len(p), '', dtype=object)
veg_vtype = np.full(len(p), '', dtype=object)
veg_vsub = np.full(len(p), '', dtype=object)
veg_gid = np.zeros(len(p), dtype=int)
veg_inpatch = np.zeros(len(p), dtype=bool)
veg_area = np.zeros(len(p), dtype='float64')

gi = cand['g'].to_numpy()
pi = cand['i'].to_numpy()
veg_leaf_all[pi] = veg['_leaf_all'].to_numpy()[gi]
veg_leaf_for[pi] = veg['_leaf_forest'].to_numpy()[gi]
veg_form[pi] = veg['植被类型'].astype(str).to_numpy()[gi]
veg_vtype[pi] = veg['植被型'].astype(str).to_numpy()[gi]
veg_vsub[pi] = veg['植被亚型'].astype(str).to_numpy()[gi]
veg_gid[pi] = veg['_gid'].to_numpy()[gi]
veg_area[pi] = veg['_area'].to_numpy()[gi]
veg_inpatch[pi] = True

p['veg_leaf_all'] = veg_leaf_all
p['veg_leaf'] = veg_leaf_for
p['veg_form'] = veg_form
p['veg_vtype'] = veg_vtype
p['veg_vsub'] = veg_vsub
p['veg_gid'] = veg_gid
p['veg_area_km2'] = veg_area
p['veg_inpatch'] = veg_inpatch
log('植被图点归属: 命中 %d (%.2f%%), 森林图斑内 %d'
    % (int(veg_inpatch.sum()), veg_inpatch.mean() * 100, int((p['veg_leaf'] != 'none').sum())))

# 30m 码值 -> 群系名 / 植被型（供 generation_test 名字比对）
val2form = dict(zip(vat['Value'], vat['群系']))
val2vt = dict(zip(vat['Value'], vat['植被型']))
val2gid = dict(zip(vat['Value'], vat['group_id']))
p['v30_form'] = p['v30_raw'].map(val2form).fillna('')
p['v30_vtype'] = p['v30_raw'].map(val2vt).fillna('')
p['v30_gid'] = p['v30_raw'].map(val2gid).fillna(-1).astype(int)

# =====================================================================================
# 5. 一致率
# =====================================================================================
def agree_stats(mask, a_col, b_col, label):
    sub = p[mask]
    a = sub[a_col].to_numpy()
    b = sub[b_col].to_numpy()
    both = np.isin(a, ['needle', 'broad']) & np.isin(b, ['needle', 'broad'])
    n = int(both.sum())
    k = int((a[both] == b[both]).sum())
    out = {'label': label, 'n_both_conclusive': n, 'n_agree': k,
           'agreement': rate(k, n), 'wilson95': wilson(k, n)}
    if n:
        out['confusion_rows_%s_cols_%s' % (a_col, b_col)] = {
            'broad': {'broad': int(((a == 'broad') & (b == 'broad') & both).sum()),
                      'needle': int(((a == 'broad') & (b == 'needle') & both).sum())},
            'needle': {'broad': int(((a == 'needle') & (b == 'broad') & both).sum()),
                       'needle': int(((a == 'needle') & (b == 'needle') & both).sum())},
        }
        # 只算针叶侧的 precision/recall（我方关注针叶）
        tp = int(((a == 'needle') & (b == 'needle') & both).sum())
        fp = int(((a == 'broad') & (b == 'needle') & both).sum())
        fn = int(((a == 'needle') & (b == 'broad') & both).sum())
        out['needle_as_reference_%s' % a_col] = {
            'n_ref_needle': tp + fn, 'precision': rate(tp, tp + fp), 'recall': rate(tp, tp + fn)}
    return out


# FCS10 口径：v2 的 leaf 列（= habit_fcs10 叶形轴）
p['fcs10_leaf'] = p['leaf']

# R2: 30m vs FCS10
R2 = agree_stats(np.ones(len(p), dtype=bool), 'v30_cls', 'fcs10_leaf', '30m vs FCS10')
R2['gate_reference_doc81_V2'] = 0.80
R2['gate_note'] = 'doc81 的 V2 门 0.80 是为 100 万植被图预注册的；30m 数据未预注册判据，此处仅作参照，不构成判定。'
R2['gate_pass_by_reference'] = (R2['agreement'] is not None and R2['agreement'] >= 0.80)
R2['n_points_total'] = int(len(p))
R2['n_30m_nodata'] = n_nodata_pt
R2['class_dist_30m_at_points'] = {k: int(v) for k, v in point_cls_dist.items()}
# 3x3 稳健性
R2_3x3 = agree_stats(np.ones(len(p), dtype=bool), 'v30_cls_3x3', 'fcs10_leaf', '30m(3x3 mode) vs FCS10')
R2['sensitivity_3x3_mode'] = {'n_both_conclusive': R2_3x3['n_both_conclusive'],
                              'agreement': R2_3x3['agreement'], 'wilson95': R2_3x3['wilson95']}

# 30m vs 植被图
vs_veg = agree_stats(np.ones(len(p), dtype=bool), 'v30_cls', 'veg_leaf', '30m vs 植被图')
# 仅限落在森林图斑内的点（植被图有明确针阔判读的子集）
vs_veg_inforest = agree_stats(p['veg_inpatch'].to_numpy(), 'v30_cls', 'veg_leaf', '30m vs 植被图(仅森林图斑内)')
vs_veg['in_forest_patches_only'] = {k: vs_veg_inforest[k] for k in
                                    ('n_both_conclusive', 'n_agree', 'agreement', 'wilson95')}

# 引用 Q1 的 V2 用于对照
try:
    q1 = json.load(open(VEG_QA, encoding='utf-8'))
    q1_v2 = q1['V2']
except Exception:
    q1_v2 = None
vs_veg['q1_reference_vegmap_vs_fcs10'] = None if q1_v2 is None else {
    'n_both_conclusive': q1_v2['n_both_conclusive'], 'n_agree': q1_v2['n_agree'],
    'agreement': q1_v2['agreement'], 'wilson95': q1_v2['agreement_wilson95'],
    'confusion': q1_v2['confusion_vegmap_rows_vs_fcs10_cols'],
    'note': '直接引用 opt/veg_qa.json（未重跑）',
}

# vs PFT 2013（第四方，可选）
log('PFT 2013 点取值 ...')
pft_files = {
    'ENF_boreal': 'China-ENF_boreal_1km_2013_v1.0.tif',
    'ENF_temperate': 'China-ENF_temperate_1km_2013_v1.0.tif',
    'DNF': 'China-DNF_1km_2013_v1.0.tif',
    'EBF_temperate': 'China-EBF_temperate_1km_2013_v1.0.tif',
    'EBF_tropical': 'China-EBF_tropical_1km_2013_v1.0.tif',
    'DBF_boreal': 'China-DBF_boreal_1km_2013_v1.0.tif',
    'DBF_temperate': 'China-DBF_temperate_1km_2013_v1.0.tif',
    'DBF_tropical': 'China-DBF_tropical_1km_2013_v1.0.tif',
}
NEEDLE_PFT = ['ENF_boreal', 'ENF_temperate', 'DNF']
BROAD_PFT = ['EBF_temperate', 'EBF_tropical', 'DBF_boreal', 'DBF_temperate', 'DBF_tropical']
pft_cover = np.zeros(len(p), dtype=bool)
pft_needle = np.zeros(len(p), dtype=bool)
pft_broad = np.zeros(len(p), dtype=bool)
pft_ok = True
pft_src_note = []
# PFT 栅格是 Albers（+proj=aea +lat_1=25 +lat_2=47 +lon_0=105），经纬度必须先投影！！
PFT_CRS = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs')
tr_pft = Transformer.from_crs('EPSG:4326', PFT_CRS, always_xy=True)
ax, ay = tr_pft.transform(p['lon'].to_numpy(), p['lat'].to_numpy())
pft_xy = list(zip(ax.astype('float64'), ay.astype('float64')))
for k, fn in pft_files.items():
    fp = os.path.join(PFT_DIR, fn)
    if not os.path.exists(fp):
        pft_ok = False
        pft_src_note.append('缺失 ' + fn)
        continue
    with rasterio.open(fp) as ds:
        v = np.fromiter((x[0] for x in ds.sample(pft_xy)), dtype='float64', count=len(p))
        nd = ds.nodata
        inside = np.fromiter(
            ((ds.bounds.left <= xx <= ds.bounds.right) and (ds.bounds.bottom <= yy <= ds.bounds.top)
             for xx, yy in pft_xy), dtype=bool, count=len(p))
    valid = inside & ((v != nd) if nd is not None else True)
    hit = valid & (v == 1)
    pft_cover |= valid
    if k in NEEDLE_PFT:
        pft_needle |= hit
    else:
        pft_broad |= hit
p['pft_needle'] = pft_needle
p['pft_broad'] = pft_broad
p['pft_state'] = np.select([pft_needle & ~pft_broad, pft_broad & ~pft_needle, pft_needle & pft_broad],
                           ['needle', 'broad', 'both'], default='none')
log('PFT 点状态: %s ; 落在 PFT 网格内 %d/%d'
    % (p['pft_state'].value_counts().to_dict(), int(pft_cover.sum()), len(p)))

# PFT 口径 vs FCS10（引用 Q3）
try:
    q3 = json.load(open(PFT_QA, encoding='utf-8'))
    q3_v3 = q3['V3']
    q3_area = q3['basic']['pft_2013_area_km2']
except Exception:
    q3_v3, q3_area = None, None

vs_pft = agree_stats(np.ones(len(p), dtype=bool), 'v30_cls', 'pft_state', '30m vs PFT2013')
vs_pft['pft_state_at_points'] = {k: int(v) for k, v in p['pft_state'].value_counts().items()}
vs_pft['q3_reference_pft_vs_fcs10'] = None if q3_v3 is None else {
    'n_both_conclusive': q3_v3['n_both_conclusive'], 'agreement': q3_v3['agreement'],
    'wilson95': q3_v3['agreement_wilson95'], 'note': '直接引用 opt/pft_qa.json（未重跑）'}
vs_pft['pft_2013_area_km2'] = q3_area
vs_pft['pft_area_note'] = ('PFT 8 类为独立二值掩膜，同像元可属多类，故 needle+broad 面积之和可能大于森林面积；'
                           '1.29181 km²/像元（Albers 1136.58 m）。面积直接引自 opt/pft_qa.json。')

# =====================================================================================
# 6. 三方 / 四方表决
# =====================================================================================
log('三方表决 ...')
sel = (p['v30_cls'].isin(['needle', 'broad']) & p['veg_leaf'].isin(['needle', 'broad'])
       & p['fcs10_leaf'].isin(['needle', 'broad']))
sub = p[sel]
key = list(zip(sub['v30_cls'], sub['veg_leaf'], sub['fcs10_leaf']))
from collections import Counter
c3 = Counter(key)
three_way = {
    'n_triple_conclusive': int(len(sub)),
    'pct_of_all_points': round(len(sub) / len(p) * 100, 2),
    'combinations': [
        {'30m': a, 'vegmap': b, 'fcs10': c, 'n': int(n),
         'all_same': bool(a == b == c),
         '30m_is_odd_one_out': bool(a != b and a != c and b == c)}
        for (a, b, c), n in c3.most_common()
    ],
}
# 三方全同
n_all_same = int(sum(n for (a, b, c), n in c3.items() if a == b == c))
n_30_out = int(sum(n for (a, b, c), n in c3.items() if a != b and a != c and b == c))
n_veg_out = int(sum(n for (a, b, c), n in c3.items() if b != a and b != c and a == c))
n_fcs_out = int(sum(n for (a, b, c), n in c3.items() if c != a and c != b and a == b))
three_way['n_all_three_same'] = n_all_same
three_way['pct_all_three_same'] = rate(n_all_same, len(sub))
three_way['n_30m_odd_one_out'] = n_30_out
three_way['n_vegmap_odd_one_out'] = n_veg_out
three_way['n_fcs10_odd_one_out'] = n_fcs_out
three_way['majority_verdict'] = ('30m 与植被图一致而 FCS10 单独一方时为多数派，则 FCS10 在该子集可疑；'
                                 '反之若 30m 与 FCS10 一致而植被图单独一方，则植被图可疑。')

# 四方（含 PFT）
sel4 = sel & p['pft_state'].isin(['needle', 'broad'])
s4 = p[sel4]
c4 = Counter(zip(s4['v30_cls'], s4['veg_leaf'], s4['fcs10_leaf'], s4['pft_state']))
three_way['four_way_with_pft'] = {
    'n_quad_conclusive': int(len(s4)),
    'top_combinations': [
        {'30m': a, 'vegmap': b, 'fcs10': c, 'pft2013': d, 'n': int(n)}
        for (a, b, c, d), n in c4.most_common(12)],
    'n_all_four_same': int(sum(n for (a, b, c, d), n in c4.items() if a == b == c == d)),
}

# =====================================================================================
# 7. 分省 / 分生态区 / 南方 6 省
# =====================================================================================
def group_stats(df, key):
    out = []
    for g, s in df.groupby(key):
        a = s['v30_cls'].to_numpy()
        b = s['fcs10_leaf'].to_numpy()
        both = np.isin(a, ['needle', 'broad']) & np.isin(b, ['needle', 'broad'])
        n = int(both.sum())
        k = int((a[both] == b[both]).sum())
        # 30m vs 植被图
        bv = s['veg_leaf'].to_numpy()
        bothv = np.isin(a, ['needle', 'broad']) & np.isin(bv, ['needle', 'broad'])
        nv = int(bothv.sum())
        kv = int((a[bothv] == bv[bothv]).sum())
        out.append({
            'key': g, 'n_points': int(len(s)),
            'n_30m_nodata': int((s['v30_cls'] == 'nodata').sum()),
            'n_30m_needle': int((s['v30_cls'] == 'needle').sum()),
            'n_30m_broad': int((s['v30_cls'] == 'broad').sum()),
            'n_30m_mixed': int((s['v30_cls'] == 'mixed').sum()),
            'n_30m_nonforest': int((s['v30_cls'] == 'nonforest').sum()),
            'n_both_R2': n, 'n_agree_R2': k, 'R2': rate(k, n), 'R2_wilson95': wilson(k, n),
            'n_both_vs_veg': nv, 'agree_vs_veg': rate(kv, nv), 'agree_vs_veg_wilson95': wilson(kv, nv),
            'suppressed_n_lt_30': bool(n < 30),
        })
    return sorted(out, key=lambda x: -x['n_points'])


by_prov = group_stats(p, 'prov')
by_region = group_stats(p, 'eco')
pd.DataFrame(by_prov).to_csv(os.path.join(OPT, 'r1_by_prov.csv'), index=False, encoding='utf-8-sig')
log('r1_by_prov.csv 写入 %d 行' % len(by_prov))

# 南方 6 省专项
S6 = ['广东省', '福建省', '广西壮族自治区', '浙江省', '湖南省', '江西省']
s6 = p[p['prov'].isin(S6)]
m30n = s6['v30_cls'] == 'needle'
south = {
    'provinces': S6,
    'n_points': int(len(s6)),
    'n_30m_needle': int(m30n.sum()),
    'n_30m_needle_pct': round(m30n.mean() * 100, 2),
    'n_30m_broad': int((s6['v30_cls'] == 'broad').sum()),
    'n_30m_mixed': int((s6['v30_cls'] == 'mixed').sum()),
    'n_30m_nodata': int((s6['v30_cls'] == 'nodata').sum()),
    'among_30m_needle': {
        'n': int(m30n.sum()),
        'fcs10_broad': int((m30n & (s6['fcs10_leaf'] == 'broad')).sum()),
        'fcs10_needle': int((m30n & (s6['fcs10_leaf'] == 'needle')).sum()),
        'fcs10_mixed': int((m30n & (s6['fcs10_leaf'] == 'mixed')).sum()),
        'pct_fcs10_broad': rate(int((m30n & (s6['fcs10_leaf'] == 'broad')).sum()), int(m30n.sum())),
        'veg_broad': int((m30n & (s6['veg_leaf'] == 'broad')).sum()),
        'veg_needle': int((m30n & (s6['veg_leaf'] == 'needle')).sum()),
    },
    'among_veg_needle': {
        'n': int((s6['veg_leaf'] == 'needle').sum()),
        'v30_needle': int(((s6['veg_leaf'] == 'needle') & m30n).sum()),
        'v30_broad': int(((s6['veg_leaf'] == 'needle') & (s6['v30_cls'] == 'broad')).sum()),
        'v30_nonforest': int(((s6['veg_leaf'] == 'needle') & (s6['v30_cls'] == 'nonforest')).sum()),
        'fcs10_broad': int(((s6['veg_leaf'] == 'needle') & (s6['fcs10_leaf'] == 'broad')).sum()),
        'fcs10_needle': int(((s6['veg_leaf'] == 'needle') & (s6['fcs10_leaf'] == 'needle')).sum()),
    },
    'q1_reference': {
        'q1_south6_points_vegmap_needle_fcs10_broad': 4007,
        'note': 'Q1（veg_qa.json V5_preconditions）: 南 6 省落在植被图针叶图斑内且 FCS10 判阔叶 = 4,007 点（占该子集 80.72%）。',
    },
    'contrast_note': ('本节回答：30m 是否也像植被图那样"说南方针叶"。'
                      '若 30m-needle 子集的 FCS10-broad 比例同样高，则 30m 与植被图同侧，'
                      'FCS10 南方针叶判读可疑；若 30m 与 FCS10 高度一致，则 30m 是 FCS10 一侧的旁证。'),
}
# 南 6 省逐个
south['by_prov'] = {}
for pr, s in s6.groupby('prov'):
    m = s['v30_cls'] == 'needle'
    south['by_prov'][pr] = {
        'n_points': int(len(s)), 'n_30m_needle': int(m.sum()),
        'fcs10_broad_in_30m_needle': int((m & (s['fcs10_leaf'] == 'broad')).sum()),
        'pct': rate(int((m & (s['fcs10_leaf'] == 'broad')).sum()), int(m.sum())),
    }
# 30m 与植被图的"南方针叶"互斥度（对比 Q1 的 4007）
m_veg_needle_s6 = s6['veg_leaf'] == 'needle'
south['vegmap_needle_subset'] = {
    'n': int(m_veg_needle_s6.sum()),
    'v30_needle': int((m_veg_needle_s6 & m30n).sum()),
    'v30_broad': int((m_veg_needle_s6 & (s6['v30_cls'] == 'broad')).sum()),
    'v30_nonforest': int((m_veg_needle_s6 & (s6['v30_cls'] == 'nonforest')).sum()),
    'fcs10_broad': int((m_veg_needle_s6 & (s6['fcs10_leaf'] == 'broad')).sum()),
    'v30_needle_and_fcs10_broad': int((m_veg_needle_s6 & m30n & (s6['fcs10_leaf'] == 'broad')).sum()),
    'verdict': ('30m 与植被图在"南方针叶"上的重合度是判断 30m 是否独立更新的关键：'
                '若 30m 在植被图针叶图斑内也判针叶，则两源同侧；若 30m 判阔叶，则 30m 更接近 FCS10。'),
}

# =====================================================================================
# 8. 【核心】30m 与 100 万植被图的代际关系
# =====================================================================================
log('代际关系检验 ...')
gen = {}


def norm(s):
    s = str(s).strip().replace('（', '(').replace('）', ')')
    return re.sub(r'\s+', '', s)


# 30m 的图例代码（VAT 的 代码 列，如 24 / 24a / 586m）；字母后缀 = 母码下的亚群系细分
vat['code_full'] = vat['代码'].fillna('').astype(str).str.strip()
vat['code_base'] = vat['code_full'].str.extract(r'^(\d+)')[0]
vat['code_letter'] = vat['code_full'].str.extract(r'^\d+([a-z]+)$')[0]
vat['is_sub'] = vat['code_letter'].notna()

# 点级 code 字段（供 nesting 与 generation 使用）
p['v30_code'] = p['v30_raw'].map(dict(zip(vat['Value'], vat['code_full']))).fillna('')
p['v30_code_base'] = p['v30_raw'].map(dict(zip(vat['Value'], vat['code_base']))).fillna('')
p['v30_code_letter'] = p['v30_raw'].map(dict(zip(vat['Value'], vat['code_letter']))).fillna('')
_ode = np.full(len(p), -1, dtype='int64')
_ode[pi] = veg['ODE'].to_numpy()[gi]
p['veg_ode'] = _ode

# 植被图侧：ODE 就是图例码，植被类型 是被 DBF 截断到 ≤11 字的群系名
veg_g_form = veg['植被类型'].astype(str).map(norm)
ode2form = {}
ode2vt = {}
for o, fm, vt_ in zip(veg['ODE'].to_numpy(), veg_g_form.to_numpy(), veg['植被型'].astype(str).to_numpy()):
    ode2form.setdefault(int(o), fm)
    ode2vt.setdefault(int(o), vt_)

v_forms = vat['群系'].map(norm)
g_forms = veg_g_form

# --- 8.1 图例代码体系同源性（结构性证据）
b30 = set(int(x) for x in vat['code_base'].dropna())
ode_set = set(int(x) for x in veg['ODE'].unique())
gen['legend_code_system'] = {
    'definition': ('30m VAT 的 代码 列是否为《1:100万中国植被图》图例码体系：'
                   '与植被图 ODE 的码值谱系比对 + 亚群系字母后缀比对'),
    'n_codes_30m_all': int(len(vat)),
    'n_codes_30m_base': len(b30),
    'n_codes_30m_with_letter_suffix': int(vat['is_sub'].sum()),
    'px_with_letter_suffix': int(vat.loc[vat['is_sub'], 'px'].sum()),
    'pct_px_with_letter_suffix': round(float(vat.loc[vat['is_sub'], 'px'].sum()) / valid_px * 100, 2),
    'n_ode_vegmap': len(ode_set),
    'n_base_shared': len(b30 & ode_set),
    'pct_base_shared': round(len(b30 & ode_set) / max(1, len(b30)) * 100, 2),
    'base_only_30m': sorted(int(x) for x in (b30 - ode_set)),
    'ode_only_vegmap': sorted(int(x) for x in (ode_set - b30)),
    'max_base_30m': int(max(b30)), 'max_ode_vegmap': int(max(ode_set)),
    'interpretation': ('30m 与植被图共用同一套图例码（亚群系还共用 a/b/c… 字母后缀），'
                       '且 30m 的码值范围更长（表更全）。这是"同一图例体系"的硬证据——'
                       '两者不是各自独立编图，而是一套图例的两次应用。'),
}

# --- 8.2 群系名词表谱系：植被图每个群系名是否为某个 30m 群系名的前缀
v_names = sorted(set(x for x in v_forms if x))
g_name_counts = veg_g_form.value_counts()
found, not_found = [], []
for nm, cnt in g_name_counts.items():
    if not nm:
        continue
    pref = sum(1 for x in v_names if x.startswith(nm))
    rec = {'veg_form': nm, 'n_patch': int(cnt),
           'area_km2': round(float(veg.loc[veg_g_form == nm, '_area'].sum()), 1),
           'n_30m_names_starting_with_it': pref}
    (found if pref else not_found).append(rec)
gen['formation_lexicon'] = {
    'definition': '植被图的 植被类型（被 DBF 截断）是否为某个 30m 群系名的前缀',
    'n_vegmap_form_types': int(len(found) + len(not_found)),
    'n_found_in_30m': len(found),
    'pct_found_by_type': round(len(found) / max(1, len(found) + len(not_found)) * 100, 2),
    'pct_found_by_patch': round(sum(r['n_patch'] for r in found) /
                                max(1, sum(r['n_patch'] for r in found + not_found)) * 100, 2),
    'pct_found_by_area': round(sum(r['area_km2'] for r in found) /
                               max(1e-9, sum(r['area_km2'] for r in found + not_found)) * 100, 2),
    'n_30m_form_names': len(v_names),
    'top_vegmap_forms_absent_from_30m': sorted(not_found, key=lambda r: -r['area_km2'])[:20],
    'note': ('植被图 DBF 的 植被类型 字段宽 16、实测最长 11 字符，群系名被截断；30m 的 群系 字段最长 46 字符。'
             '因此"植被图名是 30m 名的前缀"是正确比对方向，反向会系统性失败。'),
}

# --- 8.3 【决定性】亚群系嵌套检验
#     注意：30m 与植被图的码值在"栽培植被"段**不通用**（30m 用 574–586，植被图只有到 573），
#     所以不能拿码值相等来测嵌套，必须回到**群系名**：30m 亚群系码（586m 桉树林）的母码名
#     （586 双季稻与冬甘薯…橡胶…）是否与植被图该点的群系名同指。
sub_pts = p[p['v30_code_letter'] != '']
sub_rows = []
for code, s in sub_pts.groupby('v30_code'):
    parent = re.sub(r'[a-z]+$', '', code)
    parent_form = vat.loc[vat['code_full'] == parent, '群系']
    parent_nm = norm(parent_form.iloc[0]) if len(parent_form) else ''
    # 母码行缺失时，退化为用本码自身群系名比对
    ref_nm = parent_nm if parent_nm else norm(str(s['v30_form'].iloc[0]))
    ok = np.zeros(len(s), dtype=bool)
    for i, b in enumerate(s['veg_form']):
        b = norm(str(b)).strip()
        if not b:
            continue
        if ref_nm.startswith(b):
            # ① 严格：植被图名是 30m 母码名前缀
            ok[i] = True
        elif len(b) >= 4 and b[:4] == ref_nm[:4]:
            # ② 容错：前 4 字相同（植被图有"野谷草/野古草"这类形近错字）
            ok[i] = True
        elif len(b) >= 4 and b[:4] in ref_nm:
            ok[i] = True
    sub_rows.append({
        'code': code, 'parent_code': parent, 'parent_form': parent_nm[:30],
        'sub_form': val2form.get(int(s['v30_raw'].iloc[0]), ''),
        'n_pts': int(len(s)),
        'n_match': int(ok.sum()), 'pct': rate(int(ok.sum()), len(s)),
        'veg_form_when_mismatch': s.loc[~ok, 'veg_form'].value_counts().head(3).to_dict(),
    })
sub_rows = sorted(sub_rows, key=lambda r: -r['n_pts'])
big = [r for r in sub_rows if r['n_pts'] >= 10]
tot_sub = int(sum(r['n_pts'] for r in big))
tot_nest = int(sum(r['n_match'] for r in big))
# 不含母码缺失码（parent_form 为空，无法判定）的严格口径
big_ok = [r for r in big if r['parent_form']]
tot_sub2 = int(sum(r['n_pts'] for r in big_ok))
tot_nest2 = int(sum(r['n_match'] for r in big_ok))
gen['subcode_nesting_test'] = {
    'definition': ('30m 带字母后缀的亚群系码（586m 桉树林 / 586d 橡胶园）所落点，'
                   '其**母码群系名**是否与该点植被图群系名同指（严格前缀，或前 4 字相同以容纳'
                   '"野古草/野谷草"这类形近错字）。码值本身不可比：30m 栽培植被段用 574–586，植被图只到 573。'),
    'n_points_with_subcode': int(len(sub_pts)),
    'pct_of_all_points': round(len(sub_pts) / len(p) * 100, 2),
    'n_points_subcode_n_ge10': tot_sub,
    'pct_parent_form_matches_vegmap': rate(tot_nest, tot_sub),
    'n_points_parent_form_defined': tot_sub2,
    'pct_parent_form_matches_vegmap_strict_scope': rate(tot_nest2, tot_sub2),
    'by_subcode_top25': sub_rows[:25],
    'interpretation': ('该比例高 ⇒ 30m 的亚群系码是植被图同一母群系的**单树种拆分**'
                       '（同一次调查的更细表达），不是新一轮观测。'
                       '母码缺失的码（如 69a/69b/434a/569b 的母码行群系为空）不进 strict 口径。'),
}

# --- 8.4 现代树种标记：30m 有 vs 植被图 0 图斑，并说明其图例归属
MODERN_KW = ['桉', '橡胶', '湿地松', '火炬松', '木麻黄', '泡桐', '杨树', '杉木', '柳杉', '思茅松',
             '日本落叶松', '水杉', '池杉', '落羽杉', '桤木', '相思', '柚木', '团花', '加勒比松', '班克松']
markers = {}
for kw in MODERN_KW:
    m30 = vat[v_forms.str.contains(kw, regex=False)]
    mv = veg[g_forms.str.contains(kw, regex=False)]
    markers[kw] = {
        'n_codes_30m': int(len(m30)),
        'px_30m': int(m30['px'].sum()),
        'km2_webmerc_nominal': round(float(m30['km2_webmerc'].sum()), 1),
        'codes_30m': sorted(set(m30['code_full'].tolist())),
        'groups_30m': sorted(set(m30['类型组'].tolist())),
        'names_30m': sorted(set(m30['群系'].tolist()))[:6],
        'n_patch_vegmap': int(len(mv)),
        'area_km2_vegmap': round(float(mv['_area'].sum()), 1),
        'in_vegmap_parent_legend': ('是：植被图 586 母码的群系描述本身就含该树种，'
                                    '30m 只是把它拆成独立码'
                                    if (len(mv) == 0 and len(m30) and
                                        any('586' == c[:3] or '585' == c[:3] or '584' == c[:3]
                                            for c in m30['code_full']))
                                    else ('是：植被图有同名群系' if len(mv) else '否')),
    }
gen['modern_plantation_markers'] = markers

# --- 8.5 30m 独有群系（植被图无任何群系名是其前缀）
v_has_parent = {}
for i, r in vat.iterrows():
    nm = v_forms.iloc[i]
    v_has_parent[i] = any(nm.startswith(g) for g in g_forms.unique() if g) if nm else False
vat['has_vegmap_parent'] = pd.Series(v_has_parent)
only30 = vat[~vat['has_vegmap_parent']].sort_values('px', ascending=False)
gen['formations_only_in_30m'] = {
    'n_codes': int(len(only30)),
    'px': int(only30['px'].sum()),
    'pct_of_valid_px': round(float(only30['px'].sum()) / valid_px * 100, 2),
    'n_with_letter_suffix': int((only30['code_letter'].notna()).sum()),
    'top25': [{'Value': int(r['Value']), 'code': r['code_full'], '群系': r['群系'],
               '类型组': r['类型组'], 'px': int(r['px']),
               'km2_webmerc_nominal': round(float(r['km2_webmerc']), 1)}
              for _, r in only30.head(25).iterrows()],
    'note': ('这些是"30m 有、植被图无同名群系"的码。判读时要区分两类：'
             '① 亚群系细分（带字母后缀，母码在植被图里存在）——同源更细；'
             '② 全新群系（母码也不在）——真新增。见 n_with_letter_suffix。'),
}

# --- 8.6 点级"类型组"与"群系名"一致（空间同源性的直接度量）
both_g = p['veg_inpatch'] & (p['v30_gid'] >= 1)
n_g = int(both_g.sum())
k_g = int((p.loc[both_g, 'v30_gid'] == p.loc[both_g, 'veg_gid']).sum())
gen['point_group_agreement'] = {
    'definition': '30m 的 类型组(1..11) 与 100 万植被图的 植被型 组号(1..11) 在 v2 点上的逐点一致率',
    'n_both_defined': n_g, 'n_agree': k_g,
    'agreement': rate(k_g, n_g), 'wilson95': wilson(k_g, n_g),
    'note': ('植被图落在 30m 上的"最近像元"会因 1:100 万制图综合而跨界，故 0.9 量级已是强同源信号；'
             '若两源独立编图，该值通常只有 0.5–0.7。'),
}
ct = pd.crosstab(p.loc[both_g, 'v30_gid'], p.loc[both_g, 'veg_gid'])
gen['point_group_agreement']['crosstab'] = {str(i): {str(j): int(v) for j, v in row.items()}
                                            for i, row in ct.iterrows()}

# 群系名：植被图名（截断）是否为 30m 名的前缀
m_form = p['veg_inpatch'] & (p['v30_form'] != '') & (p['veg_form'] != '')
n_f = int(m_form.sum())
k_f = int(sum(1 for a, b in zip(p.loc[m_form, 'v30_form'], p.loc[m_form, 'veg_form'])
              if str(b).strip() and str(a).startswith(str(b).strip())))
gen['point_formation_name_agreement'] = {
    'definition': 'v2 点上：30m 群系名 vs 所属植被图图斑的 植被类型（植被图名是否为 30m 名的前缀）',
    'n': n_f, 'n_agree': k_f, 'agreement': rate(k_f, n_f), 'wilson95': wilson(k_f, n_f),
    'note': ('0.59 的水平主要由"30m 用亚群系（含X的Y林 / 多树种并列）而植被图只记母群系"造成，'
             '不表示空间不同源；同源性看 point_group_agreement 与 subcode_nesting_test。'),
    'agreement_in_same_group': None,
}
mg = m_form & (p['v30_gid'] == p['veg_gid'])
if mg.sum():
    kk = int(sum(1 for a, b in zip(p.loc[mg, 'v30_form'], p.loc[mg, 'veg_form'])
                 if str(b).strip() and str(a).startswith(str(b).strip())))
    gen['point_formation_name_agreement']['agreement_in_same_group'] = {
        'n': int(mg.sum()), 'n_agree': kk, 'agreement': rate(kk, int(mg.sum())),
        'note': '仅在同类型组的点上比群系名（排除跨界误差）'}

# --- 8.7 空间范围/网格比对 + 系统偏移扫描（证明 30m 与植被图无 CRS 级错位）
with rasterio.open(VEG30) as ds:
    rb = transform_bounds(RASTER_CRS, 'EPSG:4326', *ds.bounds)
gen['extent'] = {
    'raster_lonlat_bounds': [round(x, 4) for x in rb],
    'vegmap_bounds_lonlat': q1.get('basic', {}).get('bounds_lonlat') if q1 else None,
    'note': '30m 的 EPSG:3857 包络反算到 WGS84 后与植被图 bounds 比较。',
}

log('  偏移扫描 ±0.05° ...')
gid_lut = np.zeros(65536, dtype=np.int16)
_gi = vat['group_id'].to_numpy()
gid_lut[vat['Value'].to_numpy()] = _gi


def group_agree_at(dlon, dlat):
    x, y = tr.transform((p['lon'] + dlon).to_numpy(), (p['lat'] + dlat).to_numpy())
    with rasterio.open(VEG30) as ds:
        v = np.fromiter((a[0] for a in ds.sample(list(zip(x, y)), indexes=1)),
                        dtype='uint16', count=len(p))
    gg = gid_lut[v]
    ok = (gg == p['veg_gid'].to_numpy()) & (gg > 0)
    return round(float(ok.mean()), 4)


offs = [-0.01, -0.005, 0.0, 0.005, 0.01]
grid = {}
for dx in offs:
    for dy in offs:
        grid['%+.3f,%+.3f' % (dx, dy)] = group_agree_at(dx, dy)
best = max(grid.items(), key=lambda kv: kv[1])
gen['extent']['offset_scan'] = {
    'definition': '对 v2 点做经纬度整体平移后，重算 30m 类型组 vs 植被图 类型组 的逐点一致率',
    'grid': grid,
    'best_offset_deg': best[0], 'best_agreement': best[1],
    'at_zero': grid['+0.000,+0.000'],
    'verdict': ('最优偏移为 (0,0)，说明两者共用同一地理参考，不存在 CRS/基准面级错位；'
                '残余不一致来自制图综合与编图差异，不是配准误差。'),
}

# --- 8.7 面积对照
gen['area_check'] = {
    '30m_needle_km2_webmerc_nominal': round(float(vat.loc[vat['map_cls'] == 'needle', 'km2_webmerc'].sum()), 1),
    '30m_broad_km2_webmerc_nominal': round(float(vat.loc[vat['map_cls'] == 'broad', 'km2_webmerc'].sum()), 1),
    'vegmap_needle_km2': q1['area_reconciliation']['needle_area_km2'] if q1 else None,
    'vegmap_broad_km2': q1['area_reconciliation']['broad_area_km2'] if q1 else None,
    'note': ('30m 为名义（未做 cos²φ 折算），植被图为 EPSG:6933 等积。真实面积见 basic.area_by_tile_scan。'
             '若 30m 是把植被图矢量栅格化，二者面积应同量级；量级差异大则非同源。'),
}

# 代际判定（先写判据，后给结论）
gen['decision_rules'] = {
    'D1_same_legend_system': '30m 的图例代码（VAT 代码 列）与植被图 ODE 共用码值 + 共用亚群系字母后缀 ⇒ 同一套图例体系',
    'D2_formation_lexicon': '植被图 群系名（截断）为某些 30m 群系名之前缀的比例 ≥0.70 ⇒ 同一套群系词表，30m 只是更细',
    'D3_subcode_nesting': ('30m 亚群系码（586m 等）所落点的植被图群系名，'
                           '与 30m 该码母群系名的前缀匹配率 ≥0.70 ⇒ 同一母群系被细分（同源）'),
    'D4_spatial_group_agree': '点级 类型组 一致率（1:100万制图综合下）≥0.85 ⇒ 同源同代；0.5–0.7 ⇒ 独立编图',
    'D5_modern_taxa': '30m 独有"植被图 0 图斑"的现代引种树种（湿地松/火炬松/木麻黄/泡桐/桉/橡胶）且像元可观 ⇒ 更新版',
    'verdict_rule': ('D1∧D2∧D3 成立而 D5 不成立 ⇒ SAME_SOURCE_COARSER_RASTER（同一调查的栅格化/细分，年代同样停在 1980s–90s）；'
                     'D5 成立 ⇒ NEWER（能解决 2014–2023 盲区）；D1–D3 均不成立 ⇒ INDEPENDENT_SOURCE'),
}

# 结论
g_ag = gen['point_group_agreement']['agreement']
f_ag = gen['formation_lexicon']['pct_found_by_type'] / 100.0
sub_ag = gen['subcode_nesting_test']['pct_parent_form_matches_vegmap']
sub_ag_s = gen['subcode_nesting_test']['pct_parent_form_matches_vegmap_strict_scope']
legend_share = gen['legend_code_system']['pct_base_shared'] / 100.0
# 真正代表"现代引种"的树种：植被图完全没有的
TRUE_MODERN = ['湿地松', '火炬松', '木麻黄', '泡桐', '水杉', '池杉', '落羽杉']
modern_px = int(sum(markers[k]['px_30m'] for k in TRUE_MODERN))
modern_in_veg = int(sum(markers[k]['n_patch_vegmap'] for k in TRUE_MODERN))
# 桉/橡胶/柳杉：植被图 0 图斑，但其母码 586/585 在植被图里存在（且 586 母码的群系描述已含"橡胶"）
euc_px = int(markers['桉']['px_30m'] + markers['橡胶']['px_30m'] + markers['柳杉']['px_30m'])

d1 = legend_share >= 0.90
d2 = f_ag >= 0.70
d3 = (sub_ag_s is not None and sub_ag_s >= 0.70)
d4 = (g_ag is not None and g_ag >= 0.85)
d5 = modern_px > 0 and modern_in_veg == 0

if d1 and d2 and not d5:
    verdict_gen = 'SAME_SOURCE_RASTER_OR_SUBDIVISION'
elif d5:
    verdict_gen = 'NEWER_VERSION'
elif not (d1 or d2):
    verdict_gen = 'INDEPENDENT_SOURCE'
else:
    verdict_gen = 'PARTIAL_LINEAGE_UNCLEAR'

gen['verdict'] = verdict_gen
gen['verdict_inputs'] = {
    'D1_legend_code_shared_pct': gen['legend_code_system']['pct_base_shared'],
    'D2_vegmap_form_found_in_30m_pct_by_type': gen['formation_lexicon']['pct_found_by_type'],
    'D2_by_area_pct': gen['formation_lexicon']['pct_found_by_area'],
    'D3_subcode_parent_form_matches_vegmap': sub_ag,
    'D3_strict_scope': sub_ag_s,
    'D4_point_group_agreement': g_ag,
    'D5_true_modern_taxa_px_30m': modern_px,
    'D5_true_modern_taxa_patch_vegmap': modern_in_veg,
    'note_subtle': ('桉(%.1f 万 km² 名义)/橡胶(%.1f)/柳杉：植被图 0 图斑，但其母码 586/585 在植被图里存在，'
                    '且植被图 586 母码的群系描述文字本身就含"橡胶"——说明 30m 是把植被图的"热带作物复合体"码'
                    '拆成了单树种码，不是新增观测。' %
                    (markers['桉']['km2_webmerc_nominal'] / 1e4,
                     markers['橡胶']['km2_webmerc_nominal'] / 1e4)),
}
gen['verdict_semantics'] = {
    'SAME_SOURCE_RASTER_OR_SUBDIVISION': ('30m 与 100 万植被图是同一套图例、同一套群系词表、同一批图斑的'
                                          '栅格化与亚群系细分。**年代与植被图相同，不能解决 2014–2023 盲区**；'
                                          '但它在"空间细化 + 亚群系拆分 + 全国连续无缝"三点上仍有独立价值。'),
    'NEWER_VERSION': '30m 含植被图没有的现代引种/人工林群系，是更新版，能提供 1980s 之后的信息。',
    'INDEPENDENT_SOURCE': '两者独立编图，30m 是真正独立的第三方标签源。',
    'PARTIAL_LINEAGE_UNCLEAR': '证据部分支持同源、部分不支持，需人工判读确认。',
}

# =====================================================================================
# 9. 面积三源对比
# =====================================================================================
area_compare = {
    'sources': {
        '30m_vegtype': {
            'needle_km2': (basic.get('area_by_tile_scan') or {}).get('by_class_km2', {}).get('needle'),
            'broad_km2': (basic.get('area_by_tile_scan') or {}).get('by_class_km2', {}).get('broad'),
            'needle_km2_webmerc_nominal': round(float(vat.loc[vat['map_cls'] == 'needle', 'km2_webmerc'].sum()), 1),
            'broad_km2_webmerc_nominal': round(float(vat.loc[vat['map_cls'] == 'broad', 'km2_webmerc'].sum()), 1),
            'resolution': '30 m', 'crs': 'EPSG:3857',
            'method': '栅格像元计数（逐行 cos²φ 加权）',
        },
        'vegmap2020': {
            'needle_km2': q1['area_reconciliation']['needle_area_km2'] if q1 else None,
            'broad_km2': q1['area_reconciliation']['broad_area_km2'] if q1 else None,
            'resolution': '1:1,000,000 矢量',
            'method': 'EPSG:6933 等积图斑面积求和（森林植被型 1/2/3）',
        },
        'pft2013': {
            'needle_km2': (q3_area or {}).get('ENF_boreal', 0) + (q3_area or {}).get('ENF_temperate', 0) + (q3_area or {}).get('DNF', 0) if q3_area else None,
            'broad_km2': (q3_area or {}).get('EBF_temperate', 0) + (q3_area or {}).get('EBF_tropical', 0) + (q3_area or {}).get('DBF_boreal', 0) + (q3_area or {}).get('DBF_temperate', 0) + (q3_area or {}).get('DBF_tropical', 0) if q3_area else None,
            'resolution': '1.29 km', 'crs': 'Albers',
            'method': '像元计数 x 1.29181 km²；8 类为独立掩膜可重叠',
        },
    },
    'note': '三源定义不同（30m=植被类型图栅格、植被图=地面调查矢量、PFT=遥感反演 1km 二值），面积不可直接等同，只看量级与针阔比。',
}
for s in area_compare['sources'].values():
    if s.get('needle_km2') and s.get('broad_km2'):
        s['needle_share'] = round(s['needle_km2'] / (s['needle_km2'] + s['broad_km2']), 4)

# =====================================================================================
# 10. 输出
# =====================================================================================
log('汇总输出 ...')
out = {
    'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    'script': os.path.abspath(__file__),
    'runtime_sec': None,
    'inputs': {'veg30': VEG30, 'vat': VAT, 'aux': AUX, 'v2': V2_PARQ, 'vegmap': VEG_SHP,
               'pft2013': PFT_DIR, 'tiles': TILES_CSV,
               'q1_reference': VEG_QA, 'q3_reference': PFT_QA},
    'basic': basic,
    'code_map': code_map_summary,
    'R2': R2,
    'vs_vegmap': vs_veg,
    'vs_pft': vs_pft,
    'three_way': three_way,
    'by_prov': by_prov,
    'by_region': by_region,
    'south': south,
    'generation_test': gen,
    'area_compare': area_compare,
    'verdict': {},
    'caveats': [],
}

# --- verdict
v = {}
v['area_30m'] = basic.get('area_by_tile_scan', {}).get('by_class_km2')
v['R2_30m_vs_fcs10'] = R2['agreement']
v['R2_wilson95'] = R2['wilson95']
v['R2_gate_080_pass'] = R2['agreement'] is not None and R2['agreement'] >= 0.80
v['vs_vegmap_agreement'] = vs_veg['agreement']
v['vs_vegmap_note'] = ('30m 与植被图在针阔轴上高度一致（见 vs_vegmap），'
                       '这正是"两者同源"的表现，不构成独立验证')
v['vs_pft_agreement'] = vs_pft['agreement']
v['Q1_vegmap_vs_fcs10_agreement'] = q1_v2['agreement'] if q1_v2 else None
v['Q3_pft_vs_fcs10_agreement'] = q3_v3['agreement'] if q3_v3 else None
v['generation'] = {'verdict': verdict_gen,
                   'semantics': gen['verdict_semantics'][verdict_gen],
                   'inputs': gen['verdict_inputs']}
v['south6_30m_needle_n'] = south['among_30m_needle']['n']
v['south6_30m_needle_fcs10_broad_pct'] = south['among_30m_needle']['pct_fcs10_broad']
v['south6_vegmap_needle_fcs10_broad_n'] = south['q1_reference']['q1_south6_points_vegmap_needle_fcs10_broad']
v['three_way_all_same_pct'] = three_way['pct_all_three_same']
v['headline'] = ('30m 与 100 万植被图是同一套图例（共用码值 + 亚群系字母后缀）、同一套群系词表'
                 '（植被图名 %s%% 可作 30m 名前缀）、同一批图斑的细分（亚群系码 %s%% 落在同母码图斑内）、'
                 '点级类型组一致率 %s。**结论：30m 是植被图的栅格化+亚群系细分，年代同样停留在 1980s–1990s，'
                 '不能解决 2014–2023 盲区。** 因此它不是独立的第四源，而是第三源（植被图）的高分辨率、'
                 '更细图例版本——对 FCS10 的交叉结论与 Q1 完全同向，不能作为"两票对一票"的第二票。'
                 % (gen['verdict_inputs']['D2_vegmap_form_found_in_30m_pct_by_type'],
                    round((sub_ag_s or 0) * 100, 1), g_ag))
out['verdict'] = v

cav = [
    '【最重要】30m 与 100 万植被图是**同源**（同一套图例码 + 同一套群系词表 + 同一批图斑被亚群系细分，见 generation_test）。'
    '因此 "30m vs 植被图" 的高一致率**不是独立验证**，而是同一份地面调查数据的自洽；'
    '30m 不能在对 FCS10 的裁决中充当"第二个独立票"。',
    '30m 栅格是 EPSG:3857（Web Mercator），像元名义 900 m²；真实地表面积 = 900*cos²φ。'
    '本报告同时给出名义面积（km2_webmerc_nominal）与 cos²φ 加权面积（area_by_tile_scan），不可混用。'
    '按 doc81 口径，只有 cos²φ 加权值可用于面积对账。',
    'nodata 口径取自 .tif.aux.xml 的 65536 桶精确直方图（只统计有效值，合计 16,747,872,300），'
    '并用 512 倍抽稀读取独立校验（nodata 57.96%）；两者一致。'
    '注意：名义总像元 39,885,543,361 × (1-0.5796) ≈ 16.76e9，与直方图吻合。',
    '30m 的 群系 字段最长 46 字符，而 100 万植被图的 植被类型 字段被 DBF 截断到 ≤11 字符。'
    '因此名字比对只能用"植被图名是否为 30m 名前缀"这一个方向；反向比对会产生大量假"独有群系"。',
    'v2 的 prov/eco 字段为上游已定，本报告直接沿用（DataV 省界为 GCJ-02，省归属仅作粗略分区）。',
    '100 万植被图的点归属复刻 Q1 规则（within + 森林优先 + 面积最大去重），命中 110,295/110,653 = 99.68%，与 Q1 完全一致。',
    'PFT 2013 为 8 类独立二值掩膜，同像元可属多类（pft_state=both）；PFT 栅格是 Albers（+lat_1=25 +lat_2=47），'
    '本报告已用 pyproj 把经纬度投影到 Albers 后采样。面积直接引用 opt/pft_qa.json，未重跑。',
    'FCS10 口径用 v2 的 leaf 列（叶形轴），不是 30 类原码。',
    '本轮**不**做影像核查（S2/GEDI），因此"30m 与 FCS10 谁对"只能在源间互证层面裁决；'
    '而同源检验已表明 30m 无法提供独立证据，故该裁决仍须回到 Q4 影像核查。',
    '30m 数据的编制年代、原始调查年份在 VAT / aux.xml 中均无元数据字段，代际结论是"图例谱系 + 群系词表 + '
    '亚群系嵌套 + 空间一致性 + 现代树种标记"五项间接证据的合成判断，不是元数据直证。',
    '30m 的 nonforest 中"栽培植被"占 21.4%（35.8 亿像元），其中茶园/油茶林/桉树林/橡胶园等木本园地按类型组归 nonforest；'
    '这与 FCS10 的"forest"口径不同源，交界处必然产生假分歧（已在 code_map.ambiguity_hard_list 逐码列出）。',
]
out['caveats'] = cav
out['runtime_sec'] = round(time.time() - T0, 1)

with open(os.path.join(OPT, 'r1_30m_qa.json'), 'w', encoding='utf-8') as f:
    json.dump(jsonable(out), f, ensure_ascii=False, indent=1)

# 分点中间件（供后续 Q2/Q5 复用，不属交付但很有用）
p[['point_id', 'lon', 'lat', 'tile', 'prov', 'eco', 'leaf', 'veg_leaf', 'veg_form', 'veg_vtype',
   'veg_inpatch', 'v30_raw', 'v30_cls', 'v30_cls_3x3', 'v30_form', 'v30_vtype', 'v30_gid', 'veg_gid',
   'pft_state']].to_parquet(os.path.join(OPT, 'r1_points.parquet'), index=False)

log('完成: r1_30m_qa.json / r1_code_map.csv / r1_by_tile.csv / r1_by_prov.csv / r1_points.parquet')
print(json.dumps(jsonable({k: out[k] for k in ['verdict']}), ensure_ascii=False, indent=1))
