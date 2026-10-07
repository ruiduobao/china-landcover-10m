# -*- coding: utf-8 -*-
"""
lc_conf.py — 本项目的单一事实来源（分类体系 / 码表映射 / 类别分组）
* 所有本地脚本 import 此文件，保证码表、Level 归并、类别名一致，避免多处维护漂移。
* 对齐 02文档 §4.1/§4.2 与 03文档 §6.4。
"""
import os

# ---------- 30 类精细体系：code -> (ename, cname, level1_group, level0_group) ----------
# level1: 10类基础体系；level0: 9类（与CLCD/WorldCover可比）
CLASSES = {
    10: ('Herbaceous rainfed cropland', '草本旱地', 'crop', 'crop'),
    11: ('Tree/shrub cropland (orchard)', '乔灌园地', 'crop', 'crop'),
    12: ('Irrigated cropland', '灌溉耕地(含水田)', 'crop', 'crop'),
    51: ('Closed evergreen broadleaved forest', '郁闭常绿阔叶林', 'forest', 'forest'),
    52: ('Open evergreen broadleaved forest', '疏闭常绿阔叶林', 'forest', 'forest'),
    61: ('Closed deciduous broadleaved forest', '郁闭落叶阔叶林', 'forest', 'forest'),
    62: ('Open deciduous broadleaved forest', '疏闭落叶阔叶林', 'forest', 'forest'),
    71: ('Closed evergreen needleleaved forest', '郁闭常绿针叶林', 'forest', 'forest'),
    72: ('Open evergreen needleleaved forest', '疏闭常绿针叶林', 'forest', 'forest'),
    81: ('Closed deciduous needleleaved forest', '郁闭落叶针叶林', 'forest', 'forest'),
    82: ('Open deciduous needleleaved forest', '疏闭落叶针叶林', 'forest', 'forest'),
    91: ('Closed mixed-leaf forest', '郁闭针阔混交林', 'forest', 'forest'),
    92: ('Open mixed-leaf forest', '疏闭针阔混交林', 'forest', 'forest'),
    120: ('Evergreen shrubland', '常绿灌丛', 'shrub', 'shrub'),
    121: ('Deciduous shrubland', '落叶灌丛', 'shrub', 'shrub'),
    130: ('Grassland', '草地', 'grass', 'grass'),
    140: ('Lichens and mosses', '地衣苔藓', 'bare', 'bare'),
    150: ('Sparse vegetation', '稀疏植被', 'sparse', 'bare'),
    180: ('Woody swamp', '木本沼泽', 'wetland', 'wetland'),
    181: ('Herbaceous marsh', '草本沼泽', 'wetland', 'wetland'),
    182: ('Lake/river flat', '湖河滩地', 'wetland', 'wetland'),
    183: ('Saline wetland', '盐渍湿地', 'wetland', 'wetland'),
    184: ('Mangrove forest', '红树林', 'wetland', 'wetland'),
    185: ('Salt marsh', '海岸盐沼', 'wetland', 'wetland'),
    186: ('Tidal flat', '潮滩', 'wetland', 'wetland'),
    190: ('Urban impervious', '城镇不透水面', 'imperv', 'imperv'),
    200: ('Rural impervious', '乡村不透水面', 'imperv', 'imperv'),
    201: ('Bare areas', '裸地', 'bare', 'bare'),
    202: ('Water body', '水体', 'water', 'water'),
    220: ('Permanent ice and snow', '冰雪', 'snow', 'snow'),
}

# level1 组编码
LEVEL1 = {'crop':1,'forest':2,'shrub':3,'grass':4,'wetland':5,'imperv':6,
          'bare':7,'sparse':8,'water':9,'snow':10}
LEVEL0 = {'crop':1,'forest':2,'shrub':3,'grass':4,'wetland':5,
          'imperv':6,'bare':7,'water':9,'snow':10}

# ============ GLC_FCS30D 原始码 → 本产品30类 映射 ============
# 依据：GLC_FCS30D 官方图例（gee-community-catalog 核实，2026-09-03）
#  + 项目内 2.0数据生成代码.js 的 names 数组（sat-io 官方），两者一致。
# GLC 输入码含义（→ 产品码）：
#   10 Rainfed cropland →10   11 Herbaceous cropland →10   12 Tree/shrub(orchard) →11   20 Irrigated →12
#   森林 51..92：奇=Open 偶=Closed（与产品相反）→ 交叉映射
#   灌丛 120=Shrubland 121=Evergreen 122=Deciduous；产品 120=常绿 121=落叶
#   湿地 181=Swamp(木本) 182=Marsh(草本) 183=Flooded flat 184=Saline 185=Mangrove 186=Salt marsh 187=Tidal flat
#        产品 180=木本 181=草本 182=湖河滩 183=盐渍 184=红树林 185=盐沼 186=潮滩（-1 平移）
#   190=Impervious(不分城乡→产品190城镇) 200=Bare 201=Consol 202=Unconsol(→产品201裸地) 210=Water→202 220=Ice→220
CODE_MAP = {
    10:10, 11:10, 12:11, 20:12,
    51:52, 52:51, 61:62, 62:61, 71:72, 72:71, 81:82, 82:81,
    91:92, 92:91,                       # GLC 91=Open mixed→产品92(Open)；92=Closed→91
    120:121, 121:120, 122:121,          # GLC 一般灌丛→产品落叶(121)；Evergreen→120；Deciduous→121
    130:130, 140:140, 150:150, 152:150, 153:150,
    181:180, 182:181, 183:182, 184:183, 185:184, 186:185, 187:186,
    190:190, 200:201, 201:201, 202:201, 210:202, 220:220,
}
# 说明：产品 200(乡村不透水) 在 GLC_FCS30D 中无直接来源（其190不分城乡），
#       需用建成区边界/夜间灯光把产品190(城镇不透水)分割出 200 —— 真实的外部补数据需求。

# **缺口类**：GLC_FCS30D 无法提供种子的类（本地统计实据，见 数据/本地处理/统计）
GAP_CLASSES = {92: '疏闭针阔混交林（用混交林/郁闭度专题产品补充）',
               180: '木本沼泽（用GWL/湿地专题产品补充）'}

TARGET_FEATURES = [f'A{i:02d}' for i in range(64)]   # AlphaEarth 64维嵌入波段名
PROJ_DIR = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
OUT = os.path.join(PROJ_DIR, '数据', '本地处理')
CODE_DIR = os.path.join(PROJ_DIR, '代码', '0.本地流水线')

def class_name(code):
    return CLASSES.get(code, ('?', '未知', '?', '?'))[1]

