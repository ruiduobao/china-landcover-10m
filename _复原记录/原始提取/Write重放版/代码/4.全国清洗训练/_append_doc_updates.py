# -*- coding: utf-8 -*-
"""一次性：给技术文档追加统一的"更新记录"章节（2026-09-09 r7 收尾 + 交付升级）"""
import io, os

DOCS = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据/技术文档'

CORE = """- **r7 年度嵌入提取完成**：4,758 块 / 14,272,711 point-years（2017–2024），12 账号编队、3,000 点/块 + 下载重试。
- **年度 QC 与八套年度子集**：`sample_year_qc.parquet`（keep 13,475,352 / downweight 570,893 / exclude_year 216,274 / persistent_change 10,192）+ `年度子集/r7_train_{2017..2024}.parquet`。
- **稀有类补样**：91/140/183/184 各补到 2,000 点（+992 / +1,737 / +1,087 / +861），全库失衡比 ≈1,700× → ≈223×。
- **空间独立评估**：OA 0.7923 / macro-F1 0.6063 / balanced accuracy 0.6036（剔除 367 个 holdout 块后）。
- **交付格式升级**：`数据/样本交付/gpkg/`（22 个 GeoPackage，保留全部属性，Point EPSG:4326）+ `数据/样本交付/分布图/`（30 类单类分布图 + 6×5 面板 + 总览图）。
- 详见 `18_V1版本的样本审稿意见落实与提升报告.md` 与 `19_样本交付与文档升级记录.md`。"""

NOTE = {
 '03_高精度样本构建方法论.md':
   '本文关于"标签时间有效期 / 逐年适用性"的方法设想，已落地为 `r7_train_validity.parquet` 的 `valid_from/valid_to` 与 `sample_year_qc.parquet` 的四态判定。',
 '10_全国生产Runbook_按试点实测定型.md':
   'Runbook 中"全国留出集"评估（原 n8_validate）已升级为 `validation_pool_v2.parquet`（25km Albers 块整块留出、块内侵蚀 5km），并由 `e6_spatial_eval.py` 出分年份/分生态区报告。',
 '11_样本数据源扩展与现成样本库清单.md':
   '按本文既有惯例（by_class PNG + GPKG），r7 交付物已生成 22 个 GPKG 与 30 类分布图；稀有类新增源：GLC_FCS10 2023（raw 92/140/184）、GMW v3 2020、ChinaWetlands CW_2020（value=2 红树林）。',
 '12_全国样本从零重建r1.md':
   '§遗留表中第 4 条"锚年标签偏差（最重要）"已由 r7 八套年度子集正面处理；第 6 条"91/140 全国本底极稀"已补到 2,000 点/类。',
 '14_完整技术流程总集.md':
   '状态表更新：r1 与 r7 样本库均已交付；新增八套年度子集、22 个 GPKG、30 类分布图；独立评估报告见 `评估_P3.4/`。',
 '15_Agent复现手册.md':
   '新增脚本：`e2b_yearly_qc.py`（年度QC/年度子集）、`e6_spatial_eval.py`（空间独立评估）、`e7_merge_rare.py`（稀有类合并）、`e8_export_gpkg.py`（GPKG 导出）、`e9_class_maps.py`（分布图）、`e1c_worker_generic.py`（通用提取 worker）、`r8/r8b/r9`（稀有类补样）。',
 '16_样本库r2提质变更记录.md':
   '§8.5 遗留第 1 条"从 FCS10 重采 92/140/180/183/185"本轮部分实施：按 `class_balance_design.json` 缺口清单完成 **91（raw 92）/140/183/184** 四类补样并补齐嵌入；92/180/185 不在本轮缺口内，未动。',
 '17_样本年度适用性改造方案.md':
   '§六状态跟踪更新：P2 提取 ✅ 完成（4,758 块）；P3.1/P3.2 年度QC与八套子集 ✅；P3.3 稀有类补样 ✅；P3.4 空间独立评估 ✅（Olofsson 面积区间待制图后补算）；P3.5（190/200 两阶段）未做。',
 '18_V1版本的样本审稿意见落实与提升报告.md':
   '本轮追加交付：`数据/样本交付/gpkg/`（22 个 GPKG）与 `数据/样本交付/分布图/`（30 类分布图），格式与清单见 `19_样本交付与文档升级记录.md`。',
}


def main():
    for fn, note in NOTE.items():
        p = os.path.join(DOCS, fn)
        if not os.path.isfile(p):
            print('MISS', fn); continue
        s = io.open(p, encoding='utf-8').read()
        if '## 更新记录（2026-09-09' in s:
            print('SKIP(已有更新记录)', fn); continue
        block = ('\n\n---\n\n## 更新记录（2026-09-09｜r7 收尾 + 交付格式升级）\n'
                 + note + '\n\n' + CORE + '\n')
        io.open(p, 'w', encoding='utf-8', newline='').write(s.rstrip() + block)
        print('APPEND', fn)


if __name__ == '__main__':
    main()
