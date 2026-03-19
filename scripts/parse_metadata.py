"""
阶段 2：标签解析与元数据标准化
=============================
parse_metadata.py - 将文件夹命名规则转化为结构化标签表。

命名规则：
  M / m = MBA (4-Mercaptobenzoic acid)
  美 = 福美双（Thiram）
  孔 / K = 孔雀石绿（Malachite Green, MG）
  银 = AgNPs
  0.4 / 0.5 / 0.6 → 4 / 5 / 6 ppm

文件夹命名模式（共50个）：
  1. 单物质+银基底:
     - mba+银+0.X       → MBA only
     - 孔+银+0.X        → MG only
     - 福美双+银+0.X    → Thiram only
  2. 二元 MBA+MG:
     - 孔XmY / 孔XMLy   → MG=X*10ppm, MBA=Y*10ppm
  3. 二元 Thiram+MBA:
     - 美XMY / 美XmY    → Thiram=X*10ppm, MBA=Y*10ppm
  4. 二元 Thiram+MG:
     - 美X孔Y(+银)      → Thiram=X*10ppm, MG=Y*10ppm
  5. 三元混合:
     - 美XKYMZ / 美XKYmZ → Thiram=X*10ppm, MG=Y*10ppm, MBA=Z*10ppm

parser_version: v1.0
"""
import os
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
INVENTORY_PATH = PROJECT_ROOT / "data" / "inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

PARSER_VERSION = "v1.0"

# ====================== 浓度编码映射 ======================
# 文件夹名中的 0.4/0.5/0.6 → 4/5/6 ppm
# 文件夹名中的 数字 4/5/6 → 4/5/6 ppm
CONC_MAP = {'0.4': 4, '0.5': 5, '0.6': 6, '4': 4, '5': 5, '6': 6}


def parse_folder_name(folder_name):
    """
    解析文件夹名称，返回组分和浓度信息。
    返回 dict:
        family, mixture_order, has_mba, has_thiram, has_mg,
        c_mba, c_thiram, c_mg, has_agnps, role_mba, parse_note
    """
    result = {
        'family': None,
        'mixture_order': None,
        'has_mba': False,
        'has_thiram': False,
        'has_mg': False,
        'c_mba': 0,
        'c_thiram': 0,
        'c_mg': 0,
        'has_agnps': True,  # 所有实验都有AgNPs基底
        'role_mba': 'probe/reference',
        'parse_note': '',
    }
    
    fn = folder_name.strip()
    
    # ========== 模式 1: 单物质 + 银基底 ==========
    # mba+银+0.X
    m = re.match(r'^mba\+银\+(\d+\.?\d*)$', fn, re.IGNORECASE)
    if m:
        conc_str = m.group(1)
        c = CONC_MAP.get(conc_str)
        if c is None:
            result['parse_note'] = f'未知浓度: {conc_str}'
            return result
        result['family'] = 'single'
        result['mixture_order'] = 1
        result['has_mba'] = True
        result['c_mba'] = c
        return result
    
    # 孔+银+0.X
    m = re.match(r'^孔\+银\+(\d+\.?\d*)$', fn)
    if m:
        conc_str = m.group(1)
        c = CONC_MAP.get(conc_str)
        if c is None:
            result['parse_note'] = f'未知浓度: {conc_str}'
            return result
        result['family'] = 'single'
        result['mixture_order'] = 1
        result['has_mg'] = True
        result['c_mg'] = c
        return result
    
    # 福美双+银+0.X
    m = re.match(r'^福美双\+银\+(\d+\.?\d*)$', fn)
    if m:
        conc_str = m.group(1)
        c = CONC_MAP.get(conc_str)
        if c is None:
            result['parse_note'] = f'未知浓度: {conc_str}'
            return result
        result['family'] = 'single'
        result['mixture_order'] = 1
        result['has_thiram'] = True
        result['c_thiram'] = c
        return result
    
    # ========== 模式 5: 三元混合（先检查，避免被二元模式匹配）==========
    # 美XKYMZ / 美XKYmZ  → Thiram=X, MG=Y, MBA=Z
    m = re.match(r'^美(\d)[Kk孔](\d)[Mm](\d)$', fn)
    if m:
        ct = CONC_MAP.get(m.group(1))
        cmg = CONC_MAP.get(m.group(2))
        cm = CONC_MAP.get(m.group(3))
        if ct is not None and cmg is not None and cm is not None:
            result['family'] = 'ternary'
            result['mixture_order'] = 3
            result['has_thiram'] = True
            result['has_mg'] = True
            result['has_mba'] = True
            result['c_thiram'] = ct
            result['c_mg'] = cmg
            result['c_mba'] = cm
            return result
    
    # ========== 模式 4: 二元 Thiram+MG ==========
    # 美X孔Y(+银)?
    m = re.match(r'^美(\d)孔(\d)(\+银)?$', fn)
    if m:
        ct = CONC_MAP.get(m.group(1))
        cmg = CONC_MAP.get(m.group(2))
        if ct is not None and cmg is not None:
            result['family'] = 'binary_Thiram_MG'
            result['mixture_order'] = 2
            result['has_thiram'] = True
            result['has_mg'] = True
            result['c_thiram'] = ct
            result['c_mg'] = cmg
            return result
    
    # ========== 模式 3: 二元 Thiram+MBA ==========
    # 美XMY / 美XmY → Thiram=X, MBA=Y
    m = re.match(r'^美(\d)[Mm](\d)$', fn)
    if m:
        ct = CONC_MAP.get(m.group(1))
        cm = CONC_MAP.get(m.group(2))
        if ct is not None and cm is not None:
            result['family'] = 'binary_MBA_Thiram'
            result['mixture_order'] = 2
            result['has_thiram'] = True
            result['has_mba'] = True
            result['c_thiram'] = ct
            result['c_mba'] = cm
            return result
    
    # ========== 模式 2: 二元 MBA+MG ==========
    # 孔XmY / 孔XMY → MG=X, MBA=Y
    m = re.match(r'^孔(\d)[Mm](\d)$', fn)
    if m:
        cmg = CONC_MAP.get(m.group(1))
        cm = CONC_MAP.get(m.group(2))
        if cmg is not None and cm is not None:
            result['family'] = 'binary_MBA_MG'
            result['mixture_order'] = 2
            result['has_mg'] = True
            result['has_mba'] = True
            result['c_mg'] = cmg
            result['c_mba'] = cm
            return result
    
    # ========== 无法解析 ==========
    result['parse_note'] = f'无法解析文件夹名称: {folder_name}'
    return result


def build_metadata(inventory_df):
    """
    基于 inventory.csv 构建 metadata_v1.csv。
    仅处理 BWRam 格式文件（排除聚合文件和不可读文件）。
    """
    # 只为 BWRam 标准格式且可读的文件生成 metadata
    valid_df = inventory_df[
        (inventory_df['format_type'] == 'bwram') & 
        (inventory_df['read_ok'] == True)
    ].copy()
    
    records = []
    sample_counter = 0
    
    for idx, row in valid_df.iterrows():
        sample_counter += 1
        folder = row['folder_name']
        parsed = parse_folder_name(folder)
        
        record = {
            'sample_id': f'S{sample_counter:04d}',
            'file_path': row['file_path'],
            'file_name': row['file_name'],
            'folder_name': folder,
            'family': parsed['family'],
            'mixture_order': parsed['mixture_order'],
            'has_mba': parsed['has_mba'],
            'has_thiram': parsed['has_thiram'],
            'has_mg': parsed['has_mg'],
            'c_mba': parsed['c_mba'],
            'c_thiram': parsed['c_thiram'],
            'c_mg': parsed['c_mg'],
            'has_agnps': parsed['has_agnps'],
            'role_mba': parsed['role_mba'],
            'group_id': folder,
            'parser_version': PARSER_VERSION,
            'parse_note': parsed['parse_note'],
        }
        records.append(record)
    
    return pd.DataFrame(records)


def generate_coverage_summary(meta_df):
    """
    生成覆盖度统计表 coverage_summary.csv。
    """
    # 按组合统计
    rows = []
    
    for family in ['single', 'binary_MBA_MG', 'binary_MBA_Thiram', 
                    'binary_Thiram_MG', 'ternary']:
        sub = meta_df[meta_df['family'] == family]
        if len(sub) == 0:
            continue
        
        if family == 'single':
            for _, grp in sub.groupby(['has_mba', 'has_thiram', 'has_mg']):
                substance = 'MBA' if grp['has_mba'].iloc[0] else ('Thiram' if grp['has_thiram'].iloc[0] else 'MG')
                for conc in sorted(grp[f'c_{substance.lower()}' if substance != 'MG' else 'c_mg'].unique()):
                    mask = grp[f'c_{substance.lower()}' if substance != 'MG' else 'c_mg'] == conc
                    s = grp[mask]
                    rows.append({
                        'family': family,
                        'substance': substance,
                        'c_mba': int(s['c_mba'].iloc[0]),
                        'c_thiram': int(s['c_thiram'].iloc[0]),
                        'c_mg': int(s['c_mg'].iloc[0]),
                        'n_spectra': len(s),
                        'n_groups': s['group_id'].nunique(),
                        'group_ids': ', '.join(sorted(s['group_id'].unique())),
                    })
        else:
            for (cm, ct, cmg), grp in sub.groupby(['c_mba', 'c_thiram', 'c_mg']):
                rows.append({
                    'family': family,
                    'substance': family,
                    'c_mba': int(cm),
                    'c_thiram': int(ct),
                    'c_mg': int(cmg),
                    'n_spectra': len(grp),
                    'n_groups': grp['group_id'].nunique(),
                    'group_ids': ', '.join(sorted(grp['group_id'].unique())),
                })
    
    return pd.DataFrame(rows)


def generate_parser_report(meta_df, coverage_df, report_path):
    """
    生成 parser_check_report.md。
    """
    total = len(meta_df)
    parsed_ok = len(meta_df[meta_df['parse_note'] == ''])
    parse_fail = len(meta_df[meta_df['parse_note'] != ''])
    
    family_counts = meta_df['family'].value_counts()
    order_counts = meta_df['mixture_order'].value_counts().sort_index()
    
    # 检查不可能标签
    impossible = []
    for _, row in meta_df.iterrows():
        if row['family'] == 'single':
            conc_sum = (1 if row['has_mba'] else 0) + (1 if row['has_thiram'] else 0) + (1 if row['has_mg'] else 0)
            if conc_sum != 1:
                impossible.append(f"{row['sample_id']}: 单物质但组分数={conc_sum}")
        elif row['mixture_order'] == 2:
            conc_sum = (1 if row['has_mba'] else 0) + (1 if row['has_thiram'] else 0) + (1 if row['has_mg'] else 0)
            if conc_sum != 2:
                impossible.append(f"{row['sample_id']}: 二元但组分数={conc_sum}")
        elif row['mixture_order'] == 3:
            if not (row['has_mba'] and row['has_thiram'] and row['has_mg']):
                impossible.append(f"{row['sample_id']}: 三元但缺少组分")
        # 浓度检查
        for c_col in ['c_mba', 'c_thiram', 'c_mg']:
            if row[c_col] not in [0, 4, 5, 6]:
                impossible.append(f"{row['sample_id']}: {c_col}={row[c_col]} 不在合法范围")
    
    # 缺失组合分析
    all_concs = [4, 5, 6]
    missing_combinations = []
    
    # 缺失的三元组合
    existing_ternary = set()
    for _, row in meta_df[meta_df['family'] == 'ternary'].iterrows():
        existing_ternary.add((row['c_thiram'], row['c_mg'], row['c_mba']))
    
    for ct in all_concs:
        for cmg in all_concs:
            for cm in all_concs:
                if (ct, cmg, cm) not in existing_ternary:
                    missing_combinations.append(f"三元: Thiram={ct}, MG={cmg}, MBA={cm}")
    
    # 缺失的二元组合
    for fam, c1, c2, desc in [
        ('binary_MBA_Thiram', 'c_thiram', 'c_mba', 'Thiram-MBA'),
        ('binary_MBA_MG', 'c_mg', 'c_mba', 'MG-MBA'),
        ('binary_Thiram_MG', 'c_thiram', 'c_mg', 'Thiram-MG'),
    ]:
        existing = set()
        for _, row in meta_df[meta_df['family'] == fam].iterrows():
            existing.add((row[c1], row[c2]))
        for a in all_concs:
            for b in all_concs:
                if (a, b) not in existing:
                    missing_combinations.append(f"二元{desc}: {c1.split('_')[1]}={a}, {c2.split('_')[1]}={b}")
    
    report = f"""# 阶段 2：标签解析检查报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 解析器版本: {PARSER_VERSION}

## 一、解析概况

| 指标 | 值 |
|------|-----|
| 总光谱数（BWRam格式） | {total} |
| 解析成功 | {parsed_ok} |
| 解析失败 | {parse_fail} |

## 二、各 family 样本分布

| Family | 光谱数 | 占比 |
|--------|--------|------|
"""
    for fam, cnt in family_counts.items():
        pct = f"{cnt/total*100:.1f}%"
        report += f"| {fam} | {cnt} | {pct} |\n"
    
    report += f"""
## 三、各 mixture_order 分布

| 层级 | 光谱数 |
|------|--------|
"""
    for order, cnt in order_counts.items():
        report += f"| {int(order)} | {cnt} |\n"
    
    # 组分浓度组合统计
    report += f"""
## 四、各浓度组合统计

### 4.1 单物质

| 物质 | 浓度(ppm) | 光谱数 | 组数 |
|------|-----------|--------|------|
"""
    single_cov = coverage_df[coverage_df['family'] == 'single']
    for _, row in single_cov.iterrows():
        conc_val = max(row['c_mba'], row['c_thiram'], row['c_mg'])
        report += f"| {row['substance']} | {conc_val} | {row['n_spectra']} | {row['n_groups']} |\n"
    
    report += f"""
### 4.2 二元混合

| 类型 | Thiram(ppm) | MG(ppm) | MBA(ppm) | 光谱数 | 组数 |
|------|-------------|---------|----------|--------|------|
"""
    for fam in ['binary_MBA_MG', 'binary_MBA_Thiram', 'binary_Thiram_MG']:
        sub = coverage_df[coverage_df['family'] == fam]
        for _, row in sub.iterrows():
            report += f"| {fam} | {row['c_thiram']} | {row['c_mg']} | {row['c_mba']} | {row['n_spectra']} | {row['n_groups']} |\n"
    
    report += f"""
### 4.3 三元混合

| Thiram(ppm) | MG(ppm) | MBA(ppm) | 光谱数 | 组数 |
|-------------|---------|----------|--------|------|
"""
    ternary_cov = coverage_df[coverage_df['family'] == 'ternary']
    for _, row in ternary_cov.iterrows():
        report += f"| {row['c_thiram']} | {row['c_mg']} | {row['c_mba']} | {row['n_spectra']} | {row['n_groups']} |\n"
    
    # 不可能标签
    report += f"""
## 五、逻辑一致性检查

"""
    if len(impossible) > 0:
        report += f"> [!WARNING]\n> 发现 {len(impossible)} 条不可能标签：\n\n"
        for imp in impossible:
            report += f"- {imp}\n"
    else:
        report += "> [!NOTE]\n> 所有标签逻辑一致，未发现不可能组合。\n"
    
    # 解析失败列表
    if parse_fail > 0:
        report += f"""
## 六、解析失败样本

| sample_id | folder_name | parse_note |
|-----------|-------------|------------|
"""
        for _, row in meta_df[meta_df['parse_note'] != ''].iterrows():
            report += f"| {row['sample_id']} | {row['folder_name']} | {row['parse_note']} |\n"
    else:
        report += "\n## 六、解析失败样本\n\n无解析失败样本。\n"
    
    # 缺失组合
    report += f"""
## 七、缺失组合提示

> [!IMPORTANT]
> 以下组合在当前数据中不存在。特别注意**福美双(Thiram) 6 ppm 的混合物**缺失严重。

"""
    # 分组显示缺失
    missing_ternary = [m for m in missing_combinations if m.startswith('三元')]
    missing_binary = [m for m in missing_combinations if m.startswith('二元')]
    
    if missing_ternary:
        report += f"### 缺失三元组合 ({len(missing_ternary)} 个)\n\n"
        for m in missing_ternary:
            report += f"- {m}\n"
    
    if missing_binary:
        report += f"\n### 缺失二元组合 ({len(missing_binary)} 个)\n\n"
        for m in missing_binary:
            report += f"- {m}\n"
    
    if not missing_ternary and not missing_binary:
        report += "无缺失组合。\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


# ====================== 主流程 ======================

def main():
    print("=" * 60)
    print("阶段 2：标签解析与元数据标准化")
    print("=" * 60)
    
    # 读取 inventory
    print(f"\n[1/5] 读取 inventory.csv...")
    inv_df = pd.read_csv(INVENTORY_PATH)
    print(f"  共 {len(inv_df)} 个文件")
    
    # 构建 metadata
    print(f"\n[2/5] 解析标签...")
    meta_df = build_metadata(inv_df)
    print(f"  有效光谱: {len(meta_df)}")
    print(f"  解析成功: {len(meta_df[meta_df['parse_note'] == ''])}")
    print(f"  解析失败: {len(meta_df[meta_df['parse_note'] != ''])}")
    
    # 保存 metadata_v1.csv
    meta_path = OUTPUT_DIR / "metadata_v1.csv"
    print(f"\n[3/5] 保存 metadata_v1.csv -> {meta_path}")
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    
    # 生成覆盖度统计
    print(f"\n[4/5] 生成 coverage_summary.csv...")
    coverage_df = generate_coverage_summary(meta_df)
    coverage_path = OUTPUT_DIR / "coverage_summary.csv"
    coverage_df.to_csv(coverage_path, index=False, encoding='utf-8-sig')
    
    # 生成解析报告
    report_path = REPORT_DIR / "parser_check_report.md"
    print(f"\n[5/5] 生成解析报告 -> {report_path}")
    generate_parser_report(meta_df, coverage_df, report_path)
    
    # 摘要打印
    print(f"\n{'=' * 60}")
    print(f"标签解析完成！")
    print(f"\nFamily 分布:")
    for fam, cnt in meta_df['family'].value_counts().items():
        print(f"  {fam}: {cnt}")
    print(f"\n浓度分布:")
    for col in ['c_mba', 'c_thiram', 'c_mg']:
        vals = meta_df[meta_df[col] > 0][col].value_counts().sort_index()
        if len(vals) > 0:
            print(f"  {col}: {dict(vals)}")
    print(f"\n输出文件:")
    print(f"  - {meta_path}")
    print(f"  - {coverage_path}")
    print(f"  - {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
