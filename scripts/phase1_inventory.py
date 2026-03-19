"""
阶段 1：数据清点与可读性检查
=============================
扫描整个数据目录，递归读取所有光谱文件，输出 inventory.csv 和 inventory_summary.md。
"""
import os
import sys
import csv
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
RAW_DATA_DIR = PROJECT_ROOT / "混合数据" / "mba+福+孔"
OUTPUT_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ====================== 统一读取接口 ======================

def detect_csv_format(filepath):
    """
    检测 CSV 文件的格式类型。
    返回:
        'bwram' - 标准 BWRam 设备导出格式（含元数据头）
        'aggregated' - 聚合格式（多条谱合并在一个文件中）
        'unknown' - 其他格式
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline().strip()
            if first_line.startswith('File Version'):
                return 'bwram'
            elif first_line.startswith('SP_'):
                return 'aggregated'
            else:
                # 尝试判断是否为纯数值
                parts = first_line.split(',')
                try:
                    float(parts[0])
                    return 'numeric'
                except ValueError:
                    return 'unknown'
    except Exception:
        return 'error'


def read_bwram_csv(filepath):
    """
    读取 BWRam 标准格式 CSV。
    返回:
        dict: {
            'wavenumber': np.array,   # Raman Shift (cm-1)
            'intensity': np.array,    # Dark Subtracted #1
            'n_points': int,
            'wn_min': float,
            'wn_max': float,
            'metadata': dict          # 文件头部元数据
        }
    """
    metadata = {}
    header_line = None
    data_start = None
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith('Pixel,Wavelength'):
                header_line = i
                data_start = i + 1
                break
            else:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
    
    if data_start is None:
        return None
    
    # 读取数据部分
    df = pd.read_csv(filepath, skiprows=data_start - 1, header=0,
                     skipinitialspace=True, on_bad_lines='skip')
    
    # 列名不同文件可能有细微差别，按位置取
    # 典型列: Pixel, Wavelength, Wavenumber, Raman Shift, Dark, Reference, Raw data #1, Dark Subtracted #1, ...
    col_names = df.columns.tolist()
    
    # 找到 Raman Shift 列（第4列，索引3）和 Dark Subtracted 列（第8列，索引7）
    raman_shift_col = None
    intensity_col = None
    
    for j, c in enumerate(col_names):
        c_lower = c.strip().lower()
        if 'raman shift' in c_lower:
            raman_shift_col = c
        if 'dark subtracted' in c_lower:
            intensity_col = c
    
    if raman_shift_col is None:
        # 尝试按位置
        if len(col_names) >= 8:
            raman_shift_col = col_names[3]
            intensity_col = col_names[7]
        else:
            return None
    
    # 提取数据
    wn = pd.to_numeric(df[raman_shift_col], errors='coerce').values
    intensity = pd.to_numeric(df[intensity_col], errors='coerce').values
    
    # 去除波数为空的行（前面几个像素没有波数校准）
    valid_mask = ~np.isnan(wn) & ~np.isnan(intensity)
    wn = wn[valid_mask]
    intensity = intensity[valid_mask]
    
    if len(wn) == 0:
        return None
    
    return {
        'wavenumber': wn,
        'intensity': intensity,
        'n_points': len(wn),
        'wn_min': float(np.nanmin(wn)),
        'wn_max': float(np.nanmax(wn)),
        'metadata': metadata
    }


def read_aggregated_csv(filepath):
    """
    读取聚合格式 CSV（如 3333.csv）。
    第一行是列名（SP_1.csv, SP_10.csv, ...），后续行是强度值，没有波数列。
    返回所有谱的列表（但不包含波数信息）。
    """
    try:
        df = pd.read_csv(filepath, header=0)
        spectra = []
        for col in df.columns:
            intensity = pd.to_numeric(df[col], errors='coerce').values
            valid = intensity[~np.isnan(intensity)]
            spectra.append({
                'source_col': col.strip(),
                'intensity': valid,
                'n_points': len(valid),
                'wn_min': None,  # 无波数信息
                'wn_max': None,
            })
        return spectra
    except Exception:
        return None


# ====================== 扫描与清点 ======================

def scan_data_directory(data_dir):
    """
    递归扫描数据目录，返回清点信息列表。
    """
    records = []
    
    for root, dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(data_dir)
            folder_name = fpath.parent.name
            extension = fpath.suffix.lower()
            
            record = {
                'file_path': str(fpath),
                'relative_path': str(rel_path),
                'file_name': fname,
                'folder_name': folder_name,
                'extension': extension,
                'file_size_bytes': fpath.stat().st_size,
                'read_ok': False,
                'format_type': 'unknown',
                'n_points': 0,
                'wn_min': None,
                'wn_max': None,
                'notes': '',
            }
            
            if extension != '.csv':
                record['notes'] = f'非CSV文件，后缀为{extension}'
                records.append(record)
                continue
            
            # 检测格式
            fmt = detect_csv_format(str(fpath))
            record['format_type'] = fmt
            
            if fmt == 'bwram':
                try:
                    result = read_bwram_csv(str(fpath))
                    if result is not None:
                        record['read_ok'] = True
                        record['n_points'] = result['n_points']
                        record['wn_min'] = result['wn_min']
                        record['wn_max'] = result['wn_max']
                    else:
                        record['notes'] = '解析失败：无法提取波数/强度数据'
                except Exception as e:
                    record['notes'] = f'读取异常: {str(e)[:100]}'
            
            elif fmt == 'aggregated':
                try:
                    spectra = read_aggregated_csv(str(fpath))
                    if spectra is not None and len(spectra) > 0:
                        record['read_ok'] = True
                        record['n_points'] = spectra[0]['n_points']
                        record['notes'] = f'聚合文件，包含{len(spectra)}条谱，无独立波数轴'
                    else:
                        record['notes'] = '聚合文件解析失败'
                except Exception as e:
                    record['notes'] = f'读取异常: {str(e)[:100]}'
            
            elif fmt == 'error':
                record['notes'] = '文件无法打开'
            else:
                record['notes'] = f'未识别格式: {fmt}'
            
            records.append(record)
    
    return records


def generate_summary(records, output_path):
    """
    生成 inventory_summary.md 报告。
    """
    df = pd.DataFrame(records)
    
    total_files = len(df)
    csv_files = len(df[df['extension'] == '.csv'])
    non_csv = len(df[df['extension'] != '.csv'])
    readable = len(df[df['read_ok'] == True])
    unreadable = len(df[df['read_ok'] == False])
    
    # 按文件夹统计
    folder_stats = df.groupby('folder_name').agg(
        file_count=('file_name', 'count'),
        readable_count=('read_ok', 'sum'),
        avg_points=('n_points', 'mean'),
    ).reset_index()
    
    # 按后缀统计
    ext_stats = df['extension'].value_counts().to_dict()
    
    # 按格式类型统计
    fmt_stats = df['format_type'].value_counts().to_dict()
    
    # 波数范围
    bwram_df = df[(df['format_type'] == 'bwram') & (df['read_ok'] == True)]
    if len(bwram_df) > 0:
        wn_mins = bwram_df['wn_min'].dropna()
        wn_maxs = bwram_df['wn_max'].dropna()
        wn_min_range = f"{wn_mins.min():.2f} ~ {wn_mins.max():.2f}" if len(wn_mins) > 0 else "N/A"
        wn_max_range = f"{wn_maxs.min():.2f} ~ {wn_maxs.max():.2f}" if len(wn_maxs) > 0 else "N/A"
        points_range = f"{int(bwram_df['n_points'].min())} ~ {int(bwram_df['n_points'].max())}"
        points_unique = sorted(bwram_df['n_points'].unique().astype(int))
    else:
        wn_min_range = wn_max_range = points_range = "N/A"
        points_unique = []
    
    # 异常文件
    anomaly_df = df[df['read_ok'] == False]
    aggregated_df = df[df['format_type'] == 'aggregated']
    
    # 计算总有效光谱数量（bwram 单文件 = 1条谱）
    total_spectra_bwram = len(bwram_df)
    total_spectra_agg = 0
    for _, row in aggregated_df.iterrows():
        if 'contain' in row.get('notes', '') or '包含' in row.get('notes', ''):
            import re
            m = re.search(r'(\d+)条谱', row['notes'])
            if m:
                total_spectra_agg += int(m.group(1))
    
    report = f"""# 阶段 1：数据清点报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、数据规模概览

| 指标 | 值 |
|------|-----|
| 总文件数 | {total_files} |
| CSV 文件数 | {csv_files} |
| 非 CSV 文件数 | {non_csv} |
| 可正常读取 | {readable} |
| 无法读取/异常 | {unreadable} |
| BWRam 标准格式谱（每文件1条） | {total_spectra_bwram} |
| 聚合格式文件内总谱数 | {total_spectra_agg} |
| 文件夹数 | {len(folder_stats)} |

## 二、文件后缀分布

| 后缀 | 文件数 |
|------|--------|
"""
    for ext, count in ext_stats.items():
        report += f"| {ext} | {count} |\n"
    
    report += f"""
## 三、格式类型分布

| 格式 | 文件数 | 说明 |
|------|--------|------|
"""
    fmt_descriptions = {
        'bwram': '标准 BWRam 设备导出格式（含元数据头+波数+强度）',
        'aggregated': '聚合格式（多条谱合并，无波数列）',
        'unknown': '未识别格式',
        'error': '文件打开失败',
    }
    for fmt, count in fmt_stats.items():
        desc = fmt_descriptions.get(fmt, '未知')
        report += f"| {fmt} | {count} | {desc} |\n"
    
    report += f"""
## 四、波数轴与点数信息

> 以下统计仅基于 BWRam 标准格式文件（可正常读取的）

| 指标 | 值 |
|------|-----|
| 波数最小值范围 | {wn_min_range} cm⁻¹ |
| 波数最大值范围 | {wn_max_range} cm⁻¹ |
| 每谱数据点数范围 | {points_range} |
| 数据点数去重列表 | {points_unique} |

## 五、各文件夹文件数统计

| 文件夹 | 文件总数 | 可读数 | 平均点数 |
|--------|---------|--------|----------|
"""
    for _, row in folder_stats.iterrows():
        avg_pts = f"{row['avg_points']:.0f}" if row['avg_points'] > 0 else "N/A"
        report += f"| {row['folder_name']} | {int(row['file_count'])} | {int(row['readable_count'])} | {avg_pts} |\n"
    
    # 异常文件列表
    if len(anomaly_df) > 0:
        report += f"""
## 六、异常/无法读取的文件

| 文件路径 | 格式类型 | 备注 |
|----------|---------|------|
"""
        for _, row in anomaly_df.iterrows():
            report += f"| {row['relative_path']} | {row['format_type']} | {row['notes']} |\n"
    else:
        report += "\n## 六、异常文件\n\n无异常文件，所有文件均可正常读取。\n"
    
    # 聚合文件说明
    if len(aggregated_df) > 0:
        report += f"""
## 七、聚合格式文件说明

> [!WARNING]
> 以下文件为聚合格式，包含多条谱但**无独立波数轴**。
> 后续预处理阶段需要特殊处理：要么从同文件夹其他 BWRam 文件中提取波数轴进行对齐，
> 要么排除这些文件。建议在阶段 3 统一处理。

| 文件 | 文件夹 | 备注 |
|------|--------|------|
"""
        for _, row in aggregated_df.iterrows():
            report += f"| {row['file_name']} | {row['folder_name']} | {row['notes']} |\n"
    
    report += f"""
## 八、数据质量结论

1. **格式统一性**：绝大部分文件为 BWRam 标准格式，格式基本统一。
2. **聚合文件**：{'存在' if len(aggregated_df) > 0 else '不存在'}聚合格式文件（如 3333.csv），这些文件的处理策略需要在阶段 3 确定。
3. **波数轴一致性**：{f'数据点数为 {points_unique}，需在阶段 3 检查是否需要插值对齐。' if len(points_unique) > 1 else f'所有标准格式文件点数一致（{points_unique[0]}），波数轴高度一致。' if len(points_unique) == 1 else '无可用数据。'}
4. **下一步建议**：
   - 使用 BWRam 标准格式文件的 `Dark Subtracted #1` 列作为强度值
   - `Raman Shift` 列作为波数轴
   - 聚合文件暂时标记，后续与同组 BWRam 文件共享波数轴
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


# ====================== 主流程 ======================

def main():
    print("=" * 60)
    print("阶段 1：数据清点与可读性检查")
    print("=" * 60)
    print(f"\n数据目录: {RAW_DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 1. 扫描
    print("\n[1/3] 正在递归扫描数据目录...")
    records = scan_data_directory(RAW_DATA_DIR)
    print(f"  扫描完成：发现 {len(records)} 个文件")
    
    # 2. 保存 inventory.csv
    inventory_path = OUTPUT_DIR / "inventory.csv"
    print(f"\n[2/3] 保存 inventory.csv -> {inventory_path}")
    df = pd.DataFrame(records)
    df.to_csv(inventory_path, index=False, encoding='utf-8-sig')
    
    # 3. 生成报告
    report_path = REPORT_DIR / "inventory_summary.md"
    print(f"\n[3/3] 生成清点报告 -> {report_path}")
    report = generate_summary(records, report_path)
    
    # 打印摘要
    readable_count = sum(1 for r in records if r['read_ok'])
    print(f"\n{'=' * 60}")
    print(f"清点完成！")
    print(f"  总文件: {len(records)}")
    print(f"  可读取: {readable_count}")
    print(f"  异常:   {len(records) - readable_count}")
    print(f"\n输出文件:")
    print(f"  - {inventory_path}")
    print(f"  - {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
