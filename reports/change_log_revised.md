# 变更日志 (Change Log)
> 生成时间: 2026-03-19 13:28:39

## 1. 废除 "batch effect" 表述
- 原: 使用 "batch effect" 和基于单条光谱的伪重复 ANOVA。
- 改: 统一为 "folder-level dependence/clustering"；撤回伪重复统计。
- 原因: 缺乏实验元数据，不能跨界定论。

## 2. 废除连续定量回归
- 原: RMSE/R² 评价的连续预测，含回归散点图。
- 改: 0/4/5/6 ppm 四分类，Macro-F1 + Balanced Accuracy。
- 原因: 数据不支持连续定量结论。

## 3. MBA 角色重定义
- 原: 与 Thiram/MG 并列作为第三目标物。
- 改: probe / internal standard，检测结果仅作内部监控。
- 原因: MBA 信号极强且分布广，其高分掩盖了真实检测难度。

## 4. 清除导师汇报包装
- 原: report_for_supervisor.md 等面向汇报的文件。
- 改: 全部归档，只保留研究导向的客观报告。
- 原因: 结论超出数据支持范围。
