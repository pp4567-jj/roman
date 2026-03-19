# 建模范围定义 (修订版)
> 生成时间: 2026-03-19 13:28:39

## 角色界定
- **MBA**: probe / internal standard / reference molecule。不是主要目标物。
- **Thiram, MG**: 主要目标分析物 (target analytes)。

## 当前正式任务
1. mixture_order: 混合复杂度分类 (1/2/3)
2. has_thiram: 福美双存在性分类 (0/1)
3. has_mg: 孔雀石绿存在性分类 (0/1)
4. c_thiram: 福美双浓度等级分类 (0/4/5/6 ppm)
5. c_mg: 孔雀石绿浓度等级分类 (0/4/5/6 ppm)
6. c_mba: MBA等级分类 (内部监控，不对外报告为主要成果)

## 当前不做的事
- 连续浓度回归 (RMSE/R² 已废弃)
- 将 MBA 检测结果作为与 Thiram/MG 并列的主要成果
