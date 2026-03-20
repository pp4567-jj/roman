# 建模范围定义 (修订版)
> 生成时间: 2026-03-20 16:50:09

## 角色界定
- **MBA**: probe / internal standard / reference molecule。不是主要目标物。
- **Thiram, MG**: 主要目标分析物 (target analytes)。

## 当前正式任务与分层
### 正文主结果
- has_thiram: 福美双存在性分类 (0/1)
- has_mg: 孔雀石绿存在性分类 (0/1)

### 正文辅助结果
- mixture_order: 混合复杂度分类 (1/2/3)
- c_mg: 孔雀石绿浓度等级分类 (0/4/5/6 ppm)

### 受限解释结果
- c_thiram: 福美双浓度等级分类 (0/4/5/6 ppm) — 6ppm 仅 3 folder，评估底座受限

### 内部监控
- c_mba: MBA 等级分类 (内部监控，不对外报告)

## 当前不做的事
- 连续浓度回归
- 将 MBA 作为主要成果
- 将 c_thiram level 作为强主结论
