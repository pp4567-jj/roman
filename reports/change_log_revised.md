# 变更日志 (Change Log)
> 生成时间: 2026-03-20 16:50:09

## 1. Split V2 → V3
- 原: 5-fold, stratify=family. c_thiram 6ppm 3 fold 验证集挂零。
- 改: 3-fold, stratify=c_thiram. 减少 fold 数以尽量覆盖稀缺等级。
- 原因: c_thiram 6ppm 仅 3 个 folder, 5 fold 物理上无法分配。

## 2. 混淆矩阵改为最佳模型
- 原: 全部硬绑 RF。
- 改: 每个任务自动选择 Macro-F1 最高的模型绘图。
- 原因: 真实数据表明不同任务最优模型不同。

## 3. 任务分层体系
- presence (主结果) > c_mg level (辅助) > c_thiram level (受限) > c_mba (内部)
- 原因: split 审计表明 c_thiram 评估底座受限。

## 4. 历史保留
- 废除 batch effect, 连续回归, MBA 作为目标物等表述不变。
