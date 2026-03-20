# 当前研究状态 (Current Research State)
> 生成时间: 2026-03-19 13:28:39

## 1. 数据直接支持的结论
- 混合拉曼光谱对 Thiram 和 MG 的存在性检测 (presence) 具有可靠的判别力 (Macro-F1 约 0.74-0.83)。
- 同一 folder 内的光谱为 technical replicates，存在 folder-level dependence，随机切分会造成泄漏。
- 当前全谱特征对"是否存在"的辨别远强于对"具体浓度等级 (4/5/6 ppm)"的区分。

## 2. 合理假设但证据不足
- Level 分类困难可能源于浓度梯度过小和多组分竞争吸附，但无法仅从模型分数确定。
- Folder-level clustering 的具体物理来源（操作员？基底？环境？）缺乏元数据确认。

## 3. 当前不能下结论
- 不能声称体系支持连续定量。
- 不能声称已证实 batch effect 的具体来源。
- 不能声称 MBA 峰点归一化是有效的内标策略（当前测试未见提升）。

## 4. 下一步优先事项
1. 补录 Thiram 6 ppm 混合物的缺失数据。
2. 提升 presence / level classification 的稳健性。
3. 探索 MBA 参考化的峰面积积分方法（子区间特征）。
4. 不将连续定量作为近期目标。
