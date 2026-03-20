# 阶段 5：数据划分报告 (V3)
> 生成时间: 2026-03-20 16:50:09

## 一、划分逻辑
1. **Folder 是最小独立样本单位**。同一 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，属于 technical replicates。
2. **随机按单条光谱切分不可用**，会造成特征泄漏。
3. **Group-aware split 是强制要求**。

## 二、方案选择依据
- V2 (5-fold, stratify=family): c_thiram 6ppm 仅存在于 3 个 folder，5 fold 下 3 个 fold 验证集必然挂零。已归档。
- **V3 (3-fold, stratify=c_thiram)**: 减少 fold 数至 3，以 c_thiram 为分层变量，尝试最大化 c_thiram 各等级在每折中的覆盖。

## 三、V3 各 Fold 标签覆盖
### c_thiram (Train / Val)
| Fold | Val Folders | Val N | Val 0 | Val 4 | Val 5 | Val 6 | Train 6 |
|------|-------------|-------|-------|-------|-------|-------|---------|
| 0 | 16 | 261 | 65 | 103 | 73 | 20 | 30 |
| 1 | 17 | 276 | 103 | 90 | 83 | 0 | 50 |
| 2 | 17 | 270 | 89 | 60 | 91 | 30 | 20 |

### c_mg (Val)
| Fold | Val 0 | Val 4 | Val 5 | Val 6 |
|------|-------|-------|-------|-------|
| 0 | 74 | 69 | 91 | 27 |
| 1 | 80 | 59 | 44 | 93 |
| 2 | 104 | 54 | 55 | 57 |

## 四、诚实声明
- 总共 50 个 folder。含 c_thiram=6 的 folder 仅 **3 个**，含 c_mg=6 的 folder 有 **12 个**。
- 3-fold 是在保持防泄漏前提下，c_thiram 6ppm 可被分配到验证集的极限方案。
- **警告**: Fold [1] 的 c_thiram=6 验证集仍为 0，说明即便 3 fold 也无法完全消除此穿孔。
- c_thiram 4 分类结果须谨慎解释(受限结果)；c_mg 覆盖显著优于 c_thiram。
