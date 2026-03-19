# 阶段 2：标签解析检查报告

> 生成时间: 2026-03-14 19:03:58
> 解析器版本: v1.0

## 一、解析概况

| 指标 | 值 |
|------|-----|
| 总光谱数（BWRam格式） | 807 |
| 解析成功 | 807 |
| 解析失败 | 0 |

## 二、各 family 样本分布

| Family | 光谱数 | 占比 |
|--------|--------|------|
| ternary | 265 | 32.8% |
| single | 187 | 23.2% |
| binary_MBA_Thiram | 139 | 17.2% |
| binary_MBA_MG | 128 | 15.9% |
| binary_Thiram_MG | 88 | 10.9% |

## 三、各 mixture_order 分布

| 层级 | 光谱数 |
|------|--------|
| 1 | 187 |
| 2 | 355 |
| 3 | 265 |

## 四、各浓度组合统计

### 4.1 单物质

| 物质 | 浓度(ppm) | 光谱数 | 组数 |
|------|-----------|--------|------|
| MG | 4 | 21 | 1 |
| MG | 5 | 27 | 1 |
| MG | 6 | 20 | 1 |
| Thiram | 4 | 18 | 1 |
| Thiram | 5 | 20 | 1 |
| Thiram | 6 | 20 | 1 |
| MBA | 4 | 20 | 1 |
| MBA | 5 | 20 | 1 |
| MBA | 6 | 21 | 1 |

### 4.2 二元混合

| 类型 | Thiram(ppm) | MG(ppm) | MBA(ppm) | 光谱数 | 组数 |
|------|-------------|---------|----------|--------|------|
| binary_MBA_MG | 0 | 4 | 4 | 14 | 1 |
| binary_MBA_MG | 0 | 5 | 4 | 14 | 1 |
| binary_MBA_MG | 0 | 6 | 4 | 15 | 1 |
| binary_MBA_MG | 0 | 4 | 5 | 12 | 1 |
| binary_MBA_MG | 0 | 5 | 5 | 14 | 1 |
| binary_MBA_MG | 0 | 6 | 5 | 15 | 1 |
| binary_MBA_MG | 0 | 4 | 6 | 14 | 1 |
| binary_MBA_MG | 0 | 5 | 6 | 16 | 1 |
| binary_MBA_MG | 0 | 6 | 6 | 14 | 1 |
| binary_MBA_Thiram | 4 | 0 | 4 | 20 | 1 |
| binary_MBA_Thiram | 5 | 0 | 4 | 16 | 1 |
| binary_MBA_Thiram | 6 | 0 | 4 | 14 | 1 |
| binary_MBA_Thiram | 4 | 0 | 5 | 19 | 1 |
| binary_MBA_Thiram | 5 | 0 | 5 | 16 | 1 |
| binary_MBA_Thiram | 6 | 0 | 5 | 16 | 1 |
| binary_MBA_Thiram | 4 | 0 | 6 | 18 | 1 |
| binary_MBA_Thiram | 5 | 0 | 6 | 20 | 1 |
| binary_Thiram_MG | 4 | 4 | 0 | 20 | 1 |
| binary_Thiram_MG | 4 | 5 | 0 | 15 | 1 |
| binary_Thiram_MG | 4 | 6 | 0 | 14 | 1 |
| binary_Thiram_MG | 5 | 4 | 0 | 13 | 1 |
| binary_Thiram_MG | 5 | 5 | 0 | 14 | 1 |
| binary_Thiram_MG | 5 | 6 | 0 | 12 | 1 |

### 4.3 三元混合

| Thiram(ppm) | MG(ppm) | MBA(ppm) | 光谱数 | 组数 |
|-------------|---------|----------|--------|------|
| 4 | 4 | 4 | 12 | 1 |
| 4 | 5 | 4 | 16 | 1 |
| 4 | 6 | 4 | 14 | 1 |
| 5 | 4 | 4 | 14 | 1 |
| 5 | 5 | 4 | 15 | 1 |
| 5 | 6 | 4 | 14 | 1 |
| 4 | 4 | 5 | 15 | 1 |
| 4 | 5 | 5 | 15 | 1 |
| 4 | 6 | 5 | 14 | 1 |
| 5 | 4 | 5 | 18 | 1 |
| 5 | 5 | 5 | 14 | 1 |
| 5 | 6 | 5 | 13 | 1 |
| 4 | 4 | 6 | 15 | 1 |
| 4 | 5 | 6 | 15 | 1 |
| 4 | 6 | 6 | 13 | 1 |
| 5 | 4 | 6 | 14 | 1 |
| 5 | 5 | 6 | 15 | 1 |
| 5 | 6 | 6 | 19 | 1 |

## 五、逻辑一致性检查

> [!NOTE]
> 所有标签逻辑一致，未发现不可能组合。

## 六、解析失败样本

无解析失败样本。

## 七、缺失组合提示

> [!IMPORTANT]
> 以下组合在当前数据中不存在。特别注意**福美双(Thiram) 6 ppm 的混合物**缺失严重。

### 缺失三元组合 (9 个)

- 三元: Thiram=6, MG=4, MBA=4
- 三元: Thiram=6, MG=4, MBA=5
- 三元: Thiram=6, MG=4, MBA=6
- 三元: Thiram=6, MG=5, MBA=4
- 三元: Thiram=6, MG=5, MBA=5
- 三元: Thiram=6, MG=5, MBA=6
- 三元: Thiram=6, MG=6, MBA=4
- 三元: Thiram=6, MG=6, MBA=5
- 三元: Thiram=6, MG=6, MBA=6

### 缺失二元组合 (4 个)

- 二元Thiram-MBA: thiram=6, mba=6
- 二元Thiram-MG: thiram=6, mg=4
- 二元Thiram-MG: thiram=6, mg=5
- 二元Thiram-MG: thiram=6, mg=6
