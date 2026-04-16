# SERS 多组分分析项目

本项目用于根据 SERS 拉曼光谱同时判断 Thiram、MG、MBA 三种物质的有无与浓度等级，当前仓库代码已经整理为可直接上传服务器运行的 Round 4 版本。

## 当前 Round 4 重点

- 保留原始 4 类浓度任务：0 / 4 / 5 / 6 ppm
- 新增 3 类半定量任务：0 / 4 / 5+6 ppm
- 深度学习损失改为 Focal Loss，并启用类别权重
- 特征工程新增二阶导特征
- 评估指标新增 Accuracy
- 结果保存逻辑已修复，4 类和 3 类结果不会互相覆盖

## 数据与评估

- 光谱数：954
- 文件夹组数：63
- 波数点：1401
- 交叉验证：5 折 StratifiedGroupKFold 思路，实际按已有 fold_id 分折
- 分组键：folder_name

## 任务设置

原始 7 个任务：

- T1_thiram_conc
- T2_mg_conc
- T3_mba_conc
- T4_thiram_pres
- T5_mg_pres
- T6_mba_pres
- T7_mixture_order

新增 3 个 3 类任务：

- T1b_thiram_3c
- T2b_mg_3c
- T3b_mba_3c

多任务 3 类阶段使用：3 类浓度 + 3 个 presence + mixture_order，共 7 个输出头。

## 模型

仓库当前支持 14 个模型名：

- RF
- SVM
- PLS-DA
- 1D-CNN
- 1D-ResNet
- Feature-KAN
- MT-CNN
- MT-ResNet
- MT-KAN-CNN
- MT-Feature-KAN
- MT-CNN-3c
- MT-ResNet-3c
- MT-KAN-CNN-3c
- MT-Feature-KAN-3c

说明：1D-CNN、1D-ResNet、Feature-KAN 会同时用于 4 类与 3 类单任务阶段，但模型名保持不变，靠 Task 区分。

## 一键运行

推荐直接运行：

```bash
python batch_train.py
```

脚本会自动：

- 检查并补齐 cv_split_v5.csv 中的 c_thiram_3c / c_mg_3c / c_mba_3c
- 自动读取 raw、p1、p2、p3、p4 五种预处理数据
- 分 6 个阶段完成全部训练与保存

6 个阶段分别是：

1. ML 跑原始 7 任务
2. ML 跑 3 类任务
3. 单任务 DL 跑原始 7 任务
4. 单任务 DL 跑 3 类任务
5. 多任务模型跑原始 7 任务
6. 多任务 3 类模型跑 7 输出头任务

## 结果文件

结果默认写入 data/models：

- cv_results_detail_*.csv
- cv_results_summary_*.csv
- cv_predictions_*.csv

完整 Round 4 跑完后，每个预处理 summary 文件理论上应包含 116 行：

- ML 原始任务 21 行
- ML 3 类任务 9 行
- 单任务 DL 原始任务 21 行
- 单任务 DL 3 类任务 9 行
- 多任务原始任务 28 行
- 多任务 3 类任务 28 行

## 关键文件

- src/config.py：任务、超参数、路径
- src/models.py：全部模型定义与注册表
- src/feature_engineering.py：领域特征与二阶导特征
- src/train_eval.py：CV、指标汇总、结果合并保存
- batch_train.py：服务器上一键跑的主脚本
- SERVER_MANUAL.md：服务器操作手册
- server_update_prompt.txt：给新服务器 GPT 的对齐提示词
- reports/experiment_history_archive.md：历史实验归档