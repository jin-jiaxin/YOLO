# 基于轻量化 YOLO 的交通场景目标检测模型对比与复现

本项目为《图像处理与分析（计算机视觉）》研究生课程结课作业。项目基于 Ultralytics 框架，在 COCO128 微型数据集上完成了轻量化目标检测模型（YOLOv5nu、YOLOv8n）的基线对比，并在此基础上成功复现与改进了 **Ghost-CBAM-YOLOv8** 轻量化模型。

## 📁 仓库结构说明 (Repository Structure)

*   **`notebook12342e2437.ipynb`**: 完整的训练代码文件（包含基线训练、自定义 GhostConv 与 CBAM 模块的注入、改进模型 YAML 构建及训练过程）。
*   **`weights/`**: 包含训练好的模型权重文件。
    *   `baseline_best.pt`: YOLOv8n 基线模型的最优权重。
    *   `improved_best.pt`: Ghost-CBAM-YOLOv8 改进模型的最优权重。
*   **`result/`**: 包含实验过程生成的可视化图表（PR 曲线、混淆矩阵、检测效果图、训练曲线等）。
*   **`data.yaml`**: COCO128 数据集配置文件。

## 🚀 模型与改进方案 (Models & Method)

1.  **Baseline 模型**: 选用 YOLOv5nu 与 YOLOv8n 作为轻量化基线，在完全相同的超参数下进行 30 Epoch 训练对比。
2.  **提出方法 (Proposed Method)**: 针对边缘计算场景，本文设计了 **Ghost-CBAM-YOLOv8** 结构。
    *   **Ghost 轻量化**: 将主干网络下采样层替换为 GhostConv，通过廉价线性变换生成冗余特征，有效压缩模型参数量。
    *   **CBAM 注意力增强**: 在网络浅层与深层特征图后插入 CBAM（通道与空间注意力）模块，抑制背景干扰，弥补轻量化带来的特征损失。

## 📊 实验结果 (Experimental Results)

*注：受限于 COCO128 微型数据集（样本量极少），整体绝对精度偏低，本实验侧重于观察模型间的相对趋势与轻量化收益。*

| 模型名称 | 参数量 (Params) | 计算量 (GFLOPs) | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| YOLOv5nu (Baseline) | 2.51 M | 7.2 | 0.01 | 0.00 |
| YOLOv8n (Baseline) | 3.01 M | 8.2 | 0.03 | 0.00 |
| **Ours (Ghost-CBAM-YOLOv8)** | **2.88 M** | **8.3** | **0.62** | **0.02** |

*(注：请根据你论文里最终的表格数据，核对一下上面表格里的 mAP 数值是否完全一致)*

## ⚙️ 环境与依赖 (Requirements)

*   Python 3.10+
*   PyTorch 2.0+
*   Ultralytics 8.0+
*   OpenCV, NumPy, Pandas

## 📜 权重下载与使用 (Weights Usage)

本项目已开源基线模型与改进模型的 `best.pt` 权重，位于 `weights/` 目录下。

快速测试权重：
```python
from ultralytics import YOLO

# 加载改进模型权重
model = YOLO('weights/improved_best.pt')

# 进行推理测试
results = model.predict('path/to/your/image.jpg', imgsz=320)
