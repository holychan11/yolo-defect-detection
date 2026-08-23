# NEU-DET 钢材表面缺陷检测 (YOLO)

基于 Ultralytics YOLO 的钢材表面缺陷检测项目，检测 6 类缺陷：
**裂纹 (crazing)、夹杂 (inclusion)、麻点 (patches)、氧化皮 (pitted_surface)、轧制氧化皮 (rolled-in_scale)、划痕 (scratches)**。

> 📦 本项目已包含数据集（`dataset/`，1800 张图像 + 标签），克隆后可直接训练。

## 克隆与运行

```bash
git clone <仓库地址>
cd yolo-defect-detection

# 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 直接推理 (需先训练获得 best.pt, 或用预训练权重)
python detect.py test_sample.jpg
```

## 训练结果 (yolo11s, 360 张验证集)

| 指标 | 新模型 (train-new) | 旧模型 (train-4) |
|---|---|---|
| Precision | **0.769** | 0.721 |
| Recall | 0.686 | 0.742 |
| mAP50 | 0.771 | 0.792 |
| mAP50-95 | 0.438 | 0.463 |

- 新模型为干净从头训练 (150 epochs)；旧模型为四轮续训 + 更大训练集 (1770 张)，故分数略高，但其指标是在仅 30 张验证集上挑选的，不可靠
- 新模型 Precision 更高（误报更少），且 360 张验证集评估结果可信
- 弱项: crazing (裂纹, mAP50=0.427) 与 rolled-in_scale (轧制氧化皮, 0.626)

## 项目结构

```
yolo-defect-detection/
├── data.yaml            # 数据集配置 (相对路径, 可移植)
├── dataset/             # 数据集 (images/ + labels/, train 1440 / val 360)
├── train.py             # 训练脚本
├── detect.py            # 推理脚本 (自动加载 best.pt)
├── evaluate.py          # 评估脚本 (mAP / 混淆矩阵 / 每类指标)
├── export.py            # 模型导出 (ONNX / TensorRT / OpenVINO 等)
├── split_dataset.py     # 数据集划分脚本 (8:2, 固定随机种子)
├── compare.py           # 新旧模型对比评估
├── requirements.txt     # 依赖
├── venv/                # Python 虚拟环境 (Python 3.11)
├── yolo11n.pt / yolo11s.pt   # 预训练权重
├── test_sample.jpg      # 推理示例图 (来自验证集)
├── archive/             # 归档: 旧推理输出 + 冗余图片副本
└── runs/detect/         # 训练与推理输出
```

## 快速开始

### 1. 环境

```bash
# 激活已有虚拟环境 (Windows)
venv\Scripts\activate

# 或新建环境并安装依赖
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 训练

```bash
# 默认: yolo11s, 150 轮, 自动选 GPU
python train.py

# 自定义参数
python train.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16

# 断点续训
python train.py --resume runs/detect/train/weights/last.pt
```

训练日志与最佳权重保存在 `runs/detect/trainN/weights/best.pt`。

### 3. 推理

```bash
# 单张图片 (自动加载最新 best.pt)
python detect.py test_sample.jpg

# 整个文件夹 / 视频 / 摄像头
python detect.py dataset/images/val/
python detect.py video.mp4
python detect.py 0

# 调整阈值 / 指定权重
python detect.py --conf 0.3 --weights runs/detect/train/weights/best.pt test_sample.jpg
```

结果保存在 `runs/detect/predictN/`，控制台会打印每个缺陷的类别和置信度。

### 4. 评估

```bash
# 在验证集上评估最新 best.pt
python evaluate.py

# 指定权重
python evaluate.py --weights runs/detect/train/weights/best.pt
```

输出 mAP50 / mAP50-95 / Precision / Recall，以及每个类别的详细指标和混淆矩阵图（保存在 `runs/detect/valN/`）。

### 5. 导出 (部署加速)

```bash
# 导出 ONNX (默认)
python export.py

# 导出 TensorRT (需 GPU, 速度最快)
python export.py --format engine --half
```

## 数据集

- 来源: NEU-DET (东北大学钢材表面缺陷数据集)，1800 张，每类 300 张
- 划分: 8:2，train 1440 / val 360（每类 240 / 60），固定随机种子 42，可复现
- 重新划分: `python split_dataset.py [--seed N]`

## 硬件

- 已测试: NVIDIA RTX 3060 Laptop 6GB, CUDA 12.1, torch 2.5.1
- 自动选设备: 训练/推理脚本默认优先使用 GPU
