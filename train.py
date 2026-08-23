"""
train.py
========
NEU-DET 钢材表面缺陷检测 - 训练脚本

用法:
    python train.py                                # 默认: yolo11s.pt, 150 epochs, 自动选设备
    python train.py --model yolo11n.pt --epochs 100
    python train.py --model yolo11m.pt --imgsz 960 --batch 16
    python train.py --resume runs/detect/train/weights/last.pt   # 断点续训

说明:
    - 自动选择 GPU (device 0) 或 CPU
    - 训练结束自动保留 best.pt (验证集最优) 和 last.pt
    - 日志与权重保存在 runs/detect/trainN/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def pick_device(prefer_cuda: bool) -> str:
    """自动选择设备: 优先 CUDA, 否则 CPU"""
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "0"
        return "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="NEU-DET 缺陷检测训练")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="预训练权重或模型配置 (默认 yolo11s.pt)")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置 (默认 data.yaml)")
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数 (默认 150)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认 640)")
    parser.add_argument("--batch", type=int, default=-1, help="批大小, -1 自动选择 (默认 -1)")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值 (默认 50)")
    parser.add_argument("--project", type=str, default="", help="输出目录前缀, 默认空= runs/detect/ 下 (相对 runs/<task>/)")
    parser.add_argument("--name", type=str, default="train", help="本次训练名称 (默认 train)")
    parser.add_argument("--device", type=str, default=None, help="设备: 0=GPU, cpu=CPU (默认自动选择)")
    parser.add_argument("--resume", type=str, default=None, help="断点续训: last.pt 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数 (默认 8)")
    parser.add_argument("--no-cuda", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    # 确定设备
    device = args.device if args.device else pick_device(not args.no_cuda)
    print(f"[train] 设备: {device}")

    # 断点续训: 直接加载 last.pt
    if args.resume:
        print(f"[train] 从 {args.resume} 断点续训")
        model = YOLO(args.resume)
        model.train(
            resume=True,
            project=args.project,
            name=args.name,
        )
        return

    # 全新训练
    model = YOLO(args.model)
    print(f"[train] 模型: {args.model}, 数据: {args.data}")
    print(f"[train] 参数: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, patience={args.patience}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=device,
        workers=args.workers,
        seed=args.seed,
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(args.project) / args.name
    print(f"\n[train] 训练完成!")
    print(f"[train] 最佳权重: {save_dir / 'weights' / 'best.pt'}")
    print(f"[train] 训练曲线: {save_dir / 'results.png'}")


if __name__ == "__main__":
    main()
