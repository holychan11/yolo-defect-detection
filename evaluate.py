"""
evaluate.py
===========
NEU-DET 钢材表面缺陷检测 - 模型评估脚本

用法:
    python evaluate.py                                        # 评估最新 best.pt
    python evaluate.py --weights runs/detect/train/weights/best.pt
    python evaluate.py --split val                            # 在验证集上评估

说明:
    - 在 val 集上计算 mAP50 / mAP50-95 / precision / recall
    - 输出每类详细指标与混淆矩阵图
    - 结果保存在 runs/detect/valN/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def find_best_weight() -> Path | None:
    """在 runs/detect 下查找最新的 best.pt"""
    runs = Path("runs/detect")
    if not runs.exists():
        return None
    candidates = sorted(runs.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="NEU-DET 模型评估")
    parser.add_argument("--weights", type=str, default=None, help="模型权重 (默认自动找最新 best.pt)")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置 (默认 data.yaml)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"], help="评估分区 (默认 val)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="评估用置信度阈值 (默认 0.001, 求全量 PR 曲线)")
    parser.add_argument("--iou", type=float, default=0.6, help="评估用 IoU 阈值 (默认 0.6)")
    parser.add_argument("--device", type=str, default=None, help="设备: 0=GPU, cpu=CPU (默认自动)")
    parser.add_argument("--project", type=str, default="", help="输出目录前缀, 默认空= runs/detect/ 下")
    parser.add_argument("--name", type=str, default="val", help="输出名称")
    args = parser.parse_args()

    weights = args.weights or (str(find_best_weight()) if find_best_weight() else "yolo11s.pt")
    print(f"[eval] 权重: {weights}")

    model = YOLO(weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        exist_ok=True,
    )

    print("\n==================== 评估结果 ====================")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print("--------------------------------------------------")
    print("  各类别指标 (P / R / mAP50):")
    for i, name in enumerate(CLASS_NAMES):
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        print(f"    {name:<20} P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}")
    print("==================================================")

    save_dir = Path(args.project) / args.name
    print(f"[eval] 混淆矩阵与曲线已保存至: {save_dir}")


if __name__ == "__main__":
    main()
