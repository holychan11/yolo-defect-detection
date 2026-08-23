"""
export.py
=========
将训练好的模型导出为 ONNX 格式 (部署/加速推理用)

用法:
    python export.py                                        # 导出最新 best.pt -> ONNX
    python export.py --weights runs/detect/train/weights/best.pt
    python export.py --format onnx --imgsz 640              # 指定格式与输入尺寸

支持格式: onnx, torchscript, openvino, engine (TensorRT), coreml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def find_best_weight() -> Path | None:
    """在 runs/detect 下查找最新的 best.pt"""
    runs = Path("runs/detect")
    if not runs.exists():
        return None
    candidates = sorted(runs.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="导出模型为 ONNX 等格式")
    parser.add_argument("--weights", type=str, default=None, help="模型权重 (默认自动找最新 best.pt)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "openvino", "engine", "coreml"], help="导出格式 (默认 onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    parser.add_argument("--simplify", action="store_true", default=True, help="ONNX 图简化 (默认开)")
    parser.add_argument("--half", action="store_true", help="导出 FP16 (TensorRT 用, 需 GPU)")
    args = parser.parse_args()

    weights = args.weights or (str(find_best_weight()) if find_best_weight() else "yolo11s.pt")
    print(f"[export] 权重: {weights}")
    print(f"[export] 格式: {args.format}, imgsz={args.imgsz}")

    model = YOLO(weights)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
    )
    print(f"[export] 导出完成: {out}")


if __name__ == "__main__":
    main()
