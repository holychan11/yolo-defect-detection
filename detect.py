"""
detect.py
=========
NEU-DET 钢材表面缺陷检测 - 推理脚本

用法:
    python detect.py images/test.jpg                    # 检测单张图片
    python detect.py images/                            # 检测整个文件夹
    python detect.py video.mp4                          # 检测视频
    python detect.py 0                                  # 摄像头实时检测
    python detect.py --weights runs/detect/train/weights/best.pt images/test.jpg
    python detect.py --conf 0.3 --iou 0.5 images/test.jpg

说明:
    - 默认加载训练好的 best.pt (若存在), 否则回退到 yolo11s.pt
    - 结果保存在 runs/detect/predictN/ 目录
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = {
    0: "crazing(裂纹)",
    1: "inclusion(夹杂)",
    2: "patches(麻点)",
    3: "pitted_surface(氧化皮)",
    4: "rolled-in_scale(轧制氧化皮)",
    5: "scratches(划痕)",
}


def find_best_weight() -> Path | None:
    """在 runs/detect 下查找最新的 best.pt"""
    runs = Path("runs/detect")
    if not runs.exists():
        return None
    candidates = sorted(runs.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="NEU-DET 缺陷检测推理")
    parser.add_argument("source", nargs="?", default="test_sample.jpg", help="图片/文件夹/视频路径, 或 0 表示摄像头")
    parser.add_argument("--weights", type=str, default=None, help="模型权重 (默认自动找 best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值 (默认 0.45)")
    parser.add_argument("--project", type=str, default="", help="输出目录前缀, 默认空= runs/detect/ 下 (相对 runs/<task>/)")
    parser.add_argument("--name", type=str, default="predict", help="输出名称 (默认 predict)")
    parser.add_argument("--device", type=str, default=None, help="设备: 0=GPU, cpu=CPU (默认自动)")
    parser.add_argument("--save-txt", action="store_true", help="同时保存标签文件")
    args = parser.parse_args()

    # 确定权重: 优先指定, 否则找 best.pt, 最后回退预训练
    weights = args.weights
    if weights is None:
        best = find_best_weight()
        if best:
            weights = str(best)
            print(f"[detect] 使用训练好的模型: {weights}")
        else:
            weights = "yolo11s.pt"
            print("[detect] 未找到 best.pt, 回退到 yolo11s.pt (COCO 预训练)")

    model = YOLO(weights)
    print(f"[detect] 加载模型: {weights}")
    print(f"[detect] 参数: conf={args.conf}, iou={args.iou}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        show_labels=True,
        show_conf=True,
        exist_ok=True,
    )

    # 打印每张图的检测结果 (类别名 + 置信度)
    for r in results:
        path = Path(r.path)
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {path.name}: 未检测到缺陷")
            continue
        print(f"  {path.name}: 检测到 {len(boxes)} 个缺陷")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            xyxy = [round(float(v), 1) for v in box.xyxy[0]]
            print(f"    - {name}: 置信度 {conf:.2f}, 坐标 {xyxy}")

    save_dir = Path(args.project) / args.name
    print(f"\n[detect] 结果保存于: {save_dir}")


if __name__ == "__main__":
    main()
