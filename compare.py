"""
compare.py
==========
新旧模型对比评估 (在同一验证集上公平对比)

用法:
    python compare.py
    python compare.py --new runs/detect/train-new/weights/best.pt --old runs/detect/train-4/weights/best.pt

说明:
    - 默认: 新模型 = 最新的 train*/best.pt, 旧模型 = train-4/best.pt
    - 两者都在 data.yaml 的 val 集 (360 张) 上评估, 结果公平可比
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def find_latest_best(prefix: str) -> Path | None:
    """在 runs/detect 下查找最新的 train*/weights/best.pt"""
    runs = Path("runs/detect")
    if not runs.exists():
        return None
    candidates = sorted(
        runs.glob(f"{prefix}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="新旧模型对比评估")
    parser.add_argument("--new", type=str, default=None, help="新模型权重 (默认自动找最新 best.pt)")
    parser.add_argument("--old", type=str, default="runs/detect/train-4/weights/best.pt", help="旧模型权重")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="评估置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="评估 IoU 阈值")
    parser.add_argument("--device", type=str, default=None, help="设备 (默认自动)")
    args = parser.parse_args()

    new_weights = args.new or (str(find_latest_best("train")) if find_latest_best("train") else None)
    if new_weights is None:
        raise SystemExit("未找到新模型 best.pt, 请先训练或指定 --new")

    print(f"[compare] 新模型: {new_weights}")
    print(f"[compare] 旧模型: {args.old}")

    results = {}
    for label, w in [("new", new_weights), ("old", args.old)]:
        if not Path(w).exists():
            print(f"[compare] {label} 权重不存在, 跳过: {w}")
            continue
        model = YOLO(w)
        m = model.val(
            data=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project="",
            name=f"compare-{label}",
            plots=True,
            exist_ok=True,
        )
        results[label] = {
            "precision": m.box.mp,
            "recall": m.box.mr,
            "map50": m.box.map50,
            "map50_95": m.box.map,
            "per_class_ap50": [float(x) for x in m.box.ap50],
        }
        print(f"[compare] {label} 评估完成: mAP50={m.box.map50:.4f}, mAP50-95={m.box.map:.4f}")

    if len(results) < 2:
        print("[compare] 不足两个模型, 无法对比")
        return

    print("\n==================== 对比结果 ====================")
    print(f"{'指标':<12} {'新模型':>10} {'旧模型':>10} {'提升':>10}")
    print("-" * 44)
    for key, name in [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP50"), ("map50_95", "mAP50-95")]:
        nv = results["new"][key]
        ov = results["old"][key]
        diff = nv - ov
        print(f"{name:<12} {nv:>10.4f} {ov:>10.4f} {diff:>+10.4f}")
    print("-" * 44)
    print("  各类别 mAP50 (新 vs 旧):")
    for i, cls in enumerate(CLASS_NAMES):
        nv = results["new"]["per_class_ap50"][i]
        ov = results["old"]["per_class_ap50"][i]
        diff = nv - ov
        print(f"    {cls:<20} {nv:.3f} vs {ov:.3f}  ({diff:+.3f})")
    print("==================================================")

    # 保存对比结果 CSV
    out = Path("runs/detect/compare_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "new", "old", "diff"])
        for key, name in [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP50"), ("map50_95", "mAP50-95")]:
            writer.writerow([name, results["new"][key], results["old"][key], results["new"][key] - results["old"][key]])
    print(f"[compare] 结果已保存: {out}")


if __name__ == "__main__":
    main()
