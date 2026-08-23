"""
split_dataset.py
================
重新划分 NEU-DET 数据集为 8:2 (train:val)，保证每类都有足够验证样本。

- 原划分: train 1770 / val 30 (每类仅 5 张验证图, 评估不可靠)
- 新划分: train 1440 / val 360 (每类 240 训练 + 60 验证)
- 固定随机种子 (seed=42), 可复现

安全策略: 先复制到临时目录 -> 校验数量 -> 再原子替换, 中途失败不会丢失原数据。

用法:
    python split_dataset.py            # 默认使用 dataset/ 目录
    python split_dataset.py --seed 7   # 自定义随机种子
"""

import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def class_of(filename: str) -> str:
    """从文件名推断类别: crazing_1.jpg -> crazing; pitted_surface_1.jpg -> pitted_surface"""
    for cls in CLASSES:
        if filename.startswith(cls + "_"):
            return cls
    raise ValueError(f"无法识别类别: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重新划分 NEU-DET 数据集 (8:2)")
    parser.add_argument("--root", type=str, default="dataset", help="数据集根目录 (含 images/ 和 labels/)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (默认 0.2)")
    args = parser.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    # 1. 收集所有图片(跨 train/val 合并) 及其标签
    all_images: list[Path] = []
    for split in ("train", "val"):
        all_images.extend((images_dir / split).glob("*.jpg"))
    all_images = sorted(all_images)
    print(f"共收集图片: {len(all_images)} 张")

    # 2. 按类别分组
    by_class: dict[str, list[Path]] = {cls: [] for cls in CLASSES}
    for img in all_images:
        by_class[class_of(img.name)].append(img)
    for cls, imgs in by_class.items():
        print(f"  {cls}: {len(imgs)} 张")
        if len(imgs) < 2:
            raise RuntimeError(f"类别 {cls} 样本太少: {len(imgs)}")

    # 3. 每类随机打乱后按比例划分
    rng = random.Random(args.seed)
    plan: dict[str, dict[str, list[Path]]] = {}
    for cls, imgs in by_class.items():
        shuffled = imgs[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(imgs) * args.val_ratio))
        plan[cls] = {"train": shuffled[n_val:], "val": shuffled[:n_val]}

    n_train = sum(len(plan[c]["train"]) for c in CLASSES)
    n_val = sum(len(plan[c]["val"]) for c in CLASSES)
    print(f"\n划分计划: train {n_train} / val {n_val} (val_ratio={args.val_ratio})")

    # 4. 先复制到临时目录
    tmp = root / "_new_split"
    if tmp.exists():
        shutil.rmtree(tmp)
    for split in ("train", "val"):
        (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing_label = 0
    for cls in CLASSES:
        for split in ("train", "val"):
            for img in plan[cls][split]:
                src_split = img.parent.name  # 原所在分区 train/val
                lbl = labels_dir / src_split / img.with_suffix(".txt").name
                shutil.copy2(str(img), str(tmp / "images" / split / img.name))
                if lbl.exists():
                    shutil.copy2(str(lbl), str(tmp / "labels" / split / lbl.name))
                else:
                    missing_label += 1

    # 5. 校验临时目录数量
    t_train = len(list((tmp / "images" / "train").glob("*.jpg")))
    t_val = len(list((tmp / "images" / "val").glob("*.jpg")))
    l_train = len(list((tmp / "labels" / "train").glob("*.txt")))
    l_val = len(list((tmp / "labels" / "val").glob("*.txt")))
    print(f"临时目录校验: train {t_train}图/{l_train}标签, val {t_val}图/{l_val}标签")
    if t_train != n_train or t_val != n_val or l_train != n_train or l_val != n_val:
        shutil.rmtree(tmp)
        raise RuntimeError("临时目录校验失败, 已回滚, 原数据未动")

    # 6. 原子替换: 清空旧目录, 移入新文件
    for split in ("train", "val"):
        for d in (images_dir / split, labels_dir / split):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    for split in ("train", "val"):
        shutil.copytree(tmp / "images" / split, images_dir / split, dirs_exist_ok=True)
        shutil.copytree(tmp / "labels" / split, labels_dir / split, dirs_exist_ok=True)
    shutil.rmtree(tmp)

    print(f"替换完成: 图片 {t_train + t_val}, 标签 {l_train + l_val}, 缺失标签 {missing_label}")
    print("\n新划分统计:")
    for cls in CLASSES:
        print(f"  {cls}: train {len(plan[cls]['train'])} / val {len(plan[cls]['val'])}")


if __name__ == "__main__":
    main()
