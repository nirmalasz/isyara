"""Train a YOLO11 image classification experiment on TERBISA crops."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/terbisa-cls"))
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", default="16")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", type=Path, default=Path("runs/classify"))
    parser.add_argument("--name", default="bisindo_words_cls")
    parser.add_argument("--output", type=Path, default=Path("models/bisindo_words_cls.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    batch = int(args.batch) if str(args.batch).isdigit() else args.batch
    model = YOLO(args.model)
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"Best weights: {best}")
    print(f"Copied to: {args.output}")


if __name__ == "__main__":
    main()
