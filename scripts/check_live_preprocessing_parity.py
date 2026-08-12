"""Compare direct classifier output with FastAPI /predict-sign preprocessing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.assessment_service.assessment_service.inference.bisindo_classifier import BisindoYoloClassifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--service-url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/preprocessing_parity.json"))
    return parser.parse_args()


def top5_direct(image_path):
    classifier = BisindoYoloClassifier()
    classifier.load()
    with image_path.open("rb") as handle:
        result = classifier.predict(handle.read())
    return result.as_dict()["raw_predictions"][:5]


def top5_encoded(image_path, fmt, quality=None):
    classifier = BisindoYoloClassifier()
    classifier.load()
    image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    if fmt == "JPEG":
        image.save(buffer, format=fmt, quality=quality)
    else:
        image.save(buffer, format=fmt)
    result = classifier.predict(buffer.getvalue())
    return result.as_dict()["raw_predictions"][:5]


def call_service(image_path, service_url):
    with image_path.open("rb") as handle:
        response = requests.post(
            f"{service_url.rstrip('/')}/predict-sign",
            files={"image": (image_path.name, handle.read(), "image/jpeg")},
            data={"frame_id": "parity", "mirrored": "false", "hands_detected": "1", "source_width": "0", "source_height": "0"},
            timeout=30,
        )
    response.raise_for_status()
    payload = response.json()
    return payload.get("raw_predictions", [])[:5], payload.get("classifier_debug", {})


def main():
    args = parse_args()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.assessment_service.assessment_service.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            args.service_url.rsplit(":", 1)[-1],
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                requests.get(f"{args.service_url.rstrip('/')}/health", timeout=1).raise_for_status()
                break
            except Exception:
                time.sleep(0.5)
        direct = top5_direct(args.image)
        jpeg_88 = top5_encoded(args.image, "JPEG", quality=88)
        jpeg_95 = top5_encoded(args.image, "JPEG", quality=95)
        png = top5_encoded(args.image, "PNG")
        service, debug = call_service(args.image, args.service_url)
    finally:
        process.terminate()
        process.wait(timeout=10)
    payload = {
        "image": str(args.image),
        "direct_top5": direct,
        "jpeg_88_top5": jpeg_88,
        "jpeg_95_top5": jpeg_95,
        "png_top5": png,
        "fastapi_top5": service,
        "fastapi_debug": debug,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
