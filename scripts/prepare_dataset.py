"""Offline dataset preparation scaffold for BISINDO sign recognition.

Raw videos should be sampled, processed with ROI detection/MediaPipe landmarks,
normalized, then saved as fixed-length sequences for model training.
"""

from pathlib import Path


DATASET_ROOT = Path("datasets/bisindo")
OUTPUT_ROOT = Path("datasets/processed")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Dataset preparation scaffold.")
    print(f"Input: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print("Implement frame sampling + MediaPipe extraction before training.")


if __name__ == "__main__":
    main()
