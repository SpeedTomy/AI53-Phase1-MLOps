from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

# Milestone 6: a comment changes the code hash, not the deterministic quality report.
EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Reject invalid data and return stable counts after fully decoding images."""
    report = {"status": "PASS", "classes": sorted(EXPECTED_CLASSES),
              "splits": {}, "total_images": 0}
    for split, expected_count in sorted(EXPECTED_SPLITS.items()):
        split_dir = raw_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        actual_classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        if actual_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {split}: {sorted(actual_classes)}; "
                f"expected {sorted(EXPECTED_CLASSES)}"
            )
        class_counts = {}
        for class_name in sorted(EXPECTED_CLASSES):
            images = sorted(
                path for path in (split_dir / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise RuntimeError(f"Empty class: {split}/{class_name}")
            for path in images:
                with Image.open(path) as image:
                    image.load()
            class_counts[class_name] = len(images)
        count = sum(class_counts.values())
        if count != expected_count:
            raise RuntimeError(
                f"Unexpected image count in {split}: {count}; expected {expected_count}"
            )
        report["splits"][split] = {"images": count, "class_counts": class_counts}
        report["total_images"] += count
    return report


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
