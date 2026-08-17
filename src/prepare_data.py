"""Convert VinText Original format (PaddleOCR/VinAI style) to VietOCR training format.

VinText ground-truth format (file per image `labels/gt_<id>.txt`):
    x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT
    x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT
    ...

Where TRANSCRIPT is the real Vietnamese text (UTF-8). '###' means skip.

This script handles actual Vietnamese UTF-8 properly by reading files as bytes
and decoding explicitly as UTF-8.

Output:
    datasets/vintext_processed/<split>_images/<img>_<idx>.png
    datasets/vintext_processed/<split>_labels/<img>_<idx>.txt
    datasets/vintext_processed/train.txt
    datasets/vintext_processed/val.txt
    datasets/vintext_processed/test.txt
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VINTEXT_DIR = PROJECT_ROOT / "datasets" / "vintext"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "vintext_processed"

SKIP_TOKEN = "###"
MIN_CROP_PIXELS = 5

# Split -> images dir name.
# VinText Original:
#   train_images/  : im0001.jpg .. im1200.jpg (1200 train)
#   test_image/    : im1201.jpg .. im1500.jpg (300 val)
#   unseen_test_images/ : im1501.jpg .. im2000.jpg (500 test)
SPLIT_IMAGE_DIR = {
    "train": "train_images",
    "val": "test_image",
    "test": "unseen_test_images",
}
SPLIT_LABEL_DIR = "labels"

LINE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),"
                      r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),"
                      r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),")


def order_corners_tl_tr_br_bl(poly: list[float]) -> np.ndarray:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_text_instance(img: np.ndarray, poly: list[float]) -> np.ndarray | None:
    """Crop quadrilateral region with perspective transform."""
    try:
        corners = order_corners_tl_tr_br_bl(poly)
    except Exception:
        return None

    w_top = np.linalg.norm(corners[1] - corners[0])
    w_bot = np.linalg.norm(corners[2] - corners[3])
    h_left = np.linalg.norm(corners[3] - corners[0])
    h_right = np.linalg.norm(corners[2] - corners[1])

    width = int(max(w_top, w_bot))
    height = int(max(h_left, h_right))

    if width < MIN_CROP_PIXELS or height < MIN_CROP_PIXELS:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    crop = cv2.warpPerspective(img, M, (width, height))

    # Pad if too narrow for VietOCR (needs >= 32 px width)
    if crop.shape[1] < 32 and crop.shape[0] > 0:
        pad = 32 - crop.shape[1]
        crop = cv2.copyMakeBorder(crop, 0, 0, pad // 2, pad - pad // 2,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return crop


def find_split_dirs(split: str) -> tuple[Path | None, Path | None]:
    """Find image dir and labels dir for the given split."""
    img_dir = VINTEXT_DIR / SPLIT_IMAGE_DIR[split]
    lbl_dir = VINTEXT_DIR / SPLIT_LABEL_DIR
    return (
        img_dir if img_dir.exists() else None,
        lbl_dir if lbl_dir.exists() else None,
    )


def parse_gt_file(path: Path) -> list[tuple[list[int], str]]:
    """Parse one gt_*.txt file. Returns list of (8 coords, transcript).

    Reads as bytes + UTF-8 decode with errors='replace' to handle any
    encoding quirks on Windows.
    """
    raw = path.read_bytes().decode("utf-8", errors="replace")
    entries: list[tuple[list[int], str]] = []
    for line in raw.splitlines():
        line = line.rstrip("\r\n")
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        coords = [int(float(x)) for x in m.groups()]
        transcript = line[m.end():].lstrip(",").strip()
        if not transcript:
            continue
        entries.append((coords, transcript))
    return entries


def parse_image_id_from_label_filename(label_path: Path) -> int | None:
    """Labels are 'gt_1.txt', 'gt_10.txt' etc. Extract the integer id."""
    stem = label_path.stem  # 'gt_1'
    m = re.match(r"gt_(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def find_image_for_label(image_id: int, img_dir: Path) -> Path | None:
    """Look for an image file matching the label id (handles multiple naming schemes)."""
    # Common naming: im<id>.jpg
    candidates = [
        img_dir / f"im{image_id}.jpg",
        img_dir / f"im{image_id:04d}.jpg",
        img_dir / f"{image_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: scan dir for any number-prefixed match
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            for prefix in [f"im{image_id}.", f"im{image_id:04d}.", f"{image_id}."]:
                if p.name.lower().startswith(prefix):
                    return p
    return None


def convert_split(split: str, img_dir: Path, lbl_dir: Path) -> tuple[int, int, int]:
    """Convert one split.

    The labels are usually all in one shared `labels/` dir across splits, but
    to avoid mixing, we use ALL files whose id falls into that split's range.

    Returns (emitted, skipped, missing_image).
    """
    img_out_dir = OUTPUT_DIR / f"{split}_images"
    lbl_out_dir = OUTPUT_DIR / f"{split}_labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    emitted = 0
    skipped = 0
    missing_img = 0

    label_files = sorted(lbl_dir.glob("gt_*.txt"))
    if not label_files:
        print(f"[warn] no gt_*.txt files in {lbl_dir}")
        return 0, 0, 0

    for label_file in tqdm(label_files, desc=f"convert {split}"):
        img_id = parse_image_id_from_label_filename(label_file)
        if img_id is None:
            skipped += 1
            continue
        img_path = find_image_for_label(img_id, img_dir)
        if img_path is None or not img_path.exists():
            missing_img += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_img += 1
            continue

        entries = parse_gt_file(label_file)
        if not entries:
            skipped += 1
            continue

        for idx, (coords, transcript) in enumerate(entries):
            if transcript == SKIP_TOKEN or not transcript.strip():
                skipped += 1
                continue

            crop = crop_text_instance(img, coords)
            if crop is None:
                skipped += 1
                continue

            stem = f"{img_id:06d}_{idx:03d}"
            out_img = img_out_dir / f"{stem}.png"
            out_lbl = lbl_out_dir / f"{stem}.txt"
            cv2.imwrite(str(out_img), crop)
            out_lbl.write_text(transcript, encoding="utf-8")
            manifest.append(f"{split}_images/{stem}.png\t{transcript}")
            emitted += 1

    # Write manifest
    (OUTPUT_DIR / f"{split}.txt").write_text(
        "\n".join(manifest), encoding="utf-8"
    )
    return emitted, skipped, missing_img


def main() -> int:
    if not VINTEXT_DIR.exists():
        print(f"[err] VinText not found at {VINTEXT_DIR}")
        print("      Run: python src/download_vintext.py")
        return 1

    print(f"VinText dir: {VINTEXT_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_emitted = 0
    total_skipped = 0
    total_missing = 0

    for split in ["train", "val", "test"]:
        img_dir, lbl_dir = find_split_dirs(split)
        if not img_dir or not lbl_dir:
            print(f"[warn] directories not found for '{split}', skipping")
            continue
        print(f"  -> {split}: images={img_dir}, labels={lbl_dir}")
        emitted, skipped, missing = convert_split(split, img_dir, lbl_dir)
        print(f"[{split}] emitted={emitted}, skipped={skipped}, missing={missing}")
        total_emitted += emitted
        total_skipped += skipped
        total_missing += missing

    print(f"\n[done] total emitted={total_emitted}, skipped={total_skipped}, "
          f"missing_img={total_missing}")
    print(f"Manifests: {OUTPUT_DIR}/train.txt, val.txt, test.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
