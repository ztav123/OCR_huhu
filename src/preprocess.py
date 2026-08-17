"""Image preprocessing to reduce Grok token usage while keeping OCR quality.

Key responsibilities:
    - Auto-rotate via EXIF orientation
    - Deskew (Hough line / minAreaRect based)
    - Denoise (fastNlMeansDenoising)
    - Resize: keep aspect ratio, max edge <= 1568 px (sweet spot for Grok 448 tiles)
    - Encode as JPEG quality 85

Output:
    - Single-image API: preprocess_image(path) -> (np.ndarray, meta_dict)
    - Batch-first file-RUN: process_directory(input_dir, output_dir) -> Manifest
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config_loader import get, load_config  # noqa: E402

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PreprocessMeta:
    source_path: str
    output_path: str
    sha256: str
    original_size: list[int]      # [w, h]
    processed_size: list[int]     # [w, h]
    rotated_deg: float
    file_size_bytes: int


def _apply_exif_rotation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation from PIL."""
    out = ImageOps.exif_transpose(img)
    if isinstance(out, int):
        # Some Pillow versions return the transpose code instead of the image
        return img
    return out


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle using Hough lines.

    Returns angle in degrees. Range ~(-45, 45).
    """
    # Downscale for speed
    h, w = gray.shape
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray

    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=200,
                            minLineLength=small.shape[1] // 6, maxLineGap=20)
    if lines is None:
        return 0.0

    # Normalize shape: works for both (N, 1, 4) and (N, 4)
    if lines.ndim == 3:
        segs = lines[:, 0, :]
    else:
        segs = lines

    angles = []
    for ln in segs:
        x1, y1, x2, y2 = float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew image; returns (rotated, angle_deg)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return img, 0.0
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def _resize_max_edge(img: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _denoise(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(img, h=7)
    return cv2.fastNlMeansDenoisingColored(img, h=7, hColor=7, templateWindowSize=7,
                                            searchWindowSize=21)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_image(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    max_edge: int | None = None,
    jpeg_quality: int | None = None,
    deskew: bool = True,
    denoise: bool = True,
    auto_rotate_exif: bool = True,
) -> PreprocessMeta:
    """Preprocess a single image and write to output_dir.

    Returns metadata for downstream stages.
    """
    cfg = load_config()
    max_edge = int(max_edge or get("preprocess.max_edge_px", 1568))
    jpeg_quality = int(jpeg_quality or get("preprocess.jpeg_quality", 85))
    deskew = bool(deskew if deskew is not None
                  else get("preprocess.deskew", True))
    denoise = bool(denoise if denoise is not None
                   else get("preprocess.denoise", True))
    auto_rotate_exif = bool(auto_rotate_exif
                             if auto_rotate_exif is not None
                             else get("preprocess.auto_rotate_exif", True))

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(source_path)
    if auto_rotate_exif:
        pil = _apply_exif_rotation(pil)
    img = _pil_to_cv(pil)

    rotated_deg = 0.0
    if deskew:
        img, rotated_deg = _deskew(img)
    if denoise:
        img = _denoise(img)
    img = _resize_max_edge(img, max_edge=max_edge)

    out_name = source_path.stem + ".jpg"
    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    meta = PreprocessMeta(
        source_path=str(source_path.resolve()),
        output_path=str(out_path.resolve()),
        sha256=_sha256_file(out_path),
        original_size=[pil.width, pil.height],
        processed_size=[img.shape[1], img.shape[0]],
        rotated_deg=round(rotated_deg, 3),
        file_size_bytes=out_path.stat().st_size,
    )
    return meta


def process_directory(input_dir: str | Path, output_dir: str | Path) -> list[PreprocessMeta]:
    """Preprocess all supported images in input_dir; write manifest of metadata."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"[preprocess] no images found in {input_dir}")
        return []

    metas: list[PreprocessMeta] = []
    for path in tqdm(files, desc="preprocess"):
        try:
            m = preprocess_image(path, output_dir)
            metas.append(m)
        except Exception as e:
            print(f"[warn] failed to preprocess {path}: {e}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(m) for m in metas], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[preprocess] {len(metas)} images -> {output_dir}")
    print(f"             manifest: {manifest_path}")
    return metas


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess images for OCR")
    parser.add_argument("--input", required=True, help="Input folder of images")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()
    process_directory(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
