"""Layout analysis: split a preprocessed image into text blocks and tables.

MVP heuristic (no heavy model):
    1. Convert to grayscale.
    2. Detect horizontal & vertical lines using morphological operations.
    3. If lines form a grid -> table region.
    4. Otherwise, segment text blocks via horizontal projection (whitespace gaps).

Example:
    layout = detect_layout(image_path)
    for region in layout.regions:
        crop = region.crop(image)
        if region.type == RegionType.TEXT:
            vietocr.predict(crop)
        elif region.type == RegionType.TABLE:
            # reconstruct grid in reconstruct.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config_loader import get  # noqa: E402


class RegionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass
class Region:
    x1: int
    y1: int
    x2: int
    y2: int
    type: RegionType

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y1:self.y2, self.x1:self.x2].copy()


@dataclass
class Layout:
    image_path: str
    regions: list[Region]

    @property
    def text_regions(self) -> list[Region]:
        return [r for r in self.regions if r.type == RegionType.TEXT]

    @property
    def table_regions(self) -> list[Region]:
        return [r for r in self.regions if r.type == RegionType.TABLE]


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold for varied lighting."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def _detect_lines(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (horizontal_mask, vertical_mask) using morphology."""
    h, w = binary.shape
    h_kernel_len = max(40, w // 30)
    v_kernel_len = max(40, h // 30)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Hough line detection for stronger line signal
    edges = cv2.Canny(binary, 50, 150)
    hough = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120,
                            minLineLength=h_kernel_len, maxLineGap=10)
    if hough is not None:
        segs = hough[:, 0, :] if hough.ndim == 3 else hough
        mask = np.zeros_like(binary)
        for ln in segs:
            x1, y1, x2, y2 = int(ln[0]), int(ln[1]), int(ln[2]), int(ln[3])
            if abs(y2 - y1) < 3:  # horizontal
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            elif abs(x2 - x1) < 3:  # vertical
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        h_lines = cv2.bitwise_or(h_lines, mask)
    return h_lines, v_lines


def _lines_to_bbox(h_lines: np.ndarray, v_lines: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Combine line masks and find bounding boxes of line intersections."""
    combined = cv2.bitwise_or(h_lines, v_lines)
    # Dilate to merge nearby lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(combined, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge overlapping boxes (union)."""
    if not boxes:
        return []
    arr = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = [arr[0]]
    for cur in arr[1:]:
        last = merged[-1]
        if cur[0] <= last[2] and cur[1] <= last[3]:
            merged[-1] = (min(last[0], cur[0]), min(last[1], cur[1]),
                          max(last[2], cur[2]), max(last[3], cur[3]))
        else:
            merged.append(cur)
    return merged


def _is_table(h_count: int, v_count: int, area: int, img_area: int) -> bool:
    """Heuristic: a region is a table if it has >=2 horizontal AND >=2 vertical lines."""
    min_area_ratio = float(get("layout.min_table_area_ratio", 0.02))
    if area < img_area * min_area_ratio:
        return False
    return h_count >= 2 and v_count >= 2


def _count_lines_in_crop(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                         axis: str) -> int:
    """Count distinct lines inside a crop of a line mask."""
    crop = mask[y1:y2, x1:x2]
    if axis == "h":
        proj = np.sum(crop > 0, axis=1)
    else:
        proj = np.sum(crop > 0, axis=0)
    # Local maxima separated by gaps
    peaks = 0
    last = -100
    gap = 5
    for i, v in enumerate(proj):
        if v > crop.shape[1] * 0.5 if axis == "h" else v > crop.shape[0] * 0.5:
            if i - last > gap:
                peaks += 1
                last = i
    return peaks


def _segment_text_blocks(gray: np.ndarray, exclude: list[tuple[int, int, int, int]]) -> list[Region]:
    """Segment text blocks using horizontal projection profile.

    `exclude` = list of bboxes (tables) to skip.
    """
    h, w = gray.shape
    binary = _binarize(gray)
    # Zero out table regions to exclude them from text detection
    for (x1, y1, x2, y2) in exclude:
        binary[y1:y2, x1:x2] = 0

    # Horizontal projection
    row_sum = np.sum(binary, axis=1)
    line_threshold = w * 0.02 * 255
    in_block = False
    blocks: list[tuple[int, int, int, int]] = []
    start = 0
    gap_threshold = 8  # px gap to break lines
    last_ink = -100
    for i, v in enumerate(row_sum):
        has_ink = v > line_threshold
        if has_ink:
            if not in_block:
                start = i
                in_block = True
            last_ink = i
        else:
            if in_block and i - last_ink > gap_threshold:
                blocks.append((0, start, w, last_ink))
                in_block = False
    if in_block:
        blocks.append((0, start, w, last_ink))

    # Merge vertically adjacent lines into blocks (gap < line_merge_gap_px)
    gap = int(get("layout.line_merge_gap_px", 20))
    merged: list[tuple[int, int, int, int]] = []
    for b in blocks:
        if merged and b[1] - merged[-1][3] < gap:
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1], prev[2], b[3])
        else:
            merged.append(b)
    return [
        Region(x1=b[0], y1=b[1], x2=b[2], y2=b[3], type=RegionType.TEXT)
        for b in merged
    ]


def detect_layout(image_path: str | Path) -> Layout:
    """Run heuristic layout detection on a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return Layout(image_path=str(image_path), regions=[])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    binary = _binarize(gray)
    h_lines, v_lines = _detect_lines(binary)

    boxes = _lines_to_bbox(h_lines, v_lines)
    boxes = _merge_overlapping(boxes)

    table_regions: list[Region] = []
    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        h_count = _count_lines_in_crop(h_lines, x1, y1, x2, y2, "h")
        v_count = _count_lines_in_crop(v_lines, x1, y1, x2, y2, "v")
        if _is_table(h_count, v_count, area, img_area):
            table_regions.append(Region(x1, y1, x2, y2, RegionType.TABLE))

    text_regions = _segment_text_blocks(gray, exclude=[r.bbox for r in table_regions])

    # Sort by reading order (top-to-bottom, left-to-right)
    regions = sorted(text_regions + table_regions, key=lambda r: (r.y1, r.x1))
    return Layout(image_path=str(image_path), regions=regions)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Detect layout regions")
    parser.add_argument("--input", required=True, help="Path to image")
    args = parser.parse_args()
    layout = detect_layout(args.input)
    print(f"Layout for {args.input}")
    for r in layout.regions:
        print(f"  {r.type.value:8s} bbox={r.bbox}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
