"""Download VinText dataset (COCO format) from VinAI Research Google Drive.

Usage:
    python src/download_vintext.py

Dataset will be extracted to: datasets/vintext/
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# VinAI dict-guided repo - VinText dataset (Original format with REAL Vietnamese text)
# https://github.com/VinAIResearch/dict-guided
# Original (non-COCO) has real Vietnamese transcripts in format:
#   x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT
# COCO-format version uses pseudo-character tokens (not real Vietnamese).
VINTEXT_GDRIVE_ID = "1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml"
EXPECTED_DIR_NAME = "vintext"


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """Download a file from Google Drive using gdown (handles confirm tokens)."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(destination), quiet=False)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    datasets_dir = root / "datasets"
    target_dir = datasets_dir / EXPECTED_DIR_NAME
    zip_path = datasets_dir / "vintext.zip"

    if target_dir.exists() and (target_dir / "train.json").exists():
        print(f"[skip] VinText already extracted at {target_dir}")
        return 0

    print(f"[1/3] Downloading VinText (COCO format) from Google Drive -> {zip_path}")
    download_file_from_google_drive(VINTEXT_GDRIVE_ID, zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"      Downloaded: {size_mb:.1f} MB")

    print(f"[2/3] Extracting -> {datasets_dir}")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(datasets_dir)

    # Some zips nest under a folder; rename to 'vintext' if needed
    extracted_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]
    if not target_dir.exists() and extracted_dirs:
        # Use the first subdirectory as the source
        src = extracted_dirs[0]
        if src.name != EXPECTED_DIR_NAME:
            print(f"[3/3] Renaming {src.name} -> {EXPECTED_DIR_NAME}")
            src.rename(target_dir)
    elif not target_dir.exists():
        # Files extracted directly to datasets_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    zip_path.unlink(missing_ok=True)
    print(f"[ok] VinText ready at: {target_dir}")
    items = sorted(p.name for p in target_dir.iterdir())
    print(f"      Contents: {items}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
