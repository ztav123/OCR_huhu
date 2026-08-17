"""Main CLI entry point for Vietnamese OCR Pipeline.

Pipeline:
    preprocess  ->  layout  ->  vietocr (+grok fallback)  ->  reconstruct

Usage:
    python main.py --input data/raw_images --output data/outputs/result.docx
    python main.py --input data/raw_images --output data/outputs/result.docx --no-grok
    python main.py --input data/raw_images --output data/outputs/result.docx --skip-preprocess
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get, load_config  # noqa: E402
from src.grok_client import GrokClient  # noqa: E402
from src.openai_client import OpenAIClient  # noqa: E402
from src.azure_client import AzureClient  # noqa: E402
from src.layout import Region, RegionType, detect_layout  # noqa: E402
from src.preprocess import process_directory  # noqa: E402
from src.recognizer import (  # noqa: E402
    VietOCRRecognizer,
    get_recognizer,
    is_low_confidence,
)
from src.reconstruct import (  # noqa: E402
    RecognizedRegion,
    save_markdown,
    to_docx,
    to_markdown,
)


def _make_fallback_client(
    provider: str, api_key: str | None
) -> object | None:
    """Instantiate fallback client by provider name."""
    if provider == "openai":
        try:
            return OpenAIClient(api_key=api_key)
        except RuntimeError as e:
            print(f"[warn] OpenAI not available: {e}")
            return None
    if provider == "azure":
        try:
            return AzureClient(api_key=api_key)
        except RuntimeError as e:
            print(f"[warn] Azure not available: {e}")
            return None
    if provider == "grok":
        try:
            return GrokClient(api_key=api_key)
        except RuntimeError as e:
            print(f"[warn] Grok not available: {e}")
            return None
    return None


def _process_image(
    image_path: Path,
    recognizer: VietOCRRecognizer,
    fallback: object | None,
    provider_name: str,
    use_fallback: bool,
) -> tuple[list[RecognizedRegion], dict[int, list[RecognizedRegion]], dict]:
    """Run all stages on a single preprocessed image. Returns (recognized, table_subs, stats)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return [], {}, {"vietocr_calls": 0, "fallback_calls": 0,
                        "fallback_cost_usd": 0.0}

    layout = detect_layout(image_path)

    recognized: list[RecognizedRegion] = []
    table_subs: dict[int, list[RecognizedRegion]] = defaultdict(list)
    table_idx_seq = 0
    stats = {"vietocr_calls": 0, "fallback_calls": 0, "fallback_cost_usd": 0.0}

    # Sort by reading order
    for region in sorted(layout.regions, key=lambda r: (r.y1, r.x1)):
        crop = region.crop(img)
        if crop.size == 0:
            continue

        text, conf = recognizer.predict(crop)
        stats["vietocr_calls"] += 1
        source = "vietocr"

        if use_fallback and is_low_confidence(text, conf):
            if fallback is not None:
                try:
                    fr = fallback.recognize(crop)
                    if fr.text:
                        text = fr.text
                    conf = fr.confidence
                    source = provider_name
                    stats["fallback_calls"] += 1
                    stats["fallback_cost_usd"] += fr.usage.estimated_cost_usd
                except Exception as e:
                    print(f"[warn] {provider_name} fallback failed: {e}")

        rec = RecognizedRegion(
            region=region, text=text, confidence=conf, source=source
        )
        if region.type == RegionType.TABLE:
            table_subs[table_idx_seq].append(rec)
            # Also emit placeholder region for table structure
            recognized.append(RecognizedRegion(
                region=Region(0, 0, 0, 0, RegionType.TABLE), text="", confidence=1.0,
                source="layout",
            ))
            table_idx_seq += 1
        else:
            recognized.append(rec)

    return recognized, dict(table_subs), stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Vietnamese OCR pipeline")
    parser.add_argument("--input", required=True,
                        help="Input folder of images (jpg/png/...)")
    parser.add_argument("--output", required=True,
                        help="Output .docx path")
    parser.add_argument("--md", default=None,
                        help="Optional .md path (defaults to <output>.md)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (assume images already in input dir)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable cloud fallback (VietOCR only)")
    parser.add_argument("--provider", default="azure",
                        choices=["openai", "azure", "grok", "none"],
                        help="Fallback provider (default: azure)")
    parser.add_argument("--api-key", default=None,
                        help="API key (overrides XAI_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--processed-dir", default=None,
                        help="Override preprocessed output directory")
    args = parser.parse_args()

    cfg = load_config()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    md_path = Path(args.md) if args.md else output_path.with_suffix(".md")
    processed_dir = Path(args.processed_dir) if args.processed_dir else (
        Path(get("paths.processed_dir", "data/processed"))
    )

    if not input_dir.exists():
        print(f"[err] input directory not found: {input_dir}")
        return 1

    # Stage 0: preprocess
    if args.skip_preprocess:
        print(f"[preprocess] skipped, using {input_dir} as-is")
        processed_dir = input_dir
        image_files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    else:
        print(f"[preprocess] {input_dir} -> {processed_dir}")
        metas = process_directory(input_dir, processed_dir)
        image_files = [Path(m.output_path) for m in metas]

    if not image_files:
        print("[err] no images to process")
        return 1

    # Stage 1: recognizer
    print(f"[load] VietOCR (model_path={get('recognizer.model_path')})")
    recognizer = get_recognizer()

    # Stage 2: cloud fallback
    provider_name = args.provider if args.provider != "none" else ""
    use_fallback = (
        not args.no_fallback
        and provider_name != ""
    )
    fallback: object | None = None
    if use_fallback:
        fallback = _make_fallback_client(provider_name, args.api_key)
        if fallback is None:
            print(f"[load] {provider_name} client unavailable")
            use_fallback = False
        else:
            print(f"[load] {provider_name} fallback ready")
    else:
        print("[load] cloud fallback disabled")

    # Stage 3: per-image pipeline
    all_results: list[RecognizedRegion] = []
    table_subs_global: dict[int, list[RecognizedRegion]] = defaultdict(list)
    total_stats = {"vietocr_calls": 0, "fallback_calls": 0, "fallback_cost_usd": 0.0}

    for img_path in tqdm(image_files, desc="OCR"):
        if not img_path.exists():
            print(f"[warn] missing file: {img_path}")
            continue
        rec, table_subs, stats = _process_image(
            img_path, recognizer, fallback, provider_name, use_fallback
        )
        all_results.extend(rec)
        # Merge table subs (shift indices by current global count)
        offset = len(table_subs_global)
        for k, v in table_subs.items():
            table_subs_global[offset + k] = v
        for k in stats:
            total_stats[k] += stats[k]

    # Stage 4: reconstruct
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_docx(all_results, dict(table_subs_global), output_path)
    save_markdown(to_markdown(all_results, dict(table_subs_global)), md_path)

    # Final report
    ocr_calls = total_stats["vietocr_calls"]
    fb_calls = total_stats["fallback_calls"]
    total_images = len(image_files)
    print("\n========== REPORT ==========")
    print(f"Images processed: {total_images}")
    print(f"VietOCR calls:    {ocr_calls}")
    print(f"{provider_name or 'Fallback'} calls: {fb_calls}")
    print(f"Total cost:       ${total_stats['fallback_cost_usd']:.6f}")
    print(f"Output .docx:     {output_path}")
    print(f"Output .md:       {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
