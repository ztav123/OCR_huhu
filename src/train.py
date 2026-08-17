"""Fine-tune VietOCR on VinText.

Workflow:
1. First, evaluate the pretrained base model on the VinText val set.
   If CER < 0.08, skip fine-tune and use the base model directly.
2. Otherwise, fine-tune vgg_transformer on train + val (15-20 epochs).
3. Save best checkpoint to models/vietocr_finetuned/vgg_transformer.pth.

Usage:
    python src/train.py                 # full pipeline (eval + possible train)
    python src/train.py --skip-eval    # force fine-tune
    python src/train.py --eval-only    # only evaluate base model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "vintext_processed"
MODEL_DIR = PROJECT_ROOT / "models" / "vietocr_finetuned"
BASE_CER_THRESHOLD = 0.08  # skip fine-tune if base model achieves CER below this


def _check_dependencies() -> None:
    try:
        import vietocr  # noqa: F401
    except ImportError as e:
        print("[err] vietocr not installed. Run:")
        print("       pip install vietocr @ git+https://github.com/pbcquoc/vietocr.git")
        raise SystemExit(1) from e


def _ensure_data_ready() -> None:
    if not (PROCESSED_DIR / "train.txt").exists():
        print(f"[err] {PROCESSED_DIR}/train.txt not found.")
        print("       Run: python src/prepare_data.py first.")
        raise SystemExit(1)


def evaluate_base(limit: int | None = None) -> float:
    """Run pretrained VietOCR on the val set and return CER."""
    from PIL import Image
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cpu"
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    val_lines = (PROCESSED_DIR / "val.txt").read_text(encoding="utf-8").splitlines()
    if limit:
        val_lines = val_lines[:limit]

    # CER uses vi_VN tokenizer (character-level)
    total_chars = 0
    total_errors = 0
    for line in val_lines:
        rel_path, label = line.split("\t", 1)
        img_path = PROCESSED_DIR / rel_path
        if not img_path.exists():
            continue
        # VietOCR's Predictor expects a PIL.Image.Image (it calls .convert("RGB"))
        pil_img = Image.open(img_path).convert("RGB")
        result = detector.predict(pil_img, return_prob=True)
        # If return_prob=True returns (text, prob), unpack; else unpack single value
        if isinstance(result, tuple):
            pred, _prob = result
        else:
            pred = result
        total_chars += max(len(label), len(pred))
        total_errors += _edit_distance_chars(label, pred)

    cer = total_errors / max(total_chars, 1)
    print(f"[base eval] samples={len(val_lines)}, CER={cer:.4f}")
    return cer


def _edit_distance_chars(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def fine_tune(epochs: int = 15, batch_size: int = 8, lr: float = 1e-4) -> Path:
    """Fine-tune VietOCR vgg_transformer on VinText processed data."""
    from vietocr.tool.config import Cfg
    from vietocr.trainer import Trainer

    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cpu"
    config["trainer"]["train_manifest"] = str(PROCESSED_DIR / "train.txt")
    config["trainer"]["valid_manifest"] = str(PROCESSED_DIR / "val.txt")
    config["trainer"]["batch_size"] = batch_size
    config["trainer"]["optimizer"]["lr"] = lr
    config["trainer"]["epochs"] = epochs
    config["trainer"]["print_every"] = 200
    config["trainer"]["export_weights"] = str(MODEL_DIR / "vgg_transformer.pth")
    config["trainer"]["check_cer"] = True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config, pretrained=True)
    print(f"[train] starting fine-tune: epochs={epochs}, batch={batch_size}, lr={lr}")
    trainer.train()
    print(f"[train] saved checkpoint to {MODEL_DIR}")
    return MODEL_DIR / "vgg_transformer.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune VietOCR on VinText")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip base-model evaluation (force fine-tune)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate base model, do not fine-tune")
    parser.add_argument("--eval-limit", type=int, default=200,
                        help="Number of val samples for base eval (default 200)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    _check_dependencies()
    _ensure_data_ready()

    base_cer: float | None = None
    if not args.skip_eval:
        print("[1/2] Evaluating base VietOCR model on VinText val subset...")
        base_cer = evaluate_base(limit=args.eval_limit)

    if args.eval_only:
        print(f"[done] base CER = {base_cer:.4f}")
        return 0

    if base_cer is not None and base_cer < BASE_CER_THRESHOLD:
        print(f"[skip-train] base CER {base_cer:.4f} < threshold {BASE_CER_THRESHOLD}")
        print("            Using base model directly. To force fine-tune, use --skip-eval.")
        return 0

    print("[2/2] Fine-tuning...")
    fine_tune(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
