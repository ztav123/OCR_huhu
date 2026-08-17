"""VietOCR recognizer wrapper with confidence scoring.

Loads fine-tuned model if available, otherwise falls back to pretrained base.
Returns (text, confidence) per crop. Confidence is the softmax probability
of the predicted token averaged across the sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config_loader import get  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "vietocr_finetuned"
MODEL_NAME = "vgg_transformer"


def _ensure_vietocr() -> None:
    try:
        import vietocr  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "vietocr not installed. Run:\n"
            "  pip install vietocr @ git+https://github.com/pbcquoc/vietocr.git"
        ) from e


class VietOCRRecognizer:
    """Lazy-loaded VietOCR predictor with confidence estimate."""

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        _ensure_vietocr()
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config = Cfg.load_config_from_name(MODEL_NAME)
        config["device"] = device
        config["predictor"]["beamsearch"] = False  # greedy for speed

        # If fine-tuned weights exist, swap them in
        ft_path = Path(model_path) if model_path else MODEL_DIR / f"{MODEL_NAME}.pth"
        if ft_path.exists():
            # VietOCR's Predictor expects weights_path
            config["weights"] = str(ft_path)
            print(f"[recognizer] using fine-tuned: {ft_path}")
        else:
            print(f"[recognizer] no fine-tuned weights at {ft_path}, using base model")

        self.predictor = Predictor(config)

    @staticmethod
    def _avg_prob(prob):
        """Average sequence probability. VietOCR returns a 1D tensor."""
        try:
            import torch
            if isinstance(prob, torch.Tensor):
                return float(prob.mean().item())
        except Exception:
            pass
        if isinstance(prob, (list, np.ndarray)) and len(prob) > 0:
            return float(np.mean(prob))
        return 0.0

    def predict(self, image: np.ndarray | str | Path) -> tuple[str, float]:
        """Run OCR on a single crop (BGR ndarray, image path, or PIL Image). Returns (text, confidence)."""
        from PIL import Image

        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        else:
            # numpy BGR ndarray from cv2 -> PIL RGB
            img = image
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img).convert("RGB")
            else:
                pil_img = Image.fromarray(img[..., ::-1]).convert("RGB")

        text, prob = self.predictor.predict(pil_img, return_prob=True)
        confidence = self._avg_prob(prob)
        return text, confidence

    def predict_batch(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(img) for img in images]


def is_low_confidence(text: str, confidence: float, threshold: float | None = None) -> bool:
    """Decide whether a prediction should fallback to cloud based on confidence + sanity."""
    if threshold is None:
        threshold = float(get("recognizer.confidence_threshold", 0.7))
    if not text or not text.strip():
        return True
    if confidence < threshold:
        return True

    # Hallucination heuristics — VietOCR base tends to confidently emit
    # gibberish (repeated tokens, very short text on long lines).
    clean = text.strip()
    tokens = clean.split()
    if tokens:
        # 1) Single-word very short output on presumably long line => uncertain
        if len(tokens) == 1 and len(clean) <= 2:
            return True
        # 2) Any token is repeated >=3 times consecutively => hallucination
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                if tokens[i] == tokens[i + 1] == tokens[i + 2]:
                    return True
        # 3) Same character repeated >=6 times in a row => hallucination
        for ch in set(clean):
            if ch * 6 in clean:
                return True
    return False


# Singleton accessor to avoid reloading model on every call
_recognizer: VietOCRRecognizer | None = None


def get_recognizer() -> VietOCRRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = VietOCRRecognizer()
    return _recognizer


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run VietOCR on a list of text-line crops")
    parser.add_argument("--input", required=True, nargs="+", help="Image path(s)")
    parser.add_argument("--model", default=None, help="Path to .pth weights (optional)")
    args = parser.parse_args()

    rec = VietOCRRecognizer(model_path=args.model)
    for p in args.input:
        text, conf = rec.predict(p)
        print(f"{Path(p).name}: conf={conf:.3f}  text={text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
