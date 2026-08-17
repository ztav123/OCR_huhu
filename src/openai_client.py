"""OpenAI GPT-4o client for fallback OCR.

Same interface as GrokClient so they can be swapped via config/provider flag.

Pricing (OpenAI 2026):
    - gpt-4o input:  $2.50 / 1M tokens
    - gpt-4o output: $10.00 / 1M tokens
    - cached input: $1.25 / 1M tokens (50% off)
    - 1 image ≈ 765 tokens (low detail) or 1024+ tiles (high detail)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config_loader import get  # noqa: E402

REQUEST_TIMEOUT = 60

SYSTEM_PROMPT = (
    "Bạn là trợ lý OCR tiếng Việt. Trích xuất CHÍNH XÁC văn bản trong ảnh, "
    "giữ nguyên dấu thanh, dấu mũ, nguyên âm đặc biệt (ă, â, ơ, ư, ô, ê, đ). "
    "Không thêm chú thích, không giải thích. Trả về JSON hợp lệ duy nhất."
)


@dataclass
class OpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class OpenAIResult:
    text: str
    confidence: float
    raw: str
    usage: OpenAIUsage
    cached: bool


class OpenAIClient:
    """Thin wrapper for OpenAI Chat Completions API (vision)."""

    INPUT_PRICE_PER_M = 2.50
    CACHED_INPUT_PRICE_PER_M = 1.25
    OUTPUT_PRICE_PER_M = 10.00

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        cfg_enabled = bool(get("openai.enabled", True))
        self._disabled = not cfg_enabled

        self.api_key = api_key or os.environ.get(
            str(get("openai.api_key_env", "OPENAI_API_KEY"))
        )
        if not self._disabled and not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or pass api_key=. "
                "Set openai.enabled=false in config to disable OpenAI."
            )

        self.base_url = (
            base_url or get("openai.base_url", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or get("openai.model", "gpt-4o")
        self.max_tokens = int(get("openai.max_tokens", 1024))
        self.temperature = float(get("openai.temperature", 0.1))
        self.use_cache = bool(use_cache if use_cache is not None
                              else get("openai.use_cache", True))
        self.cache_key = str(get("openai.prompt_cache_key", "ocr_vn_printed_v1"))

        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / "data" / "cache_openai"
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _encode_image(img: np.ndarray) -> tuple[str, str]:
        """Encode BGR ndarray as base64 data URL."""
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Failed to encode image to JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return b64, "image/jpeg"

    def _cache_key_for_image(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return hashlib.sha256(buf.tobytes()).hexdigest()

    def _read_cache(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, payload: dict) -> None:
        if not self.use_cache:
            return
        p = self.cache_dir / f"{key}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def recognize(self, image: np.ndarray | str | Path) -> OpenAIResult:
        """Send a single cropped image to GPT-4o and return parsed text."""
        if self._disabled:
            raise RuntimeError("OpenAI client disabled in config")

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
        else:
            img = image

        cache_key = self._cache_key_for_image(img)
        cached = self._read_cache(cache_key)
        if cached is not None:
            usage = OpenAIUsage(**cached["usage"])
            return OpenAIResult(
                text=cached["text"],
                confidence=float(cached["confidence"]),
                raw=cached.get("raw", ""),
                usage=usage,
                cached=True,
            )

        b64, mime = self._encode_image(img)
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": '{"text": "...", "confidence": 0.0}'},
                    ],
                },
            ],
            # OpenAI prompt cache key (works for GPT-4o + newer models)
            "prompt_cache_key": self.cache_key,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage_obj = data.get("usage", {})
        prompt_tokens = int(usage_obj.get("prompt_tokens", 0))
        completion_tokens = int(usage_obj.get("completion_tokens", 0))
        cached_tokens = int(usage_obj.get("cached_tokens", 0) or 0)
        if cached_tokens > 0:
            input_cost = (
                (prompt_tokens - cached_tokens) * self.INPUT_PRICE_PER_M
                + cached_tokens * self.CACHED_INPUT_PRICE_PER_M
            ) / 1_000_000
        else:
            input_cost = prompt_tokens * self.INPUT_PRICE_PER_M / 1_000_000
        output_cost = completion_tokens * self.OUTPUT_PRICE_PER_M / 1_000_000
        used = OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(input_cost + output_cost, 6),
        )

        parsed = self._extract_json(content) or {}
        text = str(parsed.get("text", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        result = OpenAIResult(
            text=text, confidence=confidence, raw=content,
            usage=used, cached=False,
        )
        self._write_cache(cache_key, {
            "text": result.text,
            "confidence": result.confidence,
            "raw": result.raw,
            "usage": asdict(result.usage),
        })
        return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenAI GPT-4o OCR on an image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    client = OpenAIClient(api_key=args.api_key)
    result = client.recognize(args.input)
    print(f"text: {result.text}")
    print(f"confidence: {result.confidence}")
    print(f"tokens: prompt={result.usage.prompt_tokens}, "
          f"completion={result.usage.completion_tokens}")
    print(f"cost: ${result.usage.estimated_cost_usd:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
