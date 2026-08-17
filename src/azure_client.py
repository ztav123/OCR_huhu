"""Azure AI Foundry client for fallback OCR.

API format (Azure AI Foundry):
    POST https://<resource>.services.ai.azure.com/api/projects/<project>/chat/completions
    Header: api-key: <your-key>

Pricing: Azure standard GPT-4o-mini pricing (~$0.15/M input, ~$0.60/M output as of 2026).
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
from src.config_loader import get

REQUEST_TIMEOUT = 60

SYSTEM_PROMPT = (
    "Bạn là trợ lý OCR tiếng Việt. Trích xuất CHÍNH XÁC văn bản trong ảnh, "
    "giữ nguyên dấu thanh, dấu mũ, nguyên âm đặc biệt (ă, â, ơ, ư, ô, ê, đ). "
    "Không thêm chú thích, không giải thích. Trả về JSON hợp lệ duy nhất."
)


@dataclass
class AzureUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class AzureResult:
    text: str
    confidence: float
    raw: str
    usage: AzureUsage
    cached: bool


class AzureClient:
    """Azure AI Foundry Chat Completions (OpenAI-compatible, Azure auth)."""

    # Azure AI Foundry GPT-4o-mini pricing (approximate)
    INPUT_PRICE_PER_M = 0.15
    OUTPUT_PRICE_PER_M = 0.60

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        project_path: str | None = None,
        api_version: str = "2024-06-01",
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        cfg_enabled = bool(get("azure.enabled", True))
        self._disabled = not cfg_enabled

        self.api_key = api_key or os.environ.get(
            str(get("azure.api_key_env", "AZURE_API_KEY"))
        )
        if not self._disabled and not self.api_key:
            raise RuntimeError(
                "AZURE_API_KEY not set. Export it or pass api_key=. "
                "Set azure.enabled=false in config to disable Azure."
            )

        self.base_url = (
            base_url
            or get("azure.base_url", "https://24522012-ai-agent-resource.services.ai.azure.com")
        ).rstrip("/")
        self.project_path = (
            project_path
            or get("azure.project_path", "api/projects/24522012-ai-agent-for-a-beginner")
        ).strip("/")
        self.model = model or get("azure.model", "gpt-4o-mini")
        self.api_version = api_version or get("azure.api_version", "2024-06-01")
        self.max_tokens = int(get("azure.max_tokens", 1024))
        self.temperature = float(get("azure.temperature", 0.1))
        self.use_cache = bool(use_cache if use_cache is not None
                              else get("azure.use_cache", True))

        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / "data" / "cache_azure"
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _chat_url(self) -> str:
        return (
            f"{self.base_url}/{self.project_path}/chat/completions"
            f"?api-version={self.api_version}"
        )

    @staticmethod
    def _encode_image(img: np.ndarray) -> tuple[str, str]:
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

    def recognize(self, image: np.ndarray | str | Path) -> AzureResult:
        """Send a single cropped image to Azure GPT-4o-mini and return parsed text."""
        if self._disabled:
            raise RuntimeError("Azure client disabled in config")

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
        else:
            img = image

        cache_key = self._cache_key_for_image(img)
        cached = self._read_cache(cache_key)
        if cached is not None:
            usage = AzureUsage(**cached["usage"])
            return AzureResult(
                text=cached["text"],
                confidence=float(cached["confidence"]),
                raw=cached.get("raw", ""),
                usage=usage,
                cached=True,
            )

        b64, mime = self._encode_image(img)
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
        }

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        resp = requests.post(
            self._chat_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )

        if resp.status_code == 401:
            raise RuntimeError(
                f"Azure API 401 Unauthorized. Check AZURE_API_KEY. "
                f"Response: {resp.text[:200]}"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Azure API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage_obj = data.get("usage", {})
        prompt_tokens = int(usage_obj.get("prompt_tokens", 0))
        completion_tokens = int(usage_obj.get("completion_tokens", 0))
        input_cost = prompt_tokens * self.INPUT_PRICE_PER_M / 1_000_000
        output_cost = completion_tokens * self.OUTPUT_PRICE_PER_M / 1_000_000
        used = AzureUsage(
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

        result = AzureResult(
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
    parser = argparse.ArgumentParser(description="Run Azure AI Foundry OCR on an image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    client = AzureClient(api_key=args.api_key)
    result = client.recognize(args.input)
    print(f"text: {result.text}")
    print(f"confidence: {result.confidence}")
    print(f"tokens: prompt={result.usage.prompt_tokens}, "
          f"completion={result.usage.completion_tokens}")
    print(f"cost: ${result.usage.estimated_cost_usd:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
