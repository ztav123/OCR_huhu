"""Force OpenAI fallback on all regions to verify key + pipeline work."""
import os
import sys
sys.path.insert(0, r'C:\Users\lamth\OCR_hihi')
import cv2
from pathlib import Path

from src.layout import detect_layout
from src.recognizer import get_recognizer
from src.openai_client import OpenAIClient

api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print('ERR: OPENAI_API_KEY not set')
    sys.exit(1)
print(f'API key prefix: {api_key[:8]}...')

img_path = Path(r'C:\Users\lamth\OCR_hihi\data\raw_images\test01.jpg')
img = cv2.imread(str(img_path))
print(f'image shape: {img.shape}')

layout = detect_layout(img_path)
print(f'regions detected: {len(layout.regions)}')

recognizer = get_recognizer()
client = OpenAIClient(api_key=api_key)

print('\n=== FORCING OpenAI on all regions ===\n')
total_cost = 0.0
for i, region in enumerate(sorted(layout.regions, key=lambda r: (r.y1, r.x1))):
    crop = region.crop(img)
    if crop.size == 0:
        continue
    vietocr_text, vietocr_conf = recognizer.predict(crop)

    try:
        result = client.recognize(crop)
        total_cost += result.usage.estimated_cost_usd
        marker = '[CACHED]' if result.cached else '[LIVE]'
        print(f'#{i+1:2d} {marker} conf={result.confidence:.3f}  '
              f'p={result.usage.prompt_tokens} c={result.usage.completion_tokens}  '
              f'${result.usage.estimated_cost_usd:.6f}  '
              f'text={result.text[:80]!r}')
    except Exception as e:
        print(f'#{i+1:2d} [ERROR] {e}')

print(f'\nTotal cost: ${total_cost:.6f}')