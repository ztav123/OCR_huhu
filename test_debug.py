"""Debug VietOCR confidence + is_low_confidence decision on real crops."""
import sys
sys.path.insert(0, r'C:\Users\lamth\OCR_hihi')
import cv2
import os

from src.layout import detect_layout
from src.recognizer import get_recognizer, is_low_confidence

img_path = r'C:\Users\lamth\OCR_hihi\data\raw_images\test01.jpg'
img = cv2.imread(img_path)
print(f'image shape: {img.shape}')

layout = detect_layout(Path := __import__('pathlib').Path(img_path))
print(f'regions detected: {len(layout.regions)}')

recognizer = get_recognizer()
print('\n=== Per-region OCR + low-confidence decision ===\n')

for i, region in enumerate(sorted(layout.regions, key=lambda r: (r.y1, r.x1))):
    crop = region.crop(img)
    if crop.size == 0:
        continue
    text, conf = recognizer.predict(crop)
    low = is_low_confidence(text, conf, threshold=0.85)
    flag = ' -> FALLBACK' if low else ' -> keep'
    print(f'#{i+1:2d} conf={conf:.4f}  text={text[:80]!r}  {flag}')