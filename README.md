# Vietnamese OCR Pipeline (MVP — Printed Text)

Pipeline xử lý ảnh chụp văn bản **tiếng Việt in** sạch → `.docx` (có table) + `.md`.

- **Recognizer local**: VietOCR (`vgg_transformer`) — fine-tune nhẹ trên VinText nếu cần.
- **Fallback thông minh**: Grok 4.5 API (xAI) cho vùng text có confidence thấp.
- **Tối ưu chi phí**: tiền xử lý ảnh local (resize, deskew) → crop vùng text → chỉ gửi crop nhỏ cho Grok → cache kết quả theo SHA256.

> Chữ viết tay sẽ được hỗ trợ ở Phase 5 (xem `Vietnamese_OCR_Plan.md`).

## Cấu trúc dự án

```
OCR_hihi/
├── data/
│   ├── raw_images/         # ảnh người dùng thả vào
│   ├── processed/          # ảnh sau tiền xử lý
│   ├── outputs/            # .docx + .md kết quả
│   └── cache_grok/         # cache response Grok (SHA256)
├── datasets/
│   ├── vintext/            # dataset VinText (tải tự động)
│   └── vintext_processed/  # processed cho VietOCR
├── models/
│   └── vietocr_finetuned/  # checkpoint fine-tune
├── src/
│   ├── config_loader.py
│   ├── download_vintext.py
│   ├── prepare_data.py     # convert VinText COCO -> VietOCR format
│   ├── train.py            # fine-tune VietOCR
│   ├── preprocess.py       # local image preprocessing
│   ├── layout.py           # heuristic text/table region detection
│   ├── recognizer.py       # VietOCR wrapper
│   ├── grok_client.py      # Grok 4.5 API client
│   └── reconstruct.py      # merge -> .docx + .md
├── config.yaml
├── requirements.txt
├── main.py
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10+.

```bash
# 1. Tạo venv (khuyến nghị)
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Cài dependencies
pip install -r requirements.txt
```

> `vietocr` được cài từ git vì PyPI version bị broken.

## Chuẩn bị dataset (chỉ làm 1 lần)

```bash
# 1. Tải VinText (COCO format, ~350 MB) từ VinAI Google Drive
python src/download_vintext.py

# 2. Convert sang format VietOCR
python src/prepare_data.py
```

## Fine-tune (tùy chọn)

Script sẽ **tự đánh giá base model trước** trên 200 ảnh val. Nếu CER < 8% thì sẽ **skip fine-tune** (tiết kiệm 4-6 giờ CPU).

```bash
# Đánh giá + fine-tune có điều kiện
python src/train.py

# Bắt buộc fine-tune (bỏ qua eval)
python src/train.py --skip-eval

# Chỉ đánh giá base model
python src/train.py --eval-only
```

Checkpoint lưu vào `models/vietocr_finetuned/vgg_transformer.pth`.

## Chạy OCR trên ảnh của bạn

```bash
# 1. Thả ảnh vào data/raw_images/

# 2. Set API key (nếu muốn dùng Grok fallback)
$env:XAI_API_KEY = "xai-..."        # PowerShell
# export XAI_API_KEY="xai-..."      # bash

# 3. Chạy pipeline
python main.py --input data/raw_images --output data/outputs/result.docx
```

File `.md` đi kèm sẽ được lưu cùng tên (`result.md`).

### Tùy chọn CLI

| Flag | Mô tả |
|---|---|
| `--input DIR` | Thư mục ảnh đầu vào (bắt buộc) |
| `--output FILE` | Đường dẫn file `.docx` đầu ra (bắt buộc) |
| `--md FILE` | Đường dẫn file `.md` (mặc định: cùng tên output) |
| `--skip-preprocess` | Bỏ qua tiền xử lý (input đã sẵn sàng) |
| `--no-grok` | Tắt Grok fallback, dùng VietOCR thuần |
| `--api-key KEY` | Ghi đè `XAI_API_KEY` env var |
| `--processed-dir DIR` | Override thư mục preprocessed |

## Cấu hình (`config.yaml`)

Chỉnh các tham số chính:

```yaml
preprocess:
  max_edge_px: 1568       # sweet spot cho Grok 448 tiles
  jpeg_quality: 85

recognizer:
  confidence_threshold: 0.7   # < threshold -> Grok fallback

grok:
  enabled: true
  model: grok-4.5
  max_tokens: 1024
  use_cache: true              # cache by SHA256 của crop
```

## Ước tính chi phí Grok

- 100 ảnh đơn giản → ~0 API call (VietOCR xử lý hết).
- 100 ảnh khó → ~50 vùng cần Grok, mỗi vùng ~500 token input + 200 token output ≈ $0.002/vùng → **~$0.1 tổng** (so với ~$2 nếu gửi full ảnh).

## Pipeline hoạt động thế nào

```
Input folder
   │
   ▼
[preprocess]  Resize (max edge 1568px), deskew, denoise, JPEG q=85
   │
   ▼
[layout]      Heuristic OpenCV tách vùng text / table
   │
   ▼
[recognizer]  VietOCR (fine-tuned) → (text, confidence)
   │           confidence < 0.7 ?
   ├─ no  ───► dùng text VietOCR
   └─ yes ───► [grok]  gửi crop đến Grok 4.5, parse JSON
   │
   ▼
[reconstruct] Gộp theo thứ tự đọc → .md + .docx (giữ table)
```

## Lấy `XAI_API_KEY`

1. Vào [console.x.ai](https://console.x.ai) → đăng ký / đăng nhập.
2. Tạo API key mới.
3. Set env var `XAI_API_KEY` hoặc truyền `--api-key`.

## Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| CPU chậm khi fine-tune | Train qua đêm, dataset nhỏ (56K instance) |
| Grok đọc sai dấu | Prompt yêu cầu giữ nguyên dấu; validate với VietOCR |
| Table bị xáo trộn cell | Reconstruct grid dựa trên bbox alignment |
| API key lộ | Đọc từ env var, không commit vào repo |

## License

Dự án cá nhân. Dataset VinText theo license VinAI Research (nghiên cứu / giáo dục).
