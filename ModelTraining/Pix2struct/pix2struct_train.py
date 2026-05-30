"""
Pix2Struct Fine-tuning for Korean Food Allergen OCR
====================================================
Google Colab 최적화 버전

[코랩 첫 번째 셀에서 실행]
!pip install -q transformers sentencepiece
from google.colab import drive
drive.mount('/content/drive')

[폴더 구조]
/content/drive/MyDrive/snack_dataset/
├── images/   ← .png / .jpg
└── labels/   ← 같은 이름 .json

[JSON 라벨 예시]
{
  "allergens": ["우유", "밀", "대두", "난류"],
  "product": "홈런볼초코"
}
"""

import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# ══════════════════════════════════════════
# ★ 설정
# ══════════════════════════════════════════
IMAGE_DIR  = "/content/drive/MyDrive/snack_dataset/images"
LABEL_DIR  = "/content/drive/MyDrive/snack_dataset/labels"
SAVE_PATH  = "/content/drive/MyDrive/pix2struct_snack_model"  # 드라이브 저장 (세션 끊겨도 유지)

MODEL_NAME  = "google/pix2struct-base"
BATCH_SIZE  = 2       # T4 GPU 기준 2 (A100이면 4~8)
EPOCHS      = 5
LR          = 5e-5
MAX_PATCHES = 512     # 원본 1024 → 코랩 메모리 고려해 512로 축소 (정확도-속도 트레이드오프)
MAX_TOKENS  = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {DEVICE}")
if DEVICE == "cpu":
    print("  GPU 없음! 런타임 > 런타임 유형 변경 > T4 GPU 선택하세요.")


# ══════════════════════════════════════════
# 1. Processor / Model
# ══════════════════════════════════════════
print("[모델 로드 중...]")
processor = Pix2StructProcessor.from_pretrained(MODEL_NAME)
model     = Pix2StructForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)
print("[모델 로드 완료]")


# ══════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════
class SnackDataset(Dataset):
    """
    Donut과 다른 Pix2Struct의 핵심 차이:
    - pixel_values 대신 flattened_patches + attention_mask 사용
    - processor에 images와 text를 함께 넘겨야 함 (Donut은 따로)
    """
    def __init__(self, image_dir, label_dir, processor, max_patches, max_tokens):
        self.image_dir   = image_dir
        self.label_dir   = label_dir
        self.processor   = processor
        self.max_patches = max_patches
        self.max_tokens  = max_tokens

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"[Dataset] {len(self.image_files)}개 이미지 로드")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_name  = os.path.splitext(image_name)[0]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, base_name + ".json")

        # 이미지
        image = Image.open(image_path).convert("RGB")

        # JSON → 문자열
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        target_text = json.dumps(label_data, ensure_ascii=False)

        # ★ Pix2Struct 핵심: images + text 동시에 processor에 전달
        encoding = self.processor(
            images=image,
            text=target_text,
            return_tensors="pt",
            max_patches=self.max_patches,
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
        )

        # 배치 차원 제거
        flattened_patches = encoding["flattened_patches"].squeeze(0)
        attention_mask    = encoding["attention_mask"].squeeze(0)
        labels            = encoding["input_ids"].squeeze(0)

        # PAD 토큰 무시
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "flattened_patches": flattened_patches,
            "attention_mask":    attention_mask,
            "labels":            labels,
        }


# ══════════════════════════════════════════
# 3. Collate (배치 패딩 맞추기)
# ══════════════════════════════════════════
def collate_fn(batch):
    """
    Pix2Struct는 배치 내 패치 수가 다를 수 있어서
    Donut처럼 단순 stack이 안 되는 경우가 있음 → 직접 처리
    """
    flattened_patches = torch.stack([b["flattened_patches"] for b in batch])
    attention_mask    = torch.stack([b["attention_mask"]    for b in batch])

    max_len = max(b["labels"].shape[0] for b in batch)
    labels  = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["labels"]
        labels[i, :l.shape[0]] = l

    return {
        "flattened_patches": flattened_patches,
        "attention_mask":    attention_mask,
        "labels":            labels,
    }


# ══════════════════════════════════════════
# 4. DataLoader
# ══════════════════════════════════════════
dataset    = SnackDataset(IMAGE_DIR, LABEL_DIR, processor, MAX_PATCHES, MAX_TOKENS)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)


# ══════════════════════════════════════════
# 5. Optimizer + Scheduler 
# ══════════════════════════════════════════
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps  = len(dataloader) * EPOCHS
warmup_steps = total_steps // 10
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ══════════════════════════════════════════
# 6. Train Loop
# ══════════════════════════════════════════
os.makedirs(SAVE_PATH, exist_ok=True)
best_loss = float("inf")

model.train()
for epoch in range(EPOCHS):
    print(f"\n{'='*40}")
    print(f"Epoch {epoch+1} / {EPOCHS}")
    print(f"{'='*40}")

    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        flattened_patches = batch["flattened_patches"].to(DEVICE)
        attention_mask    = batch["attention_mask"].to(DEVICE)
        labels            = batch["labels"].to(DEVICE)

        # ★ Donut과 다른 점: pixel_values 대신 flattened_patches + attention_mask
        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1:>4} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"\n▶ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save_pretrained(SAVE_PATH)
        processor.save_pretrained(SAVE_PATH)
        print(f"  → Best 모델 저장 완료 (loss={best_loss:.4f})")

print("\n✅ 학습 완료!")
print(f"저장 위치: {SAVE_PATH}")


# ══════════════════════════════════════════
# 7. 추론 (학습 후 별도 셀에서 실행)
# ══════════════════════════════════════════
def predict(image_path: str) -> dict:
    """
    사용법:
        result = predict("/content/drive/MyDrive/snack_dataset/images/test.jpg")
        print(result)
        # → {"allergens": ["우유", "밀"], "product": "홈런볼초코"}
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
        max_patches=MAX_PATCHES,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
        )

    raw = processor.decode(output_ids[0], skip_special_tokens=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw}


# 테스트:
# result = predict("/content/drive/MyDrive/snack_dataset/images/test.jpg")
# print(result)
