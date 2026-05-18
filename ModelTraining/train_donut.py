# train_donut.py

import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel
)

# =========================
# 1. 경로 설정
# =========================

IMAGE_DIR = "./dataset/images"
LABEL_DIR = "./dataset/labels"

MODEL_NAME = "naver-clova-ix/donut-base"

BATCH_SIZE = 2
EPOCHS = 5
LR = 3e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Processor / Model
# =========================

processor = DonutProcessor.from_pretrained(MODEL_NAME)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# =========================
# 3. Dataset
# =========================

class SnackDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor

        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(self.image_dir, image_name)

        label_name = image_name.replace(".png", ".json")
        label_path = os.path.join(self.label_dir, label_name)

        # 이미지 열기
        image = Image.open(image_path).convert("RGB")

        # JSON 읽기
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # JSON -> 문자열
        target_sequence = json.dumps(
            label_data,
            ensure_ascii=False
        )

        # 이미지 처리
        pixel_values = processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze()

        # 텍스트 토큰화
        labels = processor.tokenizer(
            target_sequence,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # PAD 토큰 무시
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# =========================
# 4. DataLoader
# =========================

dataset = SnackDataset(
    IMAGE_DIR,
    LABEL_DIR,
    processor
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# 5. Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

# =========================
# 6. Train Loop
# =========================

model.train()

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1} =====")

    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"Batch {batch_idx+1} "
            f"| Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch Average Loss: {avg_loss:.4f}")

# =========================
# 7. 모델 저장
# =========================

SAVE_PATH = "./donut_snack_model"

model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("\n모델 저장 완료!")