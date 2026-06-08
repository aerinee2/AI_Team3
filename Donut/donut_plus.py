# -*- coding: utf-8 -*-
"""Donut 3차 연장학습 실험용 정리본.

실험 흐름은 유지하되, Colab 전용 셸 명령과 중복된 디버그 블록을 정리한 버전입니다.
"""

import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    from peft import PeftModel
except Exception as exc:
    raise ImportError("peft가 필요합니다. pip install peft 를 먼저 실행하세요.") from exc

# ------------------------------------------------------------
# 공통 경로
# ------------------------------------------------------------
BASE_DIR = Path("/content/drive/MyDrive/인지응 3팀 공유폴더")
DATA_DIR = Path("/content/dataset")
TAR_3000_PATH = BASE_DIR / "3000장" / "solo.tar"
ZIP_5000_PATH = BASE_DIR / "5000장" / "cleaned_images_jpg.zip"
LABEL_3000_PATH = BASE_DIR / "3000장" / "label_log.txt"
LABEL_5000_PATH = BASE_DIR / "5000장" / "label_log.txt"
TEMP_3000 = DATA_DIR / "temp_3000"
TEMP_5000 = DATA_DIR / "temp_5000"
TOTAL_IMG_DIR = DATA_DIR / "images_total_8000"
OLD_MODEL_PATH = BASE_DIR / "3000장" / "donut_snack_v3_final" / "final_v3_complete"
NEW_SAVE_PATH = BASE_DIR / "donut_snack_v4_ocr_free_complete"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------
def collect_images(base_dir: Path):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG")
    paths = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(valid_ext) and not filename.startswith("."):
                paths.append(Path(root) / filename)
    return sorted(paths)


def prepare_dataset():
    for folder in (TEMP_3000, TEMP_5000, TOTAL_IMG_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    with tarfile.open(TAR_3000_PATH, "r") as tar_ref:
        tar_ref.extractall(TEMP_3000)

    with zipfile.ZipFile(ZIP_5000_PATH, "r") as zip_ref:
        zip_ref.extractall(TEMP_5000)

    files_3000 = collect_images(TEMP_3000)
    files_5000 = collect_images(TEMP_5000)

    current_idx = 1
    for src in files_3000 + files_5000:
        shutil.copy(src, TOTAL_IMG_DIR / f"sequence.{current_idx}.jpg")
        current_idx += 1

    with open(LABEL_3000_PATH, "r", encoding="utf-8") as f:
        labels_3000 = [line.strip() for line in f if line.strip()]
    with open(LABEL_5000_PATH, "r", encoding="utf-8") as f:
        labels_5000 = [line.strip() for line in f if line.strip()]

    total_labels = labels_3000 + labels_5000
    assert len(total_labels) == len(list(TOTAL_IMG_DIR.iterdir())), "이미지 수와 라벨 수가 일치하지 않습니다."

    random.seed(42)
    all_indices = list(range(1, len(total_labels) + 1))
    random.shuffle(all_indices)

    split_point = int(len(all_indices) * 0.95)
    return total_labels, all_indices[:split_point], all_indices[split_point:]


# ------------------------------------------------------------
# 데이터셋 / 학습 함수
# ------------------------------------------------------------
class DonutOCRFreeDataset(Dataset):
    def __init__(self, img_dir, total_labels, indices, processor, train=True):
        self.img_dir = Path(img_dir)
        self.total_labels = total_labels
        self.indices = indices
        self.processor = processor
        self.train = train
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        ]) if train else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_num = self.indices[idx]
        image = Image.open(self.img_dir / f"sequence.{seq_num}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        raw_label = self.total_labels[seq_num - 1].replace("Ingredients:", "").strip()
        target = f"<s><s_food><s_원재료>{raw_label}</s_원재료></s_food></s>"

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target,
            add_special_tokens=False,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def train_extension(start_epoch=30, total_epochs=38, resume_path=OLD_MODEL_PATH, save_path=NEW_SAVE_PATH):
    total_labels, train_indices, val_indices = prepare_dataset()

    processor = DonutProcessor.from_pretrained(resume_path)
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", torch_dtype=torch.float16).to(DEVICE)
    base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True).to(DEVICE)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_food>")

    train_dataset = DonutOCRFreeDataset(TOTAL_IMG_DIR, total_labels, train_indices, processor, train=True)
    val_dataset = DonutOCRFreeDataset(TOTAL_IMG_DIR, total_labels, val_indices, processor, train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    print(f"학습 시작: {start_epoch + 1} ~ {total_epochs} 에폭")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            labels = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / 4

            scaler.scale(loss).backward()
            if (step + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * 4
            progress.set_postfix(loss=loss.item() * 4)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
                labels = batch["labels"].to(DEVICE)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        print(f"Epoch {epoch + 1}: train_loss={train_loss / len(train_loader):.4f}, val_loss={val_loss / len(val_loader):.4f}")

        epoch_save_path = save_path.parent / "donut_snack_v4_ocr_free" / f"epoch_{epoch + 1}"
        epoch_save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_save_path)
        processor.save_pretrained(epoch_save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"최종 저장 완료: {save_path}")

    return train_indices, val_indices, total_labels


# ------------------------------------------------------------
# 평가 함수
# ------------------------------------------------------------
def evaluate_model(model_path=NEW_SAVE_PATH, sample_limit=None):
    processor = DonutProcessor.from_pretrained(model_path)
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", torch_dtype=torch.float16).to(DEVICE)
    base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path).to(DEVICE).eval()

    total_labels, _, val_indices = prepare_dataset()
    dataset = DonutOCRFreeDataset(TOTAL_IMG_DIR, total_labels, val_indices, processor, train=False)
    if sample_limit is not None:
        dataset.indices = dataset.indices[:sample_limit]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            outputs = model.generate(
                pixel_values=pixel_values,
                max_length=80,
                num_beams=3,
                repetition_penalty=3.5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s_food>"),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            target = total_labels[dataset.indices[len(results)] - 1].replace("Ingredients:", "").strip()
            results.append({"file_name": f"sequence.{dataset.indices[len(results)]}.jpg", "target": target, "prediction": prediction})

    return results


# ------------------------------------------------------------
# 실행 예시
# ------------------------------------------------------------
if __name__ == "__main__":
    # 학습을 이어서 돌리고 싶다면 아래 주석을 풀어 사용하세요.
    # train_extension()

    # 빠른 평가만 확인하고 싶다면 아래를 실행하세요.
    # for row in evaluate_model(sample_limit=20):
    #     print(row)
    pass
