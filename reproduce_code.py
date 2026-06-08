# -*- coding: utf-8 -*-
"""Donut 3차 연장학습 실험용 정리본 (v4 Master Model).

1. Backbone Model: Naver Clova IX - Donut (https://github.com/clovaai/donut)
2. Core Library: Hugging Face Transformers (https://github.com/huggingface/transformers)
3. Parameter-Efficient Fine-Tuning: Hugging Face PEFT (https://github.com/huggingface/peft)
4. Augmentation Strategy: PyTorch Torchvision (https://github.com/pytorch/vision)

이 스크립트는 가공식품 라벨 데이터셋 8,000장(1차 3K + 2차 5K) 기반의 
연장학습(Incremental Learning) 및 실물 이미지 추론 검증을 재현하기 위해 작성되었습니다.
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
    raise ImportError("peft 라이브러리가 필요합니다. 'pip install peft'를 실행하세요.") from exc

# ------------------------------------------------------------
# 공통 경로 및 하이퍼파라미터 설정
# ------------------------------------------------------------
DRIVE_BASE = "/content/drive/MyDrive/인지응 3팀 공유폴더"
DATA_DIR = Path("/content/dataset")
TAR_3000_PATH = Path(DRIVE_BASE) / "3000장" / "solo.tar"
ZIP_5000_PATH = Path(DRIVE_BASE) / "5000장" / "cleaned_images_jpg.zip"
LABEL_3000_PATH = Path(DRIVE_BASE) / "3000장" / "label_log.txt"
LABEL_5000_PATH = Path(DRIVE_BASE) / "5000장" / "label_log.txt"
TEMP_3000 = DATA_DIR / "temp_3000"
TEMP_5000 = DATA_DIR / "temp_5000"
TOTAL_IMG_DIR = DATA_DIR / "images_total_8000"
OLD_MODEL_PATH = Path(DRIVE_BASE) / "3000장" / "donut_snack_v3_final" / "final_v3_complete"
NEW_SAVE_PATH = Path(DRIVE_BASE) / "donut_snack_v4_ocr_free_complete"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 훈련 옵션 동기화
START_EPOCH = 32
TOTAL_EPOCHS = 38
ACCUMULATION_STEPS = 1  # 배치 8 기준 다이렉트 갱신 (필요 시 조절)


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
# 데이터셋 클래스 (안전한 인덱스 반환 추가)
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
        
        #  추론 매칭 버그 방지를 위해 seq_num을 함께 리턴
        return {"pixel_values": pixel_values, "labels": labels, "seq_num": seq_num}


# ------------------------------------------------------------
# 고도화 연장 학습 함수
# ------------------------------------------------------------
def train_extension(start_epoch=START_EPOCH, total_epochs=TOTAL_EPOCHS, resume_path=OLD_MODEL_PATH, save_path=NEW_SAVE_PATH):
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
    
    # 최신 권장 정석 문법 적용
    scaler = torch.amp.GradScaler('cuda')

    print(f"학습 시작: {start_epoch + 1} ~ {total_epochs} 에폭 스타트")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f" Real Train Epoch {epoch + 1}/{total_epochs}")
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUMULATION_STEPS
            progress.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
                labels = batch["labels"].to(DEVICE)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        print(f" 에포크 [{epoch + 1}] 완료 -> 평균 Train Loss: {train_loss / len(train_loader):.4f} | 평균 Val Loss: {val_loss / len(val_loader):.4f}")

        # 매 에폭 구글 드라이브 실시간 실물 백업
        epoch_save_path = save_path.parent / "donut_snack_v4_ocr_free" / f"epoch_{epoch + 1}"
        epoch_save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_save_path)
        processor.save_pretrained(epoch_save_path)
        print(f"[백업 완료] -> {epoch_save_path}")

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"최종 가중치 영구 보존 완료: {save_path}")

    return train_indices, val_indices, total_labels


# ------------------------------------------------------------
# 정밀 정량 평가 함수
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
            seq_nums = batch["seq_num"] # 배치에서 다이렉트로 원본 인덱스 추출 (안전)
            
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
            
            # 1:1 정합성이 완벽히 보장되는 target 추출
            target = total_labels[seq_nums[0].item() - 1].replace("Ingredients:", "").strip()
            results.append({
                "file_name": f"sequence.{seq_nums[0].item()}.jpg", 
                "target": target, 
                "prediction": prediction
            })

    return results


if __name__ == "__main__":
    print("[System] Donut v4 OCR-Free 파이프라인 재현 모듈 가동")
    # 예시: 연장 학습 및 평가 파이프라인 정석 구동
    # total_labels, train_idx, val_idx = prepare_dataset()
    # train_extension()
    pass