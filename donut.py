# -*- coding: utf-8 -*-
"""Donut 실험용 정리본.

Colab에서 실행한 실험 코드를 깔끔한 단일 Python 파일 형태로 정리한 버전입니다.
실험 중 확인한 학습/평가/추론 흐름은 유지하고, 중복된 디버그 블록과 셀 전용 구문은 제거했습니다.
"""

import os
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

try:
    from peft import PeftModel
except Exception as exc:
    raise ImportError("peft가 필요합니다. pip install peft 를 먼저 실행하세요.") from exc

# ------------------------------------------------------------
# 공통 경로
# ------------------------------------------------------------
BASE_DIR = Path("/content/drive/MyDrive/인지응 3팀 공유폴더/3000장")
LABEL_FILE = BASE_DIR / "label_log.txt"
TAR_PATH = BASE_DIR / "solo.tar"
SAVE_PATH = BASE_DIR / "donut_snack_v3_final"
IMAGE_FINAL_DIR = Path("/content/dataset/images")
FINAL_MODEL_PATH = SAVE_PATH / "final_v3_complete"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 데이터셋
# ------------------------------------------------------------
class SnackDataset(Dataset):
    def __init__(self, img_dir, label_file, processor, train=True):
        self.processor = processor
        self.train = train
        self.img_dir = Path(img_dir)

        with open(label_file, "r", encoding="utf-8") as f:
            self.lines = [line.strip().replace("Ingredients:", "").strip() for line in f if line.strip()]

        existing_imgs = set(self.img_dir.iterdir())
        self.img_files = [f"sequence.{i + 1}.png" for i in range(len(self.lines)) if (self.img_dir / f"sequence.{i + 1}.png") in existing_imgs]

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        ]) if train else None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        image = Image.open(self.img_dir / img_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        line_idx = int(re.findall(r"\d+", img_name)[0]) - 1
        target = f"<s><s_food><s_원재료>{self.lines[line_idx]}</s_원재료></s_food></s>"

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


# ------------------------------------------------------------
# 실험용 유틸
# ------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("s_food>", "").replace("s_원재료>", "")
    text = text.strip()
    return re.sub(r"[。、\.]", ",", text)


def calculate_cer(pred, target):
    if not target:
        return 1.0
    import Levenshtein

    return Levenshtein.distance(pred, target) / len(target)


def build_master_dict(label_file=LABEL_FILE):
    with open(label_file, "r", encoding="utf-8") as f:
        lines = [line.strip().replace("Ingredients:", "").strip() for line in f if line.strip()]

    master_dict = set()
    for line in lines:
        for word in line.split(","):
            w = word.strip()
            if w and len(w) > 1:
                master_dict.add(w)
    return master_dict


def advanced_clean_and_correct(text, master_dict):
    text = clean_text(text)
    raw_words = [w.strip() for w in text.split(",") if w.strip()]
    corrected = []

    for word in raw_words:
        if word in master_dict:
            corrected.append(word)
            continue

        prefix_matches = [v for v in master_dict if v.startswith(word) and len(word) >= 2]
        if prefix_matches:
            corrected.append(min(prefix_matches, key=len))
            continue

        close_matches = __import__("difflib").get_close_matches(word, master_dict, n=1, cutoff=0.5)
        corrected.append(close_matches[0] if close_matches else word)

    unique_words = []
    for item in corrected:
        if item not in unique_words:
            unique_words.append(item)
    return ", ".join(unique_words)


# ------------------------------------------------------------
# 1) 학습 (실험 기록 유지)
# ------------------------------------------------------------
def train_model():
    IMAGE_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    START_EPOCH = 20
    TOTAL_EPOCHS = 30
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 3e-5

    processor = DonutProcessor.from_pretrained(FINAL_MODEL_PATH)
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", device_map="auto", torch_dtype=torch.float16)
    base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base_model, FINAL_MODEL_PATH, is_trainable=True)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_food>")

    dataset = SnackDataset(IMAGE_FINAL_DIR, LABEL_FILE, processor, train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    model.train()
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        if (epoch + 1) % 5 == 0:
            save_path = SAVE_PATH / f"epoch_{epoch + 1}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"중간 저장: {save_path}")

    final_path = SAVE_PATH / "final_v3_complete"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"최종 저장: {final_path}")


# ------------------------------------------------------------
# 2) 평가 (실험 결과 확인용)
# ------------------------------------------------------------
def evaluate_model(model_path=FINAL_MODEL_PATH, sample_limit=None):
    processor = DonutProcessor.from_pretrained(model_path)
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path).to(DEVICE)
    model.eval()

    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip().replace("Ingredients:", "").strip() for line in f if line.strip()]

    dataset = SnackDataset(IMAGE_FINAL_DIR, LABEL_FILE, processor, train=False)
    if sample_limit is not None:
        dataset.img_files = dataset.img_files[:sample_limit]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    total_cer = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            labels = batch["labels"]
            _ = labels

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

            pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            pred_clean = clean_text(pred)
            target_clean = lines[int(re.findall(r"\d+", dataset.img_files[len(results)])[0]) - 1]
            cer = calculate_cer(pred_clean, target_clean)
            total_cer += cer

            results.append({"file_name": dataset.img_files[len(results)], "target": target_clean, "prediction": pred_clean, "cer": cer})

    avg_cer = total_cer / max(len(results), 1)
    accuracy = (1 - avg_cer) * 100 if avg_cer <= 1 else 0

    print(f"평균 CER: {avg_cer:.4f}")
    print(f"정확도: {accuracy:.2f}%")

    return pd.DataFrame(results), avg_cer


# ------------------------------------------------------------
# 3) 단일 이미지 추론
# ------------------------------------------------------------
def predict_ingredients(image_path, model_path=FINAL_MODEL_PATH, master_dict=None):
    if master_dict is None:
        master_dict = build_master_dict()

    processor = DonutProcessor.from_pretrained(model_path)
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", torch_dtype=torch.float16)
    base_model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path).to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.6)
    image = image.resize((800, 800), Image.LANCZOS)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE, dtype=torch.float16)
    decoder_start_id = processor.tokenizer.convert_tokens_to_ids("<s_food>")

    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=180,
            num_beams=3,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=decoder_start_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    raw_pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    corrected = advanced_clean_and_correct(raw_pred, master_dict)
    return corrected


# ------------------------------------------------------------
# 실험 실행 예시
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. 학습을 이어서 돌리고 싶다면 주석을 풀어서 사용하세요.
    # train_model()

    # 2. 평가만 빠르게 확인하고 싶다면 아래를 실행하세요.
    # df, avg_cer = evaluate_model(sample_limit=50)
    # print(df.head())

    # 3. 단일 이미지 추론 예시
    # example_path = "/content/test_inputs/1003_망고향분말_2.jpg"
    # print(predict_ingredients(example_path))
    pass
