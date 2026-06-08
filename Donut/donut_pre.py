# -*- coding: utf-8 -*-
"""Donut_train_리팩터링_최종.ipynb"""

# ==========================================
# 1. 라이브러리 설치 및 구글 드라이브 마운트
# ==========================================
!pip install -q transformers datasets sentencepiece timm bitsandbytes accelerate peft evaluate

from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# 2. 전역 경로 설정 및 전처리 (통일 가이드)
# ==========================================
import os
import zipfile
import json
import gc
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import evaluate
from tqdm import tqdm

# 구글 드라이브 기본 경로
BASE_PATH = "/content/drive/MyDrive/인지응 3팀 공유폴더"
ZIP_PATH = f"{BASE_PATH}/GeneratedData.zip"
LABEL_TXT_PATH = f"{BASE_PATH}/labels.txt"

# 코랩 로컬 임시 작업 공간
IMAGE_DIR = "/content/dataset_images"
OUTPUT_LABEL_DIR = "/content/labels"
MODEL_SAVE_PATH = "/content/donut_model_temp"  # <--- 모든 가중치 경로를 이 주소 하나로 통일합니다!

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 압축 해제
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(IMAGE_DIR)
    print("압축 해제 완료!")

# 라벨 변환 (TXT -> JSON)
if os.path.exists(LABEL_TXT_PATH):
    with open(LABEL_TXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(".png")
        if len(parts) < 2: continue

        img_name = parts[0].strip() + ".png"
        content = parts[1].strip()
        label_dict = {"gt_parse": {"원재료명": content}}

        json_path = os.path.join(OUTPUT_LABEL_DIR, img_name.replace(".png", ".json"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(label_dict, f, ensure_ascii=False, indent=4)
    print("라벨 변환 완료!")

# ==========================================
# 3. 환경 최적화 및 모델/LoRA 로드
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

MODEL_NAME = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

# ==========================================
# 4. 방어적 데이터셋 클래스 정의
# ==========================================
class SnackDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        
        all_images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        self.image_files = []
        for f in all_images:
            label_name = f.replace(".png", ".json")
            if os.path.exists(os.path.join(label_dir, label_name)):
                self.image_files.append(f)

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")

        label_path = os.path.join(self.label_dir, image_name.replace(".png", ".json"))
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        target_sequence = json.dumps(label_data.get("gt_parse", label_data), ensure_ascii=False)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(target_sequence, add_special_tokens=True, max_length=512, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

# ==========================================
# 5. 모델 학습 (Training)
# ==========================================
dataset = SnackDataset(IMAGE_DIR, OUTPUT_LABEL_DIR, processor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

model.train()
print("학습 시작!")

for epoch in range(5):
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to("cuda")
        labels = batch["labels"].to("cuda")

        loss = model(pixel_values=pixel_values, labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# 학습 완료 후 지정된 단일 로컬 경로에 깔끔하게 저장
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"학습 완료 및 모델 저장 완료: {MODEL_SAVE_PATH}")

# 백업용 zip 생성 및 다운로드
!zip -r /content/donut_model_backup.zip {MODEL_SAVE_PATH}
print("백업용 zip 파일이 /content/donut_model_backup.zip 에 생성되었습니다.")

# ==========================================
# 6. 스모크 테스트 (반복 방지 파라미터 적용)
# ==========================================
model.eval()
test_dataset = SnackDataset(IMAGE_DIR, OUTPUT_LABEL_DIR, processor)
subset_indices = range(min(5, len(test_dataset)))

print("\n--- 모델 작동 여부 확인 (스모크 테스트) ---")
with torch.no_grad():
    for i in subset_indices:
        batch = test_dataset[i]
        pixel_values = batch["pixel_values"].unsqueeze(0).to("cuda")

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=512,
            decoder_start_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)

        print(f"\n[데이터 {i+1}]")
        print(f"정답: {reference_text}")
        print(f"예측: {generated_text}")

# ==========================================
# 7. 본 평가 및 전수 검증 (이어가기 모드 내장)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

RESULT_FILE = "/content/evaluation_results.json"
processed_files = set()

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
        processed_files = {item['id'] for item in results}
else:
    results = []

test_dataloader = DataLoader(test_dataset, batch_size=1)
cer_metric = evaluate.load("cer")

predictions = []
references = []

print("\n전수 성능 평가 시작 (이어가기 모드)...")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        image_name = test_dataset.image_files[batch_idx]
        if image_name in processed_files: continue

        pixel_values = batch["pixel_values"].to(device)
        if pixel_values.dim() == 3: pixel_values = pixel_values.unsqueeze(0)

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=512,
            decoder_start_token_id=processor.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            num_beams=3
        )

        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        json_match = re.search(r'\{.*\}', raw_text)
        final_result = json_match.group() if json_match else raw_text

        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)[0]

        results.append({
            "id": image_name,
            "prediction": final_result,
            "ground_truth": reference_text
        })

        # 실시간 저장 안정화
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

# 최종 CER 스코어 컴파일 출력
for item in results:
    predictions.append(item["prediction"])
    references.append(item["ground_truth"])

if predictions and references:
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    print(f"\n[평가 완료] 최종 글로벌 CER 점수: {cer_score:.4f}")
    print(f"상세 결과 장부 저장 완료: {RESULT_FILE}")
