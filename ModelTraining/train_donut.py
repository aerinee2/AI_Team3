# 1. 라이브러리 설치
!pip install -q transformers datasets sentencepiece timm
!pip install -q bitsandbytes accelerate
!pip install -q peft

# 2. 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3. 경로 설정
import os
import zipfile
import json

BASE_PATH = "/content/drive/MyDrive/인지응 3팀 공유폴더"
ZIP_PATH = f"{BASE_PATH}/GeneratedData.zip"
LABEL_TXT_PATH = f"{BASE_PATH}/labels.txt"

# 작업용 임시 폴더 경로 (코랩 내부 공간)
IMAGE_DIR = "/content/dataset_images"
OUTPUT_LABEL_DIR = "/content/labels"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 4. 압축 해제 (공유 폴더 -> 코랩 임시 폴더)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(IMAGE_DIR)

# 5. 라벨(txt) -> JSON 변환 (공유 폴더 -> 코랩 임시 폴더)
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

print("1단계 완료: 압축 해제와 JSON 변환이 코랩 내부 공간에 성공적으로 끝났습니다!")



import torch
import gc
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# 1. 환경 설정 및 메모리 최적화
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# 2. 모델 로드 (8-bit 양자화)
MODEL_NAME = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
)

# 토큰 정보 강제 지정
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id

# LoRA 적용 (학습 효율성 극대화)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# 3. 데이터셋 클래스 (방어적 코드 적용)
class SnackDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        # 라벨이 존재하는 파일만 리스트에 추가
        all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.image_files = []
        for f in all_files:
            if os.path.exists(os.path.join(label_dir, f.replace(".png", ".json"))):
                self.image_files.append(f)
            else:
                print(f"Warning: 라벨 파일 누락되어 건너뜁니다: {f}")

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        
        label_path = os.path.join(self.label_dir, image_name.replace(".png", ".json"))
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        
        target_sequence = json.dumps(label_data.get("gt_parse", label_data), ensure_ascii=False)
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = processor.tokenizer(target_sequence, add_special_tokens=True, max_length=512, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

# 4. 학습 실행
IMAGE_DIR = "/content/dataset_images"
OUTPUT_LABEL_DIR = "/content/labels"
SAVE_PATH = "/content/drive/MyDrive/인지응 3팀 공유폴더/donut_snack_model"

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

# 5. 모델 저장
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print("학습 완료!")


# 1. 드라이브가 아닌 코랩의 로컬 임시 폴더에 먼저 저장 (이건 무조건 성공합니다)
TEMP_SAVE_PATH = "/content/donut_model_temp"
model.save_pretrained(TEMP_SAVE_PATH)
processor.save_pretrained(TEMP_SAVE_PATH)

print("모델이 로컬 임시 폴더에 저장되었습니다!")

# 2. 그런 다음, 이 폴더를 수동으로 드라이브로 옮기거나 압축해서 다운로드합니다.
!zip -r /content/donut_model.zip /content/donut_model_temp
from google.colab import files
files.download("/content/donut_model.zip")