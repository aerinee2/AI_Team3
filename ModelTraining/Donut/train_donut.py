# 1. 라이브러리 설치
!pip install -q \
transformers \
datasets \
sentencepiece \
timm \
bitsandbytes \
accelerate \
peft

# 2. 구글 드라이브 마운트 + import
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import zipfile
import random
import shutil

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# 3. 경로 설정
BASE_PATH = "/content/drive/MyDrive/인지응 3팀 공유폴더/1000장"

ZIP_PATH = f"{BASE_PATH}/solo.zip"

LABEL_TXT_PATH = f"{BASE_PATH}/label_log.txt"

IMAGE_DIR = "/content/dataset_images"

LABEL_DIR = "/content/labels"

TRAIN_IMAGE_DIR = "/content/train_images"
TRAIN_LABEL_DIR = "/content/train_labels"

TEST_IMAGE_DIR = "/content/test_images"
TEST_LABEL_DIR = "/content/test_labels"

SAVE_PATH = f"{BASE_PATH}/donut_snack_model"

MERGED_PATH = f"{BASE_PATH}/merged_donut_model"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_LABEL_DIR, exist_ok=True)

# 4. 압축 해제 (공유 폴더 -> 코랩 임시 폴더)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(IMAGE_DIR)

print("압축 해제 완료!")

# 5. 라벨(txt) -> JSON 변환 ()

all_images = []

for root, dirs, files in os.walk(IMAGE_DIR):

    for file in files:

        if file.endswith(".png"):

            full_path = os.path.join(
                root,
                file
            )

            all_images.append(full_path)

all_images = sorted(all_images)

with open(
    LABEL_TXT_PATH,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()

for image_path, line in zip(all_images, lines):

    folder_name = os.path.basename(
        os.path.dirname(image_path)
    )

    ingredient = line.replace(
        "Ingredients:",
        ""
    ).strip()

    label_dict = {
        "ingredient": ingredient
    }

    json_path = os.path.join(
        LABEL_DIR,
        folder_name + ".json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as jf:

        json.dump(
            label_dict,
            jf,
            ensure_ascii=False,
            indent=4
        )

print("JSON 생성 완료!")

#6. Train/Test 분리
random.seed(42)

random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)

train_images = all_images[:split_idx]
test_images = all_images[split_idx:]

print("Train:", len(train_images))
print("Test:", len(test_images))

# =========================
# Train 복사
# =========================

for img_path in train_images:

    folder_name = os.path.basename(
        os.path.dirname(img_path)
    )

    new_image_name = f"{folder_name}.png"

    shutil.copy(
        img_path,
        os.path.join(
            TRAIN_IMAGE_DIR,
            new_image_name
        )
    )

    shutil.copy(
        os.path.join(
            LABEL_DIR,
            folder_name + ".json"
        ),
        os.path.join(
            TRAIN_LABEL_DIR,
            folder_name + ".json"
        )
    )

# =========================
# Test 복사
# =========================

for img_path in test_images:

    folder_name = os.path.basename(
        os.path.dirname(img_path)
    )

    new_image_name = f"{folder_name}.png"

    shutil.copy(
        img_path,
        os.path.join(
            TEST_IMAGE_DIR,
            new_image_name
        )
    )

    shutil.copy(
        os.path.join(
            LABEL_DIR,
            folder_name + ".json"
        ),
        os.path.join(
            TEST_LABEL_DIR,
            folder_name + ".json"
        )
    )

print("Train/Test 분리 완료!")

#7. 도넛 베이스 모델 로드
MODEL_NAME = "naver-clova-ix/donut-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.pad_token_id = (
    processor.tokenizer.pad_token_id
)

model.config.decoder_start_token_id = (
    processor.tokenizer.cls_token_id
)

#model = prepare_model_for_kbit_training(
    #model
#)

print("모델 로드 완료!")

#8. LoRA 적용
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

#9. Dataset 클래스
class SnackDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir,
        processor
    ):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor

        self.image_files = sorted(
            os.listdir(image_dir)
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        label_path = os.path.join(
            self.label_dir,
            image_name.replace(".png", ".json")
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        ingredient = label_data["ingredient"]

        target_sequence = (
            "<s_food>" +
            json.dumps(
                {"ingredient": ingredient},
                ensure_ascii=False
            )
            +"</s_food>"
        )

        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values[0]

        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    
#10. DataLoader
dataset = SnackDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    processor
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

#11. Optimizer
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=3e-5
)

#12. 학습
EPOCHS = 5

ACCUMULATION_STEPS = 4

model.train()

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1} =====")

    total_loss = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):

        pixel_values = batch["pixel_values"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        loss = loss / ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:

            optimizer.step()

            optimizer.zero_grad()

        total_loss += loss.item()

        print(
            f"Step {step+1}"
            f" | Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(dataloader)

    print(
        f"Epoch Average Loss: "
        f"{avg_loss:.4f}"
    )

#13. LoRA 어댑터 저장
model.save_pretrained(SAVE_PATH)

processor.save_pretrained(SAVE_PATH)

print("LoRA adapter 저장 완료!")

#14. merge
merged_model = model.merge_and_unload()

#15. merged model 저장
merged_model.save_pretrained(
    MERGED_PATH
)

processor.save_pretrained(
    MERGED_PATH
)

print("Merged 모델 저장 완료!")
