#라이브러리 설치
!pip install -q \
transformers \
datasets \
sentencepiece \
timm \
bitsandbytes \
accelerate \
peft \
jiwer \
scikit-learn

#import
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import zipfile
import random
import shutil
import re

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel
)

from jiwer import cer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from tqdm import tqdm


#3
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

#4.
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(IMAGE_DIR)

print("압축 해제 완료!")

#5.
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

#6.
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

#7.
processor = AutoProcessor.from_pretrained(
    MERGED_PATH
)

model = VisionEncoderDecoderModel.from_pretrained(
    MERGED_PATH,
    device_map="auto"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()

print("Merged 모델 로드 완료!")

#8. test 데이터셋
test_dataset = SnackDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    processor
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

#9. 평가코드
#셸7-평가코드
ALLERGY_KEYWORDS = [
    "밀",
    "우유",
    "대두",
    "땅콩",
    "메밀",
    "고등어",
    "게",
    "새우",
    "돼지고기",
    "복숭아",
    "토마토",
    "호두",
    "알"
]

results = []

all_cer_scores = []

y_true = []
y_pred = []

print("평가 시작!")

with torch.no_grad():

    for batch_idx, batch in enumerate(
        tqdm(test_dataloader)
    ):

        image_path = test_dataset.image_files[
            batch_idx
        ]

        image_name = os.path.basename(
            image_path
        )

        pixel_values = batch[
            "pixel_values"
        ].to(DEVICE)

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        generated_ids = model.generate(

            pixel_values=pixel_values,

            max_length=256,

            decoder_start_token_id=
            processor.tokenizer.cls_token_id,

            pad_token_id=
            processor.tokenizer.pad_token_id,

            eos_token_id=
            processor.tokenizer.sep_token_id,

            no_repeat_ngram_size=3,

            repetition_penalty=1.2,

            num_beams=3 #3->1
        )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        json_match = re.search(
            r'\{.*\}',
            prediction
        )

        final_prediction = (
            json_match.group()
            if json_match
            else prediction
        )

        labels = batch["labels"]

        labels[
            labels == -100
        ] = processor.tokenizer.pad_token_id

        reference = processor.batch_decode(
            labels,
            skip_special_tokens=True
        )[0]

        # CER
        cer_score = cer(
            reference,
            final_prediction
        )

        all_cer_scores.append(cer_score)

        # Recall / F1
        gt_vector = []
        pred_vector = []

        for keyword in ALLERGY_KEYWORDS:

            gt_exist = int(
                keyword in reference
            )

            pred_exist = int(
                keyword in final_prediction
            )

            gt_vector.append(gt_exist)

            pred_vector.append(pred_exist)

        y_true.extend(gt_vector)

        y_pred.extend(pred_vector)

        results.append({

            "id": image_name,

            "prediction": final_prediction,

            "ground_truth": reference,

            "cer": cer_score
        })

avg_cer = sum(all_cer_scores) / len(
    all_cer_scores
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print("\n===== 최종 평가 결과 =====")

print(f"CER: {avg_cer:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1-score: {f1:.4f}")

final_output = {

    "metrics": {

        "CER": avg_cer,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1
    },

    "results": results
}

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_output,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\n평가 저장 완료!")
