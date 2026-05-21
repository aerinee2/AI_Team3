import torch
import json
import os
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. 기존 결과 파일이 있으면 로드 (이어가기 로직)
RESULT_FILE = "/content/evaluation_results.json"
results = []
processed_files = set()

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
        processed_files = {item['id'] for item in results}
    print(f"이미 {len(processed_files)}개의 평가 데이터가 존재합니다. 이어서 진행합니다.")

# 2. 평가 루프 (이어가기 기능 포함)
model.eval()
test_dataset = EvaluationDataset(IMAGE_DIR, OUTPUT_LABEL_DIR, processor)
test_dataloader = DataLoader(test_dataset, batch_size=1)

print("전체 성능 평가를 시작합니다...")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        # 데이터셋에서 현재 파일명 가져오기
        image_name = test_dataset.image_files[batch_idx]
        
        # 이미 평가한 파일이면 건너뜀
        if image_name in processed_files:
            continue
            
        pixel_values = batch["pixel_values"].to("cuda")
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        # 반복 방지 옵션이 적용된 추론
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=512,
            decoder_start_token_id=processor.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,        
            repetition_penalty=1.5,        
            num_beams=3                    
        )
        
        # 텍스트 디코딩 및 정제
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        json_match = re.search(r'\{.*\}', raw_text)
        final_result = json_match.group() if json_match else raw_text
        
        # 정답지 디코딩
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)[0]
        
        # 결과 저장
        results.append({
            "id": image_name,
            "prediction": final_result,
            "ground_truth": reference_text
        })
        
        # 매번 파일에 저장 (런타임 끊겨도 안전!)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n평가 완료! 결과가 {RESULT_FILE}에 저장되었습니다.")