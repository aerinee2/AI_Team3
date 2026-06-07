import os
import re
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from transformers import DonutProcessor, VisionEncoderDecoderModel
from peft import PeftModel

# =====================================================================
# [SafeEat Korea] 환경 설정 및 글로벌 모델 초기화
# =====================================================================
BASE_MODEL = "naver-clova-ix/donut-base"
ADAPTER_PATH = "/content/drive/MyDrive/SafeEat-Korea/donut-adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[SafeEat] ⚙️ 기반 VLM 엔진 초기화 중... (구동 디바이스: {DEVICE})")
processor = DonutProcessor.from_pretrained(BASE_MODEL)
base_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
base_model.decoder.resize_token_embeddings(len(processor.tokenizer))

# 파인튜닝된 Donut 어댑터 로드
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
model = model.to(DEVICE)
print("[SafeEat] 🎉 Donut VLM 단독 추론 엔진 세팅 완료.")


def predict(image_path: str) -> dict:
    """
    Donut VLM 모델 단독 구조를 활용하여 이미지 내 식품 성분표를 스캔하고,
    오타 정제 및 알레르기 유발 항원을 추론하여 최종 원재료명 리스트를 반환합니다.
    """
    # 1. 이미지 로드 및 비주얼 임계 최적화 (검정 박스 및 빛 반사 제어)
    img_raw = Image.open(image_path).convert("RGB")
    
    # 명도 및 대비 밸런싱을 통해 글자 외곽선 가시성 확보
    img_enhanced = ImageEnhance.Brightness(img_raw).enhance(0.85)
    img_enhanced = ImageEnhance.Contrast(img_enhanced).enhance(2.3)
    img_enhanced.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    
    # 2. Donut VLM 단독 문맥 추론 엔진 구동
    print("[SafeEat] 🧠 Donut VLM 단독 원문 추론을 시작합니다...")
    pixel_values = processor(img_enhanced, return_tensors="pt").pixel_values.to(DEVICE)
    decoder_start_id = processor.tokenizer.convert_tokens_to_ids("<s_food>")
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=150,
            num_beams=3,
            repetition_penalty=1.5,
            early_stopping=True,
            decoder_start_token_id=decoder_start_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    # 구조화 태그 제거 및 텍스트 정제
    pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    donut_raw_text = re.sub(r'<.*?>', '', pred).strip()
    print(f"[SafeEat] Donut 검증 원문 로그: {donut_raw_text}")

    # =====================================================================
    # 🎯 3. 원문 텍스트 슬라이싱 및 1차 정제 벨트
    # =====================================================================
    raw_splits = donut_raw_text.split(',')
    processed_tokens = []
    
    for token in raw_splits:
        token = token.strip()
        # 불필요한 특수문자 및 기호 가공 제거
        token = re.sub(r'[.\s\[\]()+\-:#]', '', token)
        
        # 유효한 문자 길이를 만족하고 연속 반복 노이즈(예: '전도도도도')가 아닌 경우만 수용
        if len(token) > 1 and not (len(token) > 4 and len(set(token)) / len(token) < 0.4):
            processed_tokens.append(token)

    # =====================================================================
    # 🤝 4. 식약처 지정 알레르기 유발 항원 보존 가드 및 오타 강력 정제 사전
    # =====================================================================
    # 육류 문맥 가드 자동 추론 작동
    if any(k in donut_raw_text for k in ['돼지', '전신', '돈']):
        processed_tokens.append("돼지고기")
    if any(k in donut_raw_text for k in ['소트닝', '우육', '소고기', '우지']):
        processed_tokens.append("쇠고기")

    CORRECTION_MAP = {
        '유란': '난류', '계란': '계란', '란백': '난백', '란황': '난황', '달갈': '달걀', '난유': '난류', '알류': '난류', '알유': '난류', '계런': '계란',
        '유유': '우유', '우우': '우유', '우휴': '우유', '유청': '유청', '탈지분우': '탈지분유', '전지분우': '전지분유', '우유함': '우유', '버러': '버터', '치저': '치즈', '가공치저': '치즈', '유청분': '유청', '요요': '우유',
        '매밀': '메밀', '메민': '메밀', '땅홍': '땅콩', '딴콩': '땅콩', '땅콜': '땅콩',
        '대도': '대두', '대두레시턴': '대두레시틴', '대두유': '대두유', '소이': '대두', '소이빈': '대두', '혼합대두': '대두', '대두함': '대두', '배두': '대두',
        '소맥분': '소맥분', '민가루': '밀가루', '소먁': '밀', '소먁분': '소맥분', '호밀': '호밀', '통밀': '밀',
        '장': '잣', '작': '잣', '호도': '호두', '오두': '호두', '개': '게', '꽃개': '꽃게',
        '새오': '새우', '새유': '새우', '생우': '새우', '새우분': '새우', '새요품미요': '새우풍미유',
        '외징어': '오징어', '오징어농축액': '오징어농축액', '오징어분말': '오징어', '고둥어': '고등어', '전뵥': '전복', '진복': '전복', '귤': '굴', '홍함': '홍합', '조개': '조개류', '바지락': '바지락',
        '돼지고기': '돼지고기', '쇠고기': '쇠고기', '돈지': '돼지고기', '돈피': '돼지고기', '돈육': '돼지고기', '돼지고가': '돼지고기',
        '우육': '쇠고기', '소고기': '쇠고기', '우지': '쇠고기', '비프': '쇠고기',
        '복숭화': '복숭아', '복숭': '복숭아', '보숭아': '복숭아', '도마토': '토마토', '토마토페이스트': '토마토페이스트', '아황산': '아황산나트륨', '아황산나트륨': '아황산나트륨', '산도조절제': '산도조절제',
        '아몬드': '아몬드', '아몬드분말': '아몬드분말', '외사비': '와사비', '외사비분말': '와사비분말', '재품명': '제품명', '팬유': '팜유', '혼함께제': '혼합제제'
    }

    # 2단계 스마트 오타 정제 매커니즘 (완전 매칭 및 서브 스트링 매칭)
    corrected_ingredients = []
    for item in processed_tokens:
        if item in CORRECTION_MAP:
            corrected_ingredients.append(CORRECTION_MAP[item])
            continue
            
        has_custom_fix = False
        for wrong_word, right_word in CORRECTION_MAP.items():
            if wrong_word in item:
                corrected_ingredients.append(right_word)
                has_custom_fix = True
                break
        
        if not has_custom_fix:
            corrected_ingredients.append(item)

    # =====================================================================
    # 🧼 5. 중복 제거 및 가비지 찌꺼기 최종 필터링
    # =====================================================================
    final_ingredients = list(dict.fromkeys([i for i in corrected_ingredients if len(i) > 1]))
    
    garbage_keywords = ['비기한', '후면', '표기일까지', '포장재질플리프로필렌', '목보고버히', '식품유형', '제품명', '과자유처리제품', '함유', '형기록', '원재료명']
    final_ingredients = [w for w in final_ingredients if w not in garbage_keywords and not re.match(r'^[a-zA-Z0-9]+$', w)]

    print(f"[SafeEat] 🚀 최종 정제 후 전송 리스트: {final_ingredients}")
    
    # 기존 main.py 및 템플릿 변수 호환성을 유지하기 위한 순정 인터페이스 적용
    return {"원재료명": final_ingredients}