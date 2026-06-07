from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from deep_translator import GoogleTranslator  # pip install googletrans==4.0.0-rc1
import os
import shutil

from model import predict  # ✅ 팀원 모델 연동 시 주석 해제

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# ============================
# 알레르기 유발 물질 딕셔너리
# ============================
ALLERGY_DB = {
    "땅콩": "Peanut",
    "우유": "Milk",
    "밀": "Wheat",
    "밀가루": "Wheat",
    "계란": "Egg",
    "달걀": "Egg",
    "새우": "Shrimp",
    "게": "Crab",
    "복숭아": "Peach",
    "메밀": "Buckwheat",
    "대두": "Soybean",
    "콩": "Soybean",
    "호두": "Walnut",
    "잣": "Pine Nut",
    "아황산류": "Sulfites",
    "오징어": "Squid",
    "조개류": "Shellfish",
    "고등어": "Mackerel",

    # ✅ 식약처 추가 항목
    "쇠고기": "Beef",
    "소고기": "Beef",
    "돼지고기": "Pork",
    "닭고기": "Chicken",
    "닭": "Chicken",
    "전복": "Abalone",
    "홍합": "Mussel",
    "굴": "Oyster",
    "전복": "Abalone",
    "잉어": "Carp",
    "연어": "Salmon",
    "참치": "Tuna",
    "토마토": "Tomato",
    "아황산": "Sulfites",

    # ✅ 견과류 확장
    "아몬드": "Almond",
    "캐슈넛": "Cashew",
    "피스타치오": "Pistachio",
    "마카다미아": "Macadamia",
    "헤이즐넛": "Hazelnut",
    "피칸": "Pecan",
    "브라질너트": "Brazil Nut",

    # ✅ 표기 변형 대응 (OCR 오인식 방지)
    "난류": "Egg",          # 계란의 공식 표기
    "난": "Egg",
    "유제품": "Dairy",
    "우유성분": "Milk",
    "소맥": "Wheat",        # 밀의 한자어
    "소맥분": "Wheat",
    "글루텐": "Gluten",
    "락토": "Lactose",
    "카제인": "Casein",     # 우유 단백질
    "유청": "Whey",         # 우유 단백질
    "새우젓": "Shrimp",
    "굴소스": "Oyster",
    "게맛살": "Crab",
}


def translate_ingredient(korean_text: str) -> str:
    try:
        result = GoogleTranslator(source='ko', target='en').translate(korean_text)
        return result if result else korean_text  # ✅ None이면 원문 반환
    except Exception:
        return korean_text  # ✅ 에러나도 원문 반환

def check_allergies(ingredients: list) -> dict:
    found_allergies = []
    translated = []

    full_text = " ".join(ingredients)  # 한국어 원문으로 합치기

    # ✅ 한국어 원문에서 먼저 알레르기 체크
    for korean, english in ALLERGY_DB.items():
        if korean in full_text:
            if english not in found_allergies:  # 중복 방지
                found_allergies.append(english)

    # 번역은 표시용으로만
    for item in ingredients:
        if item in ALLERGY_DB:
            translated.append(ALLERGY_DB[item])
        else:
            translated.append(translate_ingredient(item))

    return {
        "found": found_allergies,
        "translated_ingredients": translated
    }


def build_warning_message(allergy_result: dict) -> str:
    found = allergy_result["found"]
    if found:
        allergens_str = ", ".join(found)
        return (
            f"⚠️ ALLERGY WARNING\n"
            f"This product contains: {allergens_str}\n\n"
            f"Please consult your doctor if you have related allergies."
        )
    return "No major allergens detected in this product."


# ============================
# 라우터
# ============================

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):

    UPLOAD_DIR = "uploaded_images"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[SafeEat] 업로드 경로: {file_path}")

    # ✅ 팀원 모델 연동 후 아래 주석 해제, dummy 주석 처리
    result = predict(file_path)
    # result = {
    #     "원재료명": ["밀가루", "땅콩", "우유", "소금", "설탕"]
    # }

    ingredients = result.get("원재료명", [])
    allergy_result = check_allergies(ingredients)
    warning_message = build_warning_message(allergy_result)
    ingredients_english = ", ".join(allergy_result["translated_ingredients"])

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "detected_text": ingredients_english,
            "detected_text_korean": ", ".join(ingredients),
            "analysis": warning_message
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)