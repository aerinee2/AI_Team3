from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from deep_translator import GoogleTranslator  # pip install googletrans==4.0.0-rc1
import os
import shutil

# from model import predict  # ✅ 팀원 모델 연동 시 주석 해제

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
}


def translate_ingredient(korean_text: str) -> str:
    try:
        return GoogleTranslator(source='ko', target='en').translate(korean_text)
    except Exception:
        return korean_text


def check_allergies(ingredients: list) -> dict:
    found_allergies = []
    translated = []

    for item in ingredients:
        if item in ALLERGY_DB:
            eng = ALLERGY_DB[item]
            found_allergies.append(eng)
        else:
            eng = translate_ingredient(item)
        translated.append(eng)

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
    # result = predict(file_path)
    result = {
        "원재료명": ["밀가루", "땅콩", "우유", "소금", "설탕"]
    }

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)