from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. 메인 화면 띄우기
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# 2. 카메라로 이미지 받아서 결과 보여주기
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    
    # 1. 이미지를 저장할 폴더 생성 (없으면 자동 생성)
    UPLOAD_DIR = "uploaded_images"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 2. 저장할 파일의 전체 경로 만들기 (예: uploaded_images/my_pic.jpg)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 3. 사용자가 업로드한 파일 내용을 진짜로 서버에 저장하기
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"업로드된 사진이 저장된 경로: {file_path}")
    
    # [학습한 AI 모델 코드가 들어감]
    #아직 AI 모델 코드 없어서 가짜 결과 출력..
    dummy_korean = "원재료명: 밀가루, 땅콩 함유"
    dummy_english = "Ingredients: Wheat, Contains Peanut. \n⚠️ WARNING: Peanut Allergy Risk!"
    
    return templates.TemplateResponse(
        request=request, 
        name="result.html", 
        context={
            "detected_text": dummy_korean,
            "analysis": dummy_english
        }
    )