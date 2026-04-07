# 🪪 ID-Card-Information-Extractor

> **신분증 이미지 내 Key-Value 자동 추출 및 구조화 모델**
> OCR과 문서 이해(Document AI) 기술을 활용하여 다양한 신분증(주민등록증, 운전면허증 등) 사진에서 이름, 번호 등 주요 정보를 자동 분류하고 추출하는 프로젝트입니다.

---

## 🚀 프로젝트 소개 (Overview)

이 프로젝트는 신분증 이미지(비정형 데이터)를 입력받아, 정해진 카테고리(Name, Birthday, ID Number 등)에 맞는 텍스트 데이터를 자동으로 분류하여 구조화(Structure)된 데이터로 변환하는 것을 목표로 합니다.

**핵심 기능:**
* **자동 문서 분류:** 입력된 이미지가 주민등록증인지, 운전면허증인지 자동으로 판단합니다.
* **지능형 정보 추출 (Key-Value Extraction):** 단순히 글자를 읽는 것을 넘어, 위치와 문맥을 파악해 '이름' 필드에 해당하는 값, '번호' 필드에 해당하는 값을 구별하여 추출합니다.
* **데이터 구조화:** 추출된 정보를 JSON 형식 등으로 내보내어 DB화하기 용이하게 만듭니다.

---

## 🛠 기술 스택 (Tech Stack) (사용예정)

### Core Technologies
* **Image Processing:** OpenCV, Pillow
* **OCR Engine:**
  * 예) Naver CLOVA OCR, Tesseract, Google Vision API
* **Document AI Model:** 
  * 예) LayoutLM, Donut, 또는 자체 개발 분류 모델
* **Deep Learning Framework:** PyTorch, TensorFlow
* **Language:** Python 3.9+

---

## 📊 결과 예시 (Result Examples)

사용자가 일일이 필드를 지정하지 않아도, 모델이 자동으로 아래와 같은 형식으로 정보를 분류하여 출력합니다.

| 입력 이미지 | 추출 결과 (Key) | 추출값 (Value) | 신뢰도 (Confidence) |
| :---: | :--- | :--- | :---: |
| *(신분증 사진)* | **Name (성명)** | 홍길동 | 99.8% |
| *(신분증 사진)* | **ID Number (주민등록번호)** | 900101-1****** | 98.5% |
| *(신분증 사진)* | **Address (주소)** | 서울특별시 중구... | 95.2% |
| *(신분증 사진)* | **Type (종류)** | 주민등록증 | 99.9% |

---

## 📋 개발 파이프라인 (Data Pipeline)

1. **Pre-processing:** 이미지 노이즈 제거, 정규화, 기울기 보정 (Orientation)
2. **Text Detection:** 이미지 내에서 글자 영역 탐지 (Bbox 생성)
3. **Text Recognition (OCR):** 탐지된 영역의 글자를 텍스트로 변환
4. **Information Extraction:** 텍스트와 위치 정보를 종합하여 Key-Value Pair로 분류 (LayoutLM 등 활용)
5. **Post-processing:** 마스킹 및 정규표현식을 이용한 데이터 정제

---
