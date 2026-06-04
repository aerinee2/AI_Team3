using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Perception.Randomization.Randomizers;
using TMPro;
using System.IO;

[Serializable]
public class FoodTextRandomizer : Randomizer
{
    public TextAsset ingredientFile;

    [Header("폰트 에셋 설정 (Elements 목록)")]
    public TMP_FontAsset[] fontAssets;

    [Header("데이터 리스트")]
    public List<string> productNames = new List<string> { "매콤 떡볶이", "진한 치즈케이크", "바삭 어니언링", "스파이시 라면", "고소한 우유" };
    public List<string> manufacturers = new List<string> { "(주)푸드테크", "우리식품", "데일리푸드", "팔도강산", "나눔푸드" };
    public List<string> allergies = new List<string> { "대두, 밀 함유", "우유, 토마토 함유", "난류, 메밀 함유", "쇠고기 함유", "새우, 게 함유" };
    public List<string> storages = new List<string> { "실온 보관", "냉동 보관", "서늘한 곳 보관" };
    public List<string> cautions = new List<string> { "신고는 1399", "개봉 후 즉시 섭취", "전자레인지 조리 불가" };

    [Header("시각적 노이즈 문구 (화면 출력O / 라벨 저장X)")]
    public List<string> factoryCautions = new List<string>
    {
        "이 제품은 알레르기 유발 물질인 알을 사용한 제품과 같은 제조시설에서 제조하고 있습니다.",
        "본 제품은 메밀, 고등어, 게, 새우, 돼지고기, 복숭아, 토마토, 호두를 사용한 제품과 같은 제조시설에서 제조하고 있습니다.",
        "이 제품은 땅콩 및 아황산류를 사용한 제품과 같은 제조시설에서 제조하고 있습니다."
    };

    [Header("라벨 디자인 랜덤 범위 설정")]
    public float fontSize = 35f;
    public float minWidth = 350f;
    public float maxWidth = 800f;

    //  실전 데이터 증강용 원산지 목록
    private string[] randomCountries = { "국산", "외국산", "미국산", "네덜란드산", "말레이시아산", "중국산", "호주산", "독일산" };

    protected override void OnIterationStart()
    {
        if (ingredientFile == null)
        {
            Debug.LogError("[에러] Ingredient File이 인스펙터에 연결되지 않았습니다!");
            return;
        }

        string[] allWords = ingredientFile.text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        var tags = tagManager.Query<TextRandomizerTag>().Where(t => t.name == "FoodTable").ToList();

        foreach (var tag in tags)
        {
            var tableImage = tag.GetComponent<Image>();
            var tableRect = tag.GetComponent<RectTransform>();

            Transform mainTrans = tag.transform.Find("LabelText");
            Transform barTrans = tag.transform.Find("AllergyBar");

            if (mainTrans == null || barTrans == null) continue;

            var mainTmp = mainTrans.GetComponent<TextMeshProUGUI>();
            var allergyBarRect = barTrans.GetComponent<RectTransform>();
            var allergyTmp = barTrans.GetComponentInChildren<TextMeshProUGUI>();

            if (mainTmp == null || allergyTmp == null) continue;

            // --- 데이터 추출 ---
            string chosenProductName = productNames[UnityEngine.Random.Range(0, productNames.Count)];
            string chosenManufacturer = manufacturers[UnityEngine.Random.Range(0, manufacturers.Count)];
            string chosenAllergy = allergies[UnityEngine.Random.Range(0, allergies.Count)];
            string chosenStorage = storages[UnityEngine.Random.Range(0, storages.Count)];
            string chosenCaution = cautions[UnityEngine.Random.Range(0, cautions.Count)];
            string chosenFactoryCaution = factoryCautions[UnityEngine.Random.Range(0, factoryCautions.Count)];

            // 랜덤으로 20~45개 단어 선택
            var selectedWords = allWords.OrderBy(x => UnityEngine.Random.value).Take(UnityEngine.Random.Range(20, 46)).ToList();

            // --------------------------------------------------------
            // 화면 출력용(노이즈 가득) vs 답지 저장용(순정) 이중 분할 처리
            // --------------------------------------------------------
            List<string> uiIngredientsList = new List<string>();
            List<string> logIngredientsList = new List<string>();

            foreach (string word in selectedWords)
            {
                string cleanWord = word.Trim();
                if (string.IsNullOrEmpty(cleanWord)) continue;

                // 1) 답지용 리스트에는 괄호 없이 무조건 '순정 알맹이 단어'만 저장
                logIngredientsList.Add(cleanWord);

                // 2) 화면 출력용 리스트에는 주사위를 굴려 실전형 괄호/원산지 노이즈 삽입
                float noiseDice = UnityEngine.Random.value;

                if (noiseDice < 0.35f) // 35% 확률로 (원산지) 노이즈 주입 -> 예: 대두(미국산)
                {
                    string country = randomCountries[UnityEngine.Random.Range(0, randomCountries.Length)];
                    uiIngredientsList.Add($"{cleanWord}({country})");
                }
                else if (noiseDice < 0.50f) // 15% 확률로 (특정성분:원산지) 복합 노이즈 주입 -> 예: 조개류엑기스(굴:국산)
                {
                    string country = randomCountries[UnityEngine.Random.Range(0, randomCountries.Length)];
                    uiIngredientsList.Add($"{cleanWord}(포도당:{country})");
                }
                else if (noiseDice < 0.60f) // 10% 확률로 (함유) 노이즈 주입 -> 예: 밀(밀함유)
                {
                    uiIngredientsList.Add($"{cleanWord}({cleanWord}함유)");
                }
                else // 나머지 40% 확률은 괄호 없이 깔끔하게 출력
                {
                    uiIngredientsList.Add(cleanWord);
                }
            }

            // UI 텍스트에 들어갈 최종 원재료명 문자열 생성
            string uiIngredientsText = string.Join(", ", uiIngredientsList);
            // 정답 로그 파일에 들어갈 순정 문자열 생성
            string logIngredientsText = string.Join(", ", logIngredientsList);


            // --------------------------------------------------------
            // [1, 2] 배경색 완전 랜덤화 및 밝기 맞춤형 글씨 색상 제어
            // --------------------------------------------------------
            Color randomBgColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);
            if (tableImage != null) tableImage.color = randomBgColor;

            float luminance = 0.2126f * randomBgColor.r + 0.7152f * randomBgColor.g + 0.0722f * randomBgColor.b;

            Color mainTextColor;
            Color allergyBarColor;
            Color allergyTextColor;

            if (luminance > 0.5f)
            {
                mainTextColor = new Color(UnityEngine.Random.Range(0.0f, 0.2f), UnityEngine.Random.Range(0.0f, 0.2f), UnityEngine.Random.Range(0.0f, 0.2f), 1f);
                allergyBarColor = new Color(0.15f, 0.15f, 0.15f, 1f);
                allergyTextColor = Color.white;
            }
            else
            {
                mainTextColor = new Color(UnityEngine.Random.Range(0.8f, 1.0f), UnityEngine.Random.Range(0.8f, 1.0f), UnityEngine.Random.Range(0.8f, 1.0f), 1f);
                allergyBarColor = new Color(0.9f, 0.9f, 0.9f, 1f);
                allergyTextColor = Color.black;
            }

            // --------------------------------------------------------
            // [3] 폰트 랜덤화 및 자간/행간 랜덤 노이즈
            // --------------------------------------------------------
            if (fontAssets != null && fontAssets.Length > 0)
            {
                TMP_FontAsset targetFont = fontAssets[UnityEngine.Random.Range(0, fontAssets.Length)];
                mainTmp.font = targetFont;
                allergyTmp.font = targetFont;
            }

            mainTmp.characterSpacing = UnityEngine.Random.Range(-5f, 3f);
            mainTmp.lineSpacing = UnityEngine.Random.Range(-9f, 12f);

            // --- UI 조립 (여기에 노이즈가 가득한 uiIngredientsText가 매핑됩니다) ---
            List<string> UI_pieces = new List<string>
            {
                $"<b>제품명:</b> {chosenProductName}",
                $"<b>제조원:</b> {chosenManufacturer}",
                $"<b>보관방법:</b> {chosenStorage}",
                $"<b>원재료명:</b> {uiIngredientsText}",
                $"<b>주의사항:</b> {chosenCaution} | {chosenFactoryCaution}"
            };

            mainTmp.text = string.Join("  |  ", UI_pieces);
            mainTmp.fontSize = fontSize;
            mainTmp.color = mainTextColor;
            mainTmp.alignment = TextAlignmentOptions.Left;

            allergyTmp.text = $"[알레르기 정보] {chosenAllergy}";
            allergyTmp.fontSize = fontSize;
            allergyTmp.color = allergyTextColor;
            allergyTmp.alignment = TextAlignmentOptions.Center;

            var allergyBarImage = barTrans.GetComponent<Image>();
            if (allergyBarImage != null) allergyBarImage.color = allergyBarColor;

            // --------------------------------------------------------
            // [4] 네모 상자 레이아웃 다양화 및 "실물 패키지 재질/주름" 랜덤화
            // --------------------------------------------------------
            GameObject cylinder = GameObject.Find("Cylinder");
            if (cylinder != null)
            {
                var renderer = cylinder.GetComponent<Renderer>();

                if (renderer != null)
                {
                    float smoothness = 0.5f;
                    float metallic = 0.0f;
                    float bumpScale = 0.0f; 

                    float materialDice = UnityEngine.Random.value;

                    if (materialDice < 0.4f)
                    {
                        // 캔 스타일 (반무광 알루미늄 메탈 + 잔주름)
                        metallic = UnityEngine.Random.Range(0.7f, 0.95f);
                        smoothness = UnityEngine.Random.Range(0.2f, 0.4f);
                        bumpScale = UnityEngine.Random.Range(0.2f, 0.8f); // 메탈은 주름이 잔잔하게
                    }
                    else if (materialDice < 0.7f)
                    {
                        // 과자 봉지 스타일 (쨍한 유광 비닐 + 구김)
                        metallic = UnityEngine.Random.Range(0.0f, 0.1f);
                        smoothness = UnityEngine.Random.Range(0.75f, 0.95f);

                        // 60% 확률로 비닐을 사정없이 쭈글쭈글하게 구겨버림
                        bumpScale = (UnityEngine.Random.value > 0.4f) ? UnityEngine.Random.Range(0.2f, 1.0f) : 0f;
                    }
                    else
                    {
                        // 우유팩/상자 스타일 (매트한 무광 종이 + 주름 없음)
                        metallic = 0.0f;
                        smoothness = UnityEngine.Random.Range(0.05f, 0.15f);
                        bumpScale = 0.0f; // 종이 상자는 주름이 없습니다.
                    }

                    // 유니티 표준 셰이더(Standard Shader) 속성에 최종 값 주입
                    renderer.material.SetFloat("_Glossiness", smoothness);
                    renderer.material.SetFloat("_Smoothness", smoothness);
                    renderer.material.SetFloat("_Metallic", metallic);

                    if (renderer.material.HasProperty("_BumpScale"))
                    {
                        renderer.material.SetFloat("_BumpScale", bumpScale);
                    }
                }

                // --- 여기서부터 레이아웃 크기 연산 (기존 형님 코드 스코프 안전지대) ---
                float chosenTableWidth;

                // 50% 확률로 가로가 길쭉한 상자 또는 세로가 홀쭉한 상자 타깃팅
                if (UnityEngine.Random.value > 0.5f)
                {
                    // 가로 넙데데 스타일 (뚱뚱한 패키지)
                    chosenTableWidth = UnityEngine.Random.Range(600f, maxWidth);
                }
                else
                {
                    // 세로 슬림 스타일 (몬스터 캔, 길쭉한 음료팩)
                    chosenTableWidth = UnityEngine.Random.Range(minWidth, 500f);
                }

                float textWidth = chosenTableWidth - 40f;

                // 세로 높이 유동적 래핑 연산
                mainTmp.rectTransform.sizeDelta = new Vector2(textWidth, 5000f);
                mainTmp.ForceMeshUpdate();

                float mainH = mainTmp.preferredHeight;
                mainTmp.rectTransform.sizeDelta = new Vector2(textWidth, mainH);

                Canvas.ForceUpdateCanvases();
                float barW = Mathf.Min(allergyBarRect.rect.width, textWidth);
                allergyBarRect.sizeDelta = new Vector2(barW, allergyBarRect.rect.height);

                float barH = allergyBarRect.rect.height;
                float totalHeight = mainH + barH + 80f;

                // 최종 결정된 네모(FoodTable) 크기 할당
                tableRect.sizeDelta = new Vector2(chosenTableWidth, totalHeight);

                // 실린더 랜덤 회전
                cylinder.transform.rotation = Quaternion.Euler(0, UnityEngine.Random.Range(0f, 360f), 0);
            }

            // --------------------------------------------------------
            // 테두리 선(Outline) 유무 및 두께 확률 랜덤 노이즈
            // --------------------------------------------------------
            var tableOutline = tag.GetComponent<Outline>();

            if (tableOutline != null)
            {
                float borderDice = UnityEngine.Random.value;

                if (borderDice < 0.35f)
                {
                    // 1) 35% 확률로 테두리 아예 끄기 (선 없는 비닐 성분표 대비)
                    tableOutline.enabled = false;
                }
                else if (borderDice < 0.70f)
                {
                    // 2) 35% 확률로 일반적인 얇은 테두리 선 유지
                    tableOutline.enabled = true;
                    tableOutline.effectDistance = new Vector2(1f, -1f);

                    // 글자색 반전 로직에서 계산된 luminance 활용하여 센스있게 선 색상 매칭
                    tableOutline.effectColor = (luminance > 0.5f) ?
                        new Color(0f, 0f, 0f, 0.4f) : new Color(1f, 1f, 1f, 0.4f);
                }
                else
                {
                    // 3) 30% 확률로 완전 굵고 진한 테두리 선 강제 부여 (눈뽕/백화 현상 방패용)
                    tableOutline.enabled = true;

                    // 대각선으로 두껍게 빡! (인스펙터의 수치를 스크립트로 조절)
                    float thick = UnityEngine.Random.Range(4f, 7f);
                    tableOutline.effectDistance = new Vector2(thick, -thick);

                    // 확실하게 컷팅되도록 투명도 없는 100% 불투명 검은색 또는 흰색 부여
                    tableOutline.effectColor = (luminance > 0.5f) ? Color.black : Color.white;
                }
            }



            // --- [정답 로그 기록 - 순정 단어만 깔끔하게 저장] ---
            string logPath = Path.Combine(Application.persistentDataPath, "label_log.txt");
            string logEntry = $"Ingredients: {logIngredientsText}";

            try
            {
                File.AppendAllText(logPath, logEntry + Environment.NewLine);
            }
            catch (Exception e)
            {
                Debug.LogWarning("로그 기록 실패: " + e.Message);
            }
        }
    }

    protected override void OnScenarioComplete()
    {
        try
        {
            string soloPath = Path.Combine(Application.dataPath, "../LocalProjectWorkspace/outputs/solo");
            string targetImgFolder = Path.Combine(soloPath, "AllImages"); // 💡 한 곳으로 모을 종착지 폴더

            if (!Directory.Exists(soloPath)) return;
            if (!Directory.Exists(targetImgFolder)) Directory.CreateDirectory(targetImgFolder);

            // 1. SOLO 하위의 모든 sequence 폴더를 탐색합니다.
            string[] subDirs = Directory.GetDirectories(soloPath);
            int fileCounter = 0;

            foreach (string dir in subDirs)
            {
                if (dir.Contains("AllImages")) continue; 

                string[] images = Directory.GetFiles(dir, "*.jpg");
                foreach (string imgPath in images)
                {
                    string newFileName = $"img_{fileCounter}.jpg"; // 겹치지 않게 순번 정렬
                    string destPath = Path.Combine(targetImgFolder, newFileName);

                    File.Copy(imgPath, destPath, true);
                    fileCounter++;
                }
            }
            Debug.Log($"자동 정리 완수: 총 {fileCounter}장의 이미지를 AllImages 폴더 하나로 이쁘게 몰아놨습니다!");
        }
        catch (Exception e)
        {
            Debug.LogWarning("SOLO 폴더 자동 통합 중 오류 발생 (경로를 확인해 주세요): " + e.Message);
        }
    }
}