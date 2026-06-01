using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Perception.Randomization.Randomizers;
using TMPro;

using System.IO; // 파일 쓰기를 위해 추가
[Serializable]
public class FoodTextRandomizer : Randomizer
{
    public TextAsset ingredientFile;

    [Header("데이터 리스트")]
    public List<string> productNames = new List<string> { "매콤 떡볶이", "진한 치즈케이크", "바삭 어니언링", "스파이시 라면", "고소한 우유" };
    public List<string> manufacturers = new List<string> { "(주)푸드테크", "우리식품", "데일리푸드", "팔도강산", "나눔푸드" };
    public List<string> allergies = new List<string> { "대두, 밀 함유", "우유, 토마토 함유", "난류, 메밀 함유", "쇠고기 함유", "새우, 게 함유" };
    public List<string> storages = new List<string> { "실온 보관", "냉동 보관", "서늘한 곳 보관" };
    public List<string> cautions = new List<string> { "신고는 1399", "개봉 후 즉시 섭취", "전자레인지 조리 불가" };

    [Header("라벨 디자인 설정")]
    public float fontSize = 35f;
    public float minWidth = 100f;
    public float maxWidth = 200f;
    public Color[] backgroundColors = new Color[] { Color.white, new Color(0.96f, 0.96f, 0.96f) };

    protected override void OnIterationStart()
    {
        // 0. 파일 체크 디버그
        if (ingredientFile == null)
        {
            Debug.LogError("[에러] Ingredient File이 인스펙터에 연결되지 않았습니다!");
            return;
        }

        string[] allWords = ingredientFile.text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
        var tags = tagManager.Query<TextRandomizerTag>().Where(t => t.name == "FoodTable").ToList();

        if (tags.Count == 0)
        {
            Debug.LogWarning("[경고] 'FoodTable' 태그를 가진 오브젝트를 찾을 수 없습니다.");
        }

        foreach (var tag in tags)
        {
            // 1. 컴포넌트 체크
            var tableImage = tag.GetComponent<Image>();
            var tableRect = tag.GetComponent<RectTransform>();

            // 2. 자식 오브젝트 찾기
            Transform mainTrans = tag.transform.Find("LabelText");
            Transform barTrans = tag.transform.Find("AllergyBar");

            if (mainTrans == null || barTrans == null)
            {
                Debug.LogError($"[에러] {tag.name}의 자식 오브젝트(LabelText/AllergyBar)를 찾을 수 없습니다.");
                continue;
            }

            // 3. 컴포넌트 가져오기
            var mainTmp = mainTrans.GetComponent<TextMeshProUGUI>();
            var allergyBarRect = barTrans.GetComponent<RectTransform>();
            var allergyTmp = barTrans.GetComponentInChildren<TextMeshProUGUI>();

            if (mainTmp == null || allergyTmp == null) continue;

            // --- 데이터 생성 (에러 방지를 위해 루프 안에서 직접 변수 선언) ---
            string chosenProductName = productNames[UnityEngine.Random.Range(0, productNames.Count)];
            string chosenManufacturer = manufacturers[UnityEngine.Random.Range(0, manufacturers.Count)];
            string chosenAllergy = allergies[UnityEngine.Random.Range(0, allergies.Count)];
            string chosenStorage = storages[UnityEngine.Random.Range(0, storages.Count)];
            string chosenCaution = cautions[UnityEngine.Random.Range(0, cautions.Count)];
            string randomIngredients = string.Join(", ", allWords.OrderBy(x => UnityEngine.Random.value).Take(UnityEngine.Random.Range(20, 30)));
            
            if (tableImage != null && backgroundColors.Length > 0)
            {
                tableImage.color = backgroundColors[UnityEngine.Random.Range(0, backgroundColors.Length)];
            }

            // 4. 배경 및 텍스트 적용
            if (tableImage != null)
                tableImage.color = backgroundColors[UnityEngine.Random.Range(0, backgroundColors.Length)];

            List<string> pieces = new List<string>
            {
                $"<b>제품명:</b> {chosenProductName}",
                $"<b>내용량:</b> {UnityEngine.Random.Range(10, 80) * 10}g",
                $"<b>제조원:</b> {chosenManufacturer}",
                $"<b>원재료명:</b> {randomIngredients}",
                $"<b>보관방법:</b> {chosenStorage}",  // 추가
                $"<b>주의사항:</b> {chosenCaution}"    // 추가
            };

            mainTmp.text = string.Join("  |  ", pieces.OrderBy(x => UnityEngine.Random.value));
            mainTmp.fontSize = fontSize;
            mainTmp.color = Color.black;
            mainTmp.alignment = TextAlignmentOptions.Justified;

            allergyTmp.text = $"[알레르기 정보] {chosenAllergy}";
            allergyTmp.fontSize = fontSize;
            allergyTmp.color = Color.white;
            allergyTmp.alignment = TextAlignmentOptions.Center;

            // 5. 3D 실린더 제어
            GameObject cylinder = GameObject.Find("Cylinder");
            if (cylinder != null)
            {
                var renderer = cylinder.GetComponent<Renderer>();
                if (renderer != null)
                {
                    float smoothness = UnityEngine.Random.Range(0.2f, 0.9f);
                    float metallic = UnityEngine.Random.Range(0.0f, 0.5f);
                    renderer.material.SetFloat("_Glossiness", smoothness);
                    renderer.material.SetFloat("_Smoothness", smoothness); // URP 대응
                    renderer.material.SetFloat("_Metallic", metallic);
                }

                // 레이아웃 계산
                float randomWidth = UnityEngine.Random.Range(minWidth, maxWidth);
                mainTmp.rectTransform.sizeDelta = new Vector2(randomWidth - 40f, 2000f);
                mainTmp.ForceMeshUpdate();
                float mainH = mainTmp.preferredHeight;
                mainTmp.rectTransform.sizeDelta = new Vector2(randomWidth - 40f, mainH);

                Canvas.ForceUpdateCanvases();
                float barW = Mathf.Min(allergyBarRect.rect.width, randomWidth - 40f);
                allergyBarRect.sizeDelta = new Vector2(barW, allergyBarRect.rect.height);

                float barH = allergyBarRect.rect.height;
                float totalH = mainH + barH + 60f;
                tableRect.sizeDelta = new Vector2(randomWidth, totalH);

                cylinder.transform.rotation = Quaternion.Euler(0, UnityEngine.Random.Range(0f, 360f), 0);
            }

            // 6. 텍스트 로그 기록 (JSON 대신 TXT 파일)
            string logPath = Path.Combine(Application.persistentDataPath, "label_log.txt");

            // 이미지 파일명과 매칭하기 쉽게 프레임 번호를 맨 앞에 둡니다.
            string logEntry = 
                            $"Ingredients: {randomIngredients}"; // <- 원재료명 추가!
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
}