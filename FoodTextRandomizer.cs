using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Perception.Randomization.Randomizers;
using TMPro;
using UnityEngine.Perception.Randomization.Scenarios;

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

    private string[] randomCountries = { "국산", "외국산", "미국산", "네덜란드산", "말레이시아산", "중국산", "호주산", "독일산" };

    [Header("방해용 UI 프리팹 설정")]
    public GameObject barcodePrefab;
    public GameObject noiseTitlePrefab;

    [Header("배경 도형 설정")]
    public int shapeCount =100;

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

            var selectedWords = allWords.OrderBy(x => UnityEngine.Random.value).Take(UnityEngine.Random.Range(20, 46)).ToList();

            List<string> uiIngredientsList = new List<string>();
            List<string> logIngredientsList = new List<string>();

            foreach (string word in selectedWords)
            {
                string cleanWord = word.Trim();
                if (string.IsNullOrEmpty(cleanWord)) continue;

                logIngredientsList.Add(cleanWord);

                float noiseDice = UnityEngine.Random.value;
                if (noiseDice < 0.35f) uiIngredientsList.Add($"{cleanWord}({randomCountries[UnityEngine.Random.Range(0, randomCountries.Length)]})");
                else if (noiseDice < 0.50f) uiIngredientsList.Add($"{cleanWord}(포도당:{randomCountries[UnityEngine.Random.Range(0, randomCountries.Length)]})");
                else if (noiseDice < 0.60f) uiIngredientsList.Add($"{cleanWord}({cleanWord}함유)");
                else uiIngredientsList.Add(cleanWord);
            }

            string uiIngredientsText = string.Join(", ", uiIngredientsList);
            string logIngredientsText = string.Join(", ", logIngredientsList);

            //  배경색 및 밝기 제어
            Color randomBgColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);
            if (tableImage != null) tableImage.color = randomBgColor;

            float luminance = 0.2126f * randomBgColor.r + 0.7152f * randomBgColor.g + 0.0722f * randomBgColor.b;
            Color mainTextColor, allergyBarColor, allergyTextColor;

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

            //  폰트 스타일 다양화
            if (fontAssets != null && fontAssets.Length > 0)
            {
                TMP_FontAsset targetFont = fontAssets[UnityEngine.Random.Range(0, fontAssets.Length)];
                mainTmp.font = targetFont;
                allergyTmp.font = targetFont;
            }
            mainTmp.characterSpacing = UnityEngine.Random.Range(-6f, 3f);
            mainTmp.lineSpacing = UnityEngine.Random.Range(-9f, 12f);

            // UI 조립
            List<string> UI_pieces = new List<string>
            {
                $"<b>제품명:</b> {chosenProductName}",
                $"<b>제조원:</b> {chosenManufacturer}",
                $"<b>보관방법:</b> {chosenStorage}",
                $"<b>원재료명:</b> {uiIngredientsText}",
                $"<b>주의사항:</b> {chosenCaution} | {chosenFactoryCaution}"
            };

            mainTmp.text = string.Join("  |   ", UI_pieces);
            mainTmp.fontSize = fontSize;
            mainTmp.color = mainTextColor;
            mainTmp.alignment = TextAlignmentOptions.Left;

            allergyTmp.text = $"[알레르기 정보] {chosenAllergy}";
            allergyTmp.fontSize = fontSize;
            allergyTmp.color = allergyTextColor;
            allergyTmp.alignment = TextAlignmentOptions.Center;

            var allergyBarImage = barTrans.GetComponent<Image>();
            if (allergyBarImage != null) allergyBarImage.color = allergyBarColor;

            // Cylinder 제어
            GameObject cylinder = GameObject.Find("Cylinder");
            if (cylinder != null)
            {
                var renderer = cylinder.GetComponent<Renderer>();
                if (renderer != null)
                {
                    Color randomCylinderColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);

                    if (renderer.material.HasProperty("_BaseColor"))
                        renderer.material.SetColor("_BaseColor", randomCylinderColor);
                    else
                        renderer.material.SetColor("_Color", randomCylinderColor);

                    float smoothness = 0.5f, metallic = 0.0f, bumpScale = 0.0f;
                    float materialDice = UnityEngine.Random.value;

                    if (materialDice < 0.4f) { metallic = UnityEngine.Random.Range(0.7f, 0.95f); smoothness = UnityEngine.Random.Range(0.2f, 0.4f); bumpScale = UnityEngine.Random.Range(0.2f, 0.8f); }
                    else if (materialDice < 0.7f) { metallic = UnityEngine.Random.Range(0.0f, 0.1f); smoothness = UnityEngine.Random.Range(0.75f, 0.95f); bumpScale = (UnityEngine.Random.value > 0.4f) ? UnityEngine.Random.Range(0.2f, 1.0f) : 0f; }
                    else { smoothness = UnityEngine.Random.Range(0.05f, 0.15f); }

                    renderer.material.SetFloat("_Glossiness", smoothness);
                    renderer.material.SetFloat("_Smoothness", smoothness);
                    renderer.material.SetFloat("_Metallic", metallic);
                    if (renderer.material.HasProperty("_BumpScale")) renderer.material.SetFloat("_BumpScale", bumpScale);
                }
                cylinder.transform.rotation = Quaternion.Euler(0, UnityEngine.Random.Range(0f, 360f), 0);
            }

            float chosenTableWidth = (UnityEngine.Random.value > 0.5f) ? UnityEngine.Random.Range(700f, maxWidth) : UnityEngine.Random.Range(550f, 700f);
            float textWidth = chosenTableWidth - 80f;

            mainTmp.rectTransform.sizeDelta = new Vector2(textWidth, 5000f);
            mainTmp.ForceMeshUpdate();
            float mainH = mainTmp.preferredHeight;
            mainTmp.rectTransform.sizeDelta = new Vector2(textWidth, mainH);

            Canvas.ForceUpdateCanvases();
            float barW = Mathf.Min(allergyBarRect.rect.width, textWidth);
            allergyBarRect.sizeDelta = new Vector2(barW, allergyBarRect.rect.height);

            float totalHeight = mainH + allergyBarRect.rect.height + UnityEngine.Random.Range(200f, 350f);
            tableRect.sizeDelta = new Vector2(chosenTableWidth, totalHeight);

            // --------------------------------------------------------
            // 빈 공간에 큼직하게 떼거지 스폰
            // --------------------------------------------------------
            Transform oldNoise = tag.transform.Find("Generated_Noise_Objects");
            if (oldNoise != null)
            {
                oldNoise.name = "Obsolete_Noise"; // 이름 바꿔서 중복 추적 방지
                GameObject.Destroy(oldNoise.gameObject);
            }
            GameObject noiseContainer = new GameObject("Generated_Noise_Objects");
            noiseContainer.transform.SetParent(tag.transform, false);

            float tableHalfW = chosenTableWidth * 0.5f;
            float tableHalfH = totalHeight * 0.5f;

            for (int zone = 0; zone < 4; zone++)
            {
                
                    Vector3 targetSpawnPos = Vector3.zero;
                    switch (zone)
                    {
                        case 0: targetSpawnPos.x = UnityEngine.Random.Range(-tableHalfW * 0.7f, tableHalfW * 0.7f); targetSpawnPos.y = tableHalfH + UnityEngine.Random.Range(10f, 40f); break;
                        case 1: targetSpawnPos.x = UnityEngine.Random.Range(-tableHalfW * 0.7f, tableHalfW * 0.7f); targetSpawnPos.y = -tableHalfH - UnityEngine.Random.Range(10f, 40f); break;
                        case 2: targetSpawnPos.x = -tableHalfW - UnityEngine.Random.Range(20f, 50f); targetSpawnPos.y = UnityEngine.Random.Range(-tableHalfH * 0.6f, tableHalfH * 0.6f); break;
                        case 3: targetSpawnPos.x = tableHalfW + UnityEngine.Random.Range(20f, 50f); targetSpawnPos.y = UnityEngine.Random.Range(-tableHalfH * 0.6f, tableHalfH * 0.6f); break;
                    }
                    targetSpawnPos.z = -1f;

                    GameObject selectedPrefab = (UnityEngine.Random.value > 0.4f) ? barcodePrefab : noiseTitlePrefab;
                    if (selectedPrefab != null)
                    {
                        GameObject spawnedNoise = GameObject.Instantiate(selectedPrefab, noiseContainer.transform);
                        RectTransform noiseRT = spawnedNoise.GetComponent<RectTransform>();
                        if (noiseRT != null) noiseRT.anchoredPosition = new Vector2(targetSpawnPos.x, targetSpawnPos.y);
                        else spawnedNoise.transform.localPosition = targetSpawnPos;

                        spawnedNoise.transform.localRotation = Quaternion.Euler(0, 0, UnityEngine.Random.Range(-20f, 20f));
                        spawnedNoise.transform.localScale = new Vector3(UnityEngine.Random.Range(1.5f, 3.0f), UnityEngine.Random.Range(1.2f, 2.0f), 1f);

                        var spawnedTmp = spawnedNoise.GetComponentInChildren<TextMeshProUGUI>();
                        if (spawnedTmp != null)
                        {
                            Color c = mainTextColor; 
                            c.a = UnityEngine.Random.Range(0.15f, 0.3f);
                            spawnedTmp.color = c;
                        }

                    }
                
            }

            GenerateBackgroundShapes();

            FoodDataLogger.SaveIngredientsLog(logIngredientsText);
        }
    }

    private void GenerateBackgroundShapes()
    {
        GameObject oldShapes = GameObject.Find("Background_Geometry_Noise");
        if (oldShapes != null)
        {
            oldShapes.name = "Obsolete_Geometry";
            GameObject.Destroy(oldShapes);
        }

        GameObject shapeHolder = new GameObject("Background_Geometry_Noise");

  
        GameObject cylinder = GameObject.Find("Cylinder");
        Vector3 cylinderPos = (cylinder != null) ? cylinder.transform.position : new Vector3(0, 0, -5.9f);

        shapeHolder.transform.position = cylinderPos;

        for (int i = 0; i < shapeCount; i++)
        {
            int shapeDice = UnityEngine.Random.Range(0, 3);
            PrimitiveType selectedType = PrimitiveType.Cube;
            if (shapeDice == 1) selectedType = PrimitiveType.Sphere;
            if (shapeDice == 2) selectedType = PrimitiveType.Cylinder;

            GameObject shape = GameObject.CreatePrimitive(selectedType);

            shape.transform.SetParent(shapeHolder.transform, false);

         
            float posX = UnityEngine.Random.Range(-10.0f,10.0f);
            float posY = UnityEngine.Random.Range(-10.0f,10.0f);

            float posZ = UnityEngine.Random.Range(8.0f, 12.5f);
            shape.transform.localPosition = new Vector3(posX, posY, posZ);

  
            float scaleX = UnityEngine.Random.Range(4.0f, 8.0f);
            float scaleY = UnityEngine.Random.Range(4.0f, 8.0f);
            float scaleZ = UnityEngine.Random.Range(5.0f, 8.0f);
            shape.transform.localScale = new Vector3(scaleX, scaleY, scaleZ);

 
            shape.transform.localRotation = Quaternion.Euler(
                UnityEngine.Random.Range(0f, 360f),
                UnityEngine.Random.Range(0f, 360f),
                UnityEngine.Random.Range(0f, 360f)
            );

            Renderer shapeRenderer = shape.GetComponent<Renderer>();
            if (shapeRenderer != null)
            {
                Color randomColor = Color.HSVToRGB(
                    UnityEngine.Random.Range(0f, 1f),
                    UnityEngine.Random.Range(0.6f, 1.0f),
                    UnityEngine.Random.Range(0.5f, 1.0f)
                );

                Material tempMat = null;
                Shader urpUnlit = Shader.Find("Universal Render Pipeline/Unlit");
                if (urpUnlit != null) tempMat = new Material(urpUnlit);
                if (tempMat == null) tempMat = shapeRenderer.material;

                if (tempMat != null)
                {
                    if (tempMat.HasProperty("_BaseColor")) tempMat.SetColor("_BaseColor", randomColor);
                    if (tempMat.HasProperty("_Color")) tempMat.SetColor("_Color", randomColor);
                    shapeRenderer.material = tempMat;
                }
            }

            shape.layer = 4;

            Collider c = shape.GetComponent<Collider>();
            if (c != null) GameObject.Destroy(c);
        }
    }
}