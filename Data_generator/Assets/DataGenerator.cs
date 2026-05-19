using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;



public class DataGenerator : MonoBehaviour
{
    [Header("���� ����")]
    public GameObject targetCube;    // ȸ���� ť��
    public TextMeshPro targetText;  // ť�꿡 ���� 3D TextMeshPro

    private string[] allIngredients;
    private int fileCount = 0;
    private bool isLoaded = false;
    [Header("���� ����")]
    public Material[] randomMaterials;
    // 1. ������ �ε� �Լ� (���� ������ ���� �и�)

    [Header("���� ����")]
    public Light targetLight; // Hierarchy�� Point Light�� ���⿡ �巡��!
    [Header("��Ʈ ����")]
    private TMP_FontAsset[] allFonts; // Resources���� �ε��� ��Ʈ��

    [Header("��� ����")]
    public Renderer backdropRenderer; // ���� Plane�� Renderer ����
    private Sprite[] bgSprites;       // ��� �̹��� ����Ʈ
    void LoadIngredients()
    {
        TextAsset textData = Resources.Load<TextAsset>("ingredients");
        if (textData != null)
        {
            // �ٹٲ����� ������ �� ĭ ����
            allIngredients = textData.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            isLoaded = true;
            Debug.Log($"<color=cyan>�ε� ����:</color> {allIngredients.Length}���� ������ �����Խ��ϴ�.");
        }
        else
        {
            Debug.LogError("<color=red>����:</color> Resources/ingredients.txt ������ ã�� �� �����ϴ�!");
        }
    }

    void Start()
    {
        // 1. ���� ������ �̸� �ε� (���� ������)
        TextAsset textData = Resources.Load<TextAsset>("ingredients");
        if (textData != null)
        {
            allIngredients = textData.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            Debug.Log("<color=cyan>���� ������ �ε� �Ϸ�!</color>");
        }

        // 2. ��� �̹��� �������� �� �� �ܾ����
        // Resources/Backgrounds ���� �ȿ� �̹������� �־�� ��!
        bgSprites = Resources.LoadAll<Sprite>("Backgrounds");

        if (bgSprites.Length > 0)
        {
            Debug.Log($"<color=lime>��� �̹��� {bgSprites.Length}�� �ε� ����!</color>");
        }
        else
        {
            Debug.LogWarning("Backgrounds ������ ����ְų� Resources ���� Ȯ���� �ʿ��մϴ�.");
        }

        allFonts = Resources.LoadAll<TMP_FontAsset>("Fonts");
        if (allFonts.Length > 0)
            Debug.Log($"<color=orange>��Ʈ {allFonts.Length}�� �ε� �Ϸ�!</color>");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // 1. �����Ͱ� ������ '�� �� ��'�� ���⼭ �ε�
            if (allIngredients == null || allIngredients.Length == 0)
            {
                TextAsset textData = Resources.Load<TextAsset>("ingredients");
                if (textData != null)
                    allIngredients = textData.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            }

            // 2. �����Ͱ� Ȯ���� ���� ���� ����
            if (allIngredients != null && allIngredients.Length > 0)
            {
                GenerateOneData();
            }
        }
    }
    IEnumerator AutoGenerateLoop(int count)
    {
        Debug.Log($"<color=orange>{count}�� ���� ������ �����մϴ�...</color>");
        for (int i = 0; i < count; i++)
        {
            GenerateOneData();
            // ĸó�� �������� ������ �ʵ��� �� ������ ���
            yield return new WaitForEndOfFrame();
        }
        Debug.Log($"<color=yellow>�� {count}���� ������ ���� �Ϸ�!</color>");
    }
    void GenerateOneData()
    {
        // --- 1. TMP �ڵ� ũ�� ���� �� ���� ���� ---
        targetText.enableAutoSizing = true;
        targetText.fontSizeMin = 0.2f; // �ּ� ũ��
        targetText.fontSizeMax = 0.6f; // �ִ� ũ��
        targetText.enableWordWrapping = true; // �ٹٲ� �ʼ�

        // ť�� ����(1x1) �ȿ��� �ؽ�Ʈ ������ 0.9�� ����
        targetText.rectTransform.sizeDelta = new Vector2(0.9f, 0.9f);
        targetText.rectTransform.localPosition = new Vector3(0, 0, -0.51f);

        // --- 2. �ؽ�Ʈ ���� ���� ---
        int pickCount = Random.Range(8, 15);
        List<string> selected = new List<string>();
        for (int i = 0; i < pickCount; i++)
        {
            string word = allIngredients[Random.Range(0, allIngredients.Length)].Trim();
            if (Random.value > 0.9f) word += $" {Random.Range(1, 99)}%";
            selected.Add(word);
        }

        string[] prefixes = { "������: ", "[������] ", "����: " };
        string fullContent = prefixes[Random.Range(0, prefixes.Length)] + string.Join(", ", selected);

        // --- 3. �ؽ�Ʈ ���� �� '����ȭ' ó�� ---
        targetText.text = fullContent;

        // [�߿�] TMP�� ������ ������Ʈ���Ѽ� �߸� ���ڰ� �ִ��� ����ϰ� ��
        targetText.ForceMeshUpdate();

        // ������ ȭ�鿡 '���̴�' ���� ����ŭ�� �������� �߶� (����ȭ �ٽ�)
        int visibleCharacters = targetText.textInfo.characterCount;
        string finalLabelText = fullContent;

        // ���� ���ڰ� �ʹ� ���Ƽ� �߷ȴٸ�, �߸� ��ŭ�� �ؽ�Ʈ �����͵� ����
        if (visibleCharacters < fullContent.Length && visibleCharacters > 0)
        {
            finalLabelText = fullContent.Substring(0, visibleCharacters);
        }

        // --- 4. ��Ÿ ���� ��� (������ ����) ---
        // (����, ����, ���, ȸ�� ����...)
        float rotX = Random.Range(-20f, 20f);
        float rotY = Random.Range(-20f, 20f);
        float rotZ = Random.Range(-5f, 5f);
        targetCube.transform.localRotation = Quaternion.Euler(rotX, rotY, rotZ);
        if (bgSprites != null && bgSprites.Length > 0)
        {
            int bgIndex = Random.Range(0, bgSprites.Length);
            // ��� Plane�� ��Ƽ���� �ؽ�ó�� ���� ��������Ʈ�� �ؽ�ó�� ����
            backdropRenderer.material.mainTexture = bgSprites[bgIndex].texture;
        }
        backdropRenderer.material.color = new Color(Random.Range(0.5f, 1f), Random.Range(0.5f, 1f), Random.Range(0.5f, 1f));


        // --- 5. ���� (����ȭ�� finalLabelText ���) ---
        string folderPath = Path.Combine(Application.dataPath, "GeneratedData");
        
        // 폴더가 없으면 생성
        if (!Directory.Exists(folderPath)) {
            Directory.CreateDirectory(folderPath);
        }

        string fileName = $"Food3D_{fileCount}_{System.DateTime.Now:HHmmss}.png";
        
        // 1. 스크린샷 저장 경로 수정 (folderPath 결합)
        string screenshotPath = Path.Combine(folderPath, fileName);
        ScreenCapture.CaptureScreenshot(screenshotPath);

        // 2. labels.txt 저장 경로 수정 (folderPath 내부로 이동)
        string logPath = Path.Combine(folderPath, "labels.txt");
        
        // 파일명과 라벨 텍스트 저장 (\t는 탭, \n은 줄바꿈)
        File.AppendAllText(logPath, fileName + "\t" + finalLabelText + "\n");

        fileCount++;
    }
}