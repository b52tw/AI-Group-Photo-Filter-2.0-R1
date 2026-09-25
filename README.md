# chunba AI Photo Classifier v2.0 R1

GUI 內部署名：**峻爸製作**

## 主要更新
- 程式、workflow、EXE、Artifact 檔名全部使用英文 `chunba`。
- 快速掃描後改成照片縮圖牆，可點圖開大圖預覽。
- 可單張勾選、全選此頁、全選篩選結果、只看已選。
- 每次掃描可自由單選／複選：純人物辨識、人數分類、日期分類、場景分類、指定人物搜尋。
- AI內容標籤保留，並顯示「AI判斷依據」。
- 保留 v2.0 的 YuNet + ArcFace 指定人物比對。
- 「精確進階分析所選」只重跑已勾選照片，使用更高解析度與大合照分塊人臉偵測。
- 操作按鈕移到畫面上方，不再放最下面。
- 原始照片不移動、不刪除；按「輸出所選分類檔案」才複製。
- 每次掃描建立不重複資料夾：`chunba_YYYYMMDD_001`、`002`……
- 匯出照片使用英文安全檔名：`chunba_YYYYMMDD_001_00001.jpg`。

## 每次會留下
- `chunba_YYYYMMDD_001_preview.csv`
- `chunba_YYYYMMDD_001_advanced.csv`
- `chunba_YYYYMMDD_001_export_manifest.csv`

## 網路 AI 選項
已加入 Gemini 2.5 Flash-Lite 選用功能，只用於場景與內容標籤二次判讀，不拿網路 AI 做指定人物身分比對。
需要使用者自己的 Google AI Studio API Key。預設只在「精確進階」使用；若勾「快速掃描也使用」會較慢且受配額影響。啟用時，縮小後的照片會傳送到服務商，請先確認照片隱私與服務條款。

## GitHub 網頁上傳
如果 `.github` 資料夾沒有跟著上傳，根目錄已另外放一份 `build-windows.yml`：
1. 先把根目錄檔案上傳並提交。
2. 點進 `build-windows.yml` → 編輯。
3. 把檔名改成 `.github/workflows/build-windows.yml`。
4. 提交後到 Actions 執行 `Build chunba AI Photo Classifier v2.0 R1`。
5. 成功後下載 Artifact：`chunba-ai-photo-classifier-v2-0-r1-windows`。

本版 Artifact 直接上傳 EXE，不再先製作中文 ZIP，避免上一版找不到 ZIP 的問題。

## 授權提醒
本機使用 OpenCV YuNet、YOLOv8n、ArcFace 模型。若要公開散布或商業販售，請另外確認各模型與權重的最新授權條款。
