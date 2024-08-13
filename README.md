安裝套件
===
- pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio===2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
- pip install ultralytics pytesseract

下載訓練資料
---
- 請到 kaggle 下載 [car-plate-license](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download "游標顯示") 車牌圖片訓練資料，解開後，將 images 及 annotations 目錄移到專案中的 car 目錄下。解開 .zip 檔後，將 “archive” 目錄改為 “car”。

xml2txt
---
- car/annotations 下的資料為 xml 格式，需將 xml 轉成 txt 格式，請新增 xml2txt.py 檔。

分割訓練及驗証資料
---

data.yaml 資料集設定
---

訓練模型
---

車牌偵測及辨識
---