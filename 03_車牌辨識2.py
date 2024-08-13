'''
tesseract 安裝
1. 下載及安裝 tesseract :https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe
2. 將 C:\Program Files\Tesseract-OCR 新增到系統 path 中
2. 安裝及設定好路徑, Pycharm 必需關掉重新進入，路徑才會生效
3. pip install pytesseract

'''
import os
import platform
import pylab as plt
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO

def text(img, text, xy=(0, 0), color=(0, 0, 0), size=20):
    pil = Image.fromarray(img)
    s = platform.system()
    if s == "Linux":
        font =ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size)
    elif s == "Darwin":
        font = ImageFont.truetype('....', size)
    else:
        font = ImageFont.truetype('simsun.ttc', size)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return np.asarray(pil)
model=YOLO("./car.pt")
path="./images"
for i, file in enumerate(os.listdir(path)):
    file=os.path.join(path, file)
    img=cv2.imdecode(
        np.fromfile(file, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )[:,:,::-1].copy()

    results=model.predict(img, save=False)
    boxes=results[0].boxes.xyxy
    for box in boxes:
        box=box.cpu().numpy()
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        try:
            cv2.rectangle(img,(x1, y1), (x2, y2), (0,255,0) , 2)
            tmp=img[y1:y2, x1:x2].copy()
            tmp=cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
            license=pytesseract.image_to_string(tmp, lang='eng', config='--psm 11')
            img=text(img, license, (x1, y1-20), (0,255,0),100)
        except:
            pass
    plt.subplot(1,3,i+1)
    plt.axis("off")
    plt.imshow(img)
plt.show()