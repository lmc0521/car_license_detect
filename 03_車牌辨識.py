#1.tesseract安裝:https://digi.bib.uni-mannheim.de/tesseract/
#2.將D:\Program Files\Tesseract-OCR新增環境變數
#3.
import os
import platform
import pylab as plt
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO

def text(img, text, xy=(0, 0), color=(0, 0, 0), size=12):
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

model=YOLO('./car.pt')
path="./images"

for i,file in enumerate(os.listdir(path)):
    full=os.path.join(path, file)
    img=cv2.imdecode(np.fromfile(full, dtype=np.uint8), cv2.IMREAD_COLOR)
    img=img[:,:,::-1].copy()

    results = model.predict(img, save=False)
    boxes = results[0].boxes.xyxy
    for box in boxes:
        # box=box.cpu().numpy()
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        try:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tmp=img[y1:y2,x1:x2].copy()
            tmp=cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)
            license=pytesseract.image_to_string(tmp,lang='eng',config='--psm 11')
            img=text(img,license,(x1,y1-20),(0,255,0),100)
        except:
            pass
        # tmp = cv2.cvtColor(img[y1:y2, x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        # img = text(img, license, (x1, y1 - 20), (0, 255, 0), 25)
    plt.subplot(2, 3, i + 1)
    plt.axis("off")
    plt.imshow(img)

plt.show()