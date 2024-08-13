#在專案跟目錄下新增data.yaml
#新增圖片訓練時，須把./car/train/labels.cache及./car/valid/labels.cache刪除
import os
import shutil
import time

from ultralytics import YOLO

if __name__=='__main__':
    train_path='./runs/detect/train'
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    model=YOLO('yolov8n.pt')
    print('開始訓練...')
    t1=time.time()
    model.train(data='./data.yaml',epochs=200,imgsz=640)
    t2 = time.time()
    print(f'訓練時間花費:{t2-t1}秒')
    path=model.export()#取得訓練後模型路徑
    print(f'模型路徑:{path}')