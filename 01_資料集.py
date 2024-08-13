#pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio===2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
#pip install ultralytics pytesseract
#資料集下載 :https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download
import os
import random
import shutil
data_path='./car'
train_path='./train'
valid_path='./valid'
if os.path.exists(train_path):
    shutil.rmtree(train_path)
if os.path.exists(valid_path):
    shutil.rmtree(valid_path)
os.makedirs(os.path.join(train_path, 'images'))
os.makedirs(os.path.join(train_path, 'labels'))
os.makedirs(os.path.join(valid_path, 'images'))
os.makedirs(os.path.join(valid_path, 'labels'))

files=[os.path.splitext(file)[0]
       for file in os.listdir(os.path.join(data_path, "images"))]
random.shuffle(files)
mid=int(len(files)*0.8)
for file in files[:mid]:
    source=os.path.join(data_path, "images", f'{file}.png')
    target=os.path.join(train_path,"images", f'{file}.png')
    print(source, target)
    shutil.copy(source, target)

    source=os.path.join(data_path, "labels", f'{file}.txt')
    target=os.path.join(train_path,"labels", f'{file}.txt')
    print(source, target)
    shutil.copy(source, target)

for file in files[mid:]:
    source=os.path.join(data_path, "images", f'{file}.png')
    target=os.path.join(valid_path,"images", f'{file}.png')
    print(source, target)
    shutil.copy(source, target)

    source=os.path.join(data_path, "labels", f'{file}.txt')
    target=os.path.join(valid_path,"labels", f'{file}.txt')
    print(source, target)
    shutil.copy(source, target)