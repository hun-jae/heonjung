import os
import torch
torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

train_data = pd.read_csv("/home/ubuntu/lm/train.csv")
test_data = pd.read_csv("/home/ubuntu/lm/test.csv")
labels = train_data['N_category'].to_list()

def get_train_data(data):
    img_path_list = data['img_path'].to_list()
    img_path_list = list(map(lambda path: '/home/ubuntu/lm' + path[1:], img_path_list))
    return img_path_list, labels

def get_test_data(data):
    img_path_list = data['img_path'].to_list()
    img_path_list = list(map(lambda path: '/home/ubuntu/lm' + path[1:], img_path_list))
    return img_path_list

all_img_path, all_label = get_train_data(train_data)
test_img_path = get_test_data(test_data)

import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None): #필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index): #index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self): #길이 return
        return len(self.img_path_list)

# Train : Validation = 0.8 : 0.2 Split
train_len = int(len(all_img_path)*0.8)
vali_len = int(len(all_img_path)*0.2)

train_img_path = all_img_path[:train_len]
train_label = all_label[:train_len]

vali_img_path = all_img_path[train_len:]
vali_label = all_label[train_len:]

train_transform = transforms.Compose([
                    transforms.ToPILImage(), #Numpy배열에서 PIL이미지로
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize([CFG['WIDTH'], CFG['HEIGHT']]), #이미지 사이즈 변형
                    transforms.ToTensor(), #이미지 데이터를 tensor
                    transforms.Normalize(mean=0.5, std=0.5) #이미지 정규화

                    ])

test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize([CFG['WIDTH'], CFG['HEIGHT']]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                    ])

# Get Dataloader

#CustomDataset class를 통하여 train dataset생성
train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform)
#만든 train dataset를 DataLoader에 넣어 batch 만들기
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0) #BATCH_SIZE : 24

#vaildation 에서도 적용
vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=test_transform)
vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


train_batches = len(train_loader)
vali_batches = len(vali_loader)

print('total train imgs :',train_len,'/ total train batches :', train_batches)
print('total valid imgs :',vali_len, '/ total valid batches :', vali_batches)

from tqdm.auto import tqdm
import torch.nn.functional as F

import torch.nn as nn # 신경망들이 포함됨
#import torch.nn.init as init # 텐서에 초기값을 줌

class CNNclassification(nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)

        self.layer2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.layer6 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer7 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer8 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential(
            nn.Linear(4194304, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.layer1(x) #1층

        x = self.layer2(x) #2층

        x = self.layer3(x) #3층

        x = self.layer4(x) #4층

        x = self.layer5(x) #5층

        x = self.layer6(x) #6층

        x = self.layer7(x) #7층

        x = self.layer8(x) #8층

        x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열
        #x = x.unsqueeze(1)
        out = self.fc_layer(x)
        out = torch.flatten(out)
        return out

def predict(model, train_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        cnt = 0
        for img, label in tqdm(iter(train_loader)):
            cnt += 1
            if cnt == 10:
                break
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(keepdim=True)

            model_pred.extend(pred_logit.tolist())
    return model_pred

test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# Validation Accuracy가 가장 뛰어난 모델을 불러옵니다.
checkpoint = torch.load('/home/ubuntu/lm/saved/best_model.pth')
model = CNNclassification().to(device)
model.load_state_dict(checkpoint)

# Inference
preds = predict(model, train_loader, device)
print(preds)
