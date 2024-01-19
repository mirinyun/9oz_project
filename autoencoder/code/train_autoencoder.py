import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
import torch.utils as utils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 생성되는 이미지를 관찰하기 위함입니다. 3차원 플롯을 그리는 용도입니다.
from matplotlib import cm  # 데이터포인트에 색상을 입히는 것에 사용됩니다.
import numpy as np
import os
import shutil


# 하이퍼파라미터 준비
EPOCH = 50
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

# data_path_train = './content/img_train'  # 트레이닝 이미지가 있는 폴더 경로
# output_path_train = './content/img_train_sorted'  # 클래스별로 정리된 폴더가 생성될 경로

# # 클래스별로 폴더를 생성하고 이미지를 해당 폴더로 이동하는 함수
# def organize_images(data_path, output_path):
#     os.makedirs(output_path, exist_ok=True)  # 출력 폴더 생성

#     for filename in os.listdir(data_path):
#         if filename.endswith('.png'):  # 확장자가 jpg인 경우에만 처리
#             class_name = filename[4:6]
#             class_path = os.path.join(output_path, class_name)

#             os.makedirs(class_path, exist_ok=True)  # 클래스별 폴더 생성

#             old_path = os.path.join(data_path, filename)
#             new_path = os.path.join(class_path, filename)

#             shutil.move(old_path, new_path)  # 이미지 이동

# # 이미지 정리 수행
# organize_images(data_path_train, output_path_train)

# import torch
# from torchvision import datasets, transforms

# 이미지가 있는 폴더의 경로
data_path_train = './content/img_test_sorted_A20+21+23'

# 폴더 내의 이미지를 불러오고 변형하는 코드
train_dataset = datasets.ImageFolder(
    root=data_path_train,
    transform=transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()  # Tensor로 변환
    ])
)

# DataLoader 설정
BATCH_SIZE = 64  # 적절한 배치 크기로 설정하세요
custom_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)

# 오토인코더 모듈 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 인코더는 간단한 신경망으로 분류모델처럼 생겼습니다.
        self.encoder = nn.Sequential(  # nn.Sequential을 사용해 encoder와 decoder 두 모듈로 묶어줍니다.
            nn.Linear(299 * 299 * 3, 128),  # 차원을 28*28에서 점차 줄여나갑니다.
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # 입력의 특징을 3차원으로 압축합니다 (출력값이 바로 잠재변수가 됩니다.)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),  # 디코더는 차원을 점차 28*28로 복원합니다.
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 299 * 299 * 3),
            nn.Sigmoid(),  # 픽셀당 0과 1 사이로 값을 출력하는 sigmoid()함수를 추가합니다.
        )
    def forward(self, x):
        encoded = self.encoder(x)  # encoder는 encoded라는 잠재변수를 만들고
        decoded = self.decoder(encoded)  # decoder를 통해 decoded라는 복원이미지를 만듭니다.
        return encoded, decoded

def run():
    torch.multiprocessing.freeze_support()

    # 학습하기 위한 함수
    def train(autoencoder, custom_loader):
        autoencoder.train()
        for step, (x, label) in enumerate(custom_loader):
            x = x.view(-1, 299 * 299 * 3).to(DEVICE)
            y = x.view(-1, 299 * 299 * 3).to(DEVICE)  # x(입력)와 y(대상 레이블)모두 원본이미지(x)인 것을 주의해야 합니다.
            label = label.to(DEVICE)
        
            encoded, decoded = autoencoder(x)

            loss = criterion(decoded, y)  # decoded와 원본이미지(y) 사이의 평균제곱오차를 구합니다
            optimizer.zero_grad()  # 기울기에 대한 정보를 초기화합니다.
            # scaler.unscale_(optimizer)  # Unscales the gradients of optimizer's assigned params in-place
            # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)  # Gradient clipping
            # scaler.scale(loss).backward()  # 기울기를 구합니다.
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()  # 최적화를 진행합니다.

    autoencoder = Autoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    # Adam()을 최적화함수로 사용합니다. Adam은 SGD의 변형함수이며 학습중인 기울기를 참고하여 학습 속도를 자동으로 변화시킵니다.
    criterion = nn.MSELoss()  # 원본값과 디코더에서 나온 값의 차이를 계산하기 위해 평균제곱오차(Mean Squared Loss) 오차함수를 사용합니다.

    # 처음 5개 이미지를 가져와서 하나의 텐서로 합치기
    num_batches_to_extract = 7500
    view = []

    for i, (images, labels) in enumerate(train_dataset):
        if i < num_batches_to_extract:
            if i%1660==0: #랜덤으로 5개뽑기
                print(f"Img {i + 1}: {images.size()} - Labels: {labels}")

                # 이미지를 하나의 텐서로 합치기
                view.append(images.view(-1, 3, 299, 299))
        else:
            break
    # view_data를 하나의 텐서로 변환
    view_data = torch.cat(view, dim=0)

    #학습하기

    for epoch in range(1, EPOCH+1):
        print("1")
        train(autoencoder, custom_loader)
        print("2")
        # 디코더에서 나온 이미지를 시각화 하기
        # 앞서 시각화를 위해 남겨둔 5개의 이미지를 한 이폭만큼 학습을 마친 모델에 넣어 복원이미지를 만듭니다.
        test_x = view_data.to(DEVICE)
        print("3")

        _, decoded_data = autoencoder(test_x.view(-1, 299*299*3))
        print("4")

        # 원본과 디코딩 결과 비교해보기
        f, a = plt.subplots(2, 5, figsize=(5, 2))
        print("[Epoch {}]".format(epoch))
        for i in range(5):
            out_img = torch.squeeze(view_data.cpu())
            img=out_img.permute(0,2,3,1)[i]
            a[0][i].imshow(img)
            a[0][i].set_xticks(()); a[0][i].set_yticks(())
        print("5")

        for i in range(5):
            img = decoded_data.to("cpu").data[i]
            img=img.reshape(-1, 3, 299, 299)
            img=img.permute(0,2,3,1)[0]
            # CUDA를 사용하면 모델 출력값이 GPU에 남아있으므로 .to("cpu") 함수로 일반메모리로 가져와 numpy행렬로 변환합니다.
            # cpu를 사용할때에도 같은 코드를 사용해도 무방합니다.
            a[1][i].imshow(img)
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
    plt.show()
    model_scripted = torch.jit.script(autoencoder)  # TorchScript 형식으로 내보내기
    model_scripted.save('autoencoder_small.pt')  # 저장하기

if __name__ == '__main__':
    run()   