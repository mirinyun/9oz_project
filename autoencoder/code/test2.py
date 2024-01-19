### 오토인코더로 이미지 특징 추출하기
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
import torch.utils as utils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #생성되는 이미지를 관찰하기 위함입니다. 3차원 플롯을 그리는 용도입니다.
from matplotlib import cm # 데이터포인트에 색상을 입히는 것에 사용됩니다.
import numpy as np
import os
import pickle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

# 이미지가 있는 폴더의 경로
data_path_test = './content/img_test_sorted_A20+21+23'

# 폴더 내의 이미지를 불러오고 변형하는 코드
test_dataset = datasets.ImageFolder(
    root=data_path_test,
    transform=transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()         # Tensor로 변환
    ])
)


# DataLoader 설정
BATCH_SIZE = 64  # 적절한 배치 크기로 설정하세요
custom_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)
# 데이터셋 파일명 가져오기
filenames = np.array(custom_loader.dataset.imgs)[:,0]

list_dataset = list(map(lambda x:os.path.basename(x), filenames))       #확장자 있는 list
list_dataset2 = []                                                     #확장자 지운 list
for i in range (len(list_dataset)) :
    list_dataset2.insert(i,list_dataset[i].split(".")[0])
# 저장된 모델 불러오기
model_ft = torch.jit.load('autoencoder_small.pt')
print("모델 불러오기")
model_ft.eval()
view = []
with open('./view_A20+21+23.pkl', 'rb') as f:
    view = pickle.load(f)
print("view 부르기 성공")
# view_data를 하나의 텐서로 변환
view_data = torch.cat(view, dim=0)
test_x = view_data.to(DEVICE)
encoded_data,decoded_data = model_ft(test_x.view(-1, 299*299*3))
encoded_data = encoded_data.to("cpu")
decoded_data = decoded_data.to("cpu")
print("텐서로 변환")

#코사인유사도 구하기
from sklearn.metrics.pairwise import cosine_similarity
encoded_data_np=encoded_data.detach().numpy()
cosine_sim = cosine_similarity(encoded_data_np, encoded_data_np)
len_bg = 115
len_bl = 1012
len_bn = 17
len_ca = 325
len_cp = 101
len_ct = 206
len_dp = 428
len_jk = 442
len_jp = 736
len_kn = 43
len_kt = 845
len_op = 980
len_pt = 946
len_sb = 48
len_sk = 289
len_tn = 81
len_ts = 1500
len_vt = 390
len_ws = 487
cosine_sim_bg = []
cosine_sim_bl = []
cosine_sim_bn = []
cosine_sim_ca = []
cosine_sim_cp = []
cosine_sim_ct = []
cosine_sim_dp = []
cosine_sim_jk = []
cosine_sim_jp = []
cosine_sim_kn = []
cosine_sim_kt = []
cosine_sim_op = []
cosine_sim_pt = []
cosine_sim_sb = []
cosine_sim_sk = []
cosine_sim_tn = []
cosine_sim_ts = []
cosine_sim_vt = []
cosine_sim_ws = []

for i in range(len_bg):
    cosine_sim_bg.insert(i,cosine_sim[i][0:len_bg])

for i in range(len_bl):
    cosine_sim_bl.insert(i,cosine_sim[i+115][115:115+len_bl])

for i in range(len_bn):
    cosine_sim_bn.insert(i,cosine_sim[i+1127][1127:1127+len_bn])

for i in range(len_ca):
    cosine_sim_ca.insert(i,cosine_sim[i+1144][1144:1144+len_ca])

for i in range(len_cp):
    cosine_sim_cp.insert(i,cosine_sim[i+1469][1469:1469+len_cp])

for i in range(len_ct):
    cosine_sim_ct.insert(i,cosine_sim[i+1570][1570:1570+len_ct])

for i in range(len_dp):
    cosine_sim_dp.insert(i,cosine_sim[i+1776][1776:1776+len_dp])

for i in range(len_jk):
    cosine_sim_jk.insert(i,cosine_sim[i+2204][2204:2204+len_jk])

for i in range(len_jp):
    cosine_sim_jp.insert(i,cosine_sim[i+2646][2646:2646+len_jp])

for i in range(len_kn):
    cosine_sim_kn.insert(i,cosine_sim[i+3382][3382:3382+len_kn])

for i in range(len_kt):
    cosine_sim_kt.insert(i,cosine_sim[i+3425][3425:3425+len_kt])

for i in range(len_op):
    cosine_sim_op.insert(i,cosine_sim[i+4270][4270:4270+len_op])

for i in range(len_pt):
    cosine_sim_pt.insert(i,cosine_sim[i+5250][5250:5250+len_pt])

for i in range(len_sb):
    cosine_sim_sb.insert(i,cosine_sim[i+6196][6196:6196+len_sb])

for i in range(len_sk):
    cosine_sim_sk.insert(i,cosine_sim[i+6244][6244:6244+len_sk])

for i in range(len_tn):
    cosine_sim_tn.insert(i,cosine_sim[i+6533][6533:6533+len_tn])

for i in range(len_ts):
    cosine_sim_ts.insert(i,cosine_sim[i+6614][6614:6614+len_ts])

for i in range(len_vt):
    cosine_sim_vt.insert(i,cosine_sim[i+8114][8114:8114+len_vt])

for i in range(len_ws):
    cosine_sim_ws.insert(i,cosine_sim[i+8504][8504:8504+len_ws])

print(cosine_sim_bg)
print(cosine_sim_bl)
print(cosine_sim_bn)



def get_recommendations_bg(idx, cosine_sim_bg = cosine_sim_bg):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_bg[idx]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]

    return img_indices

def get_recommendations_bl(idx, cosine_sim_bl = cosine_sim_bl):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_bl[idx-115]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 115

    return img_indices

def get_recommendations_bn(idx, cosine_sim_bn = cosine_sim_bn):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_bn[idx-1127]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 1127

    return img_indices

def get_recommendations_ca(idx, cosine_sim_ca = cosine_sim_ca):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_ca[idx-1144]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 1144

    return img_indices

def get_recommendations_cp(idx, cosine_sim_cp = cosine_sim_cp):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_cp[idx-1469]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 1469

    return img_indices

def get_recommendations_ct(idx, cosine_sim_ct = cosine_sim_ct):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_ct[idx-1570]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 1570

    return img_indices

def get_recommendations_dp(idx, cosine_sim_dp = cosine_sim_dp):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_dp[idx-1776]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 1776

    return img_indices

def get_recommendations_jk(idx, cosine_sim_jk = cosine_sim_jk):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_jk[idx-2204]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 2204

    return img_indices

def get_recommendations_jp(idx, cosine_sim_jp = cosine_sim_jp):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_jp[idx-2646]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 2646

    return img_indices

def get_recommendations_kn(idx, cosine_sim_kn = cosine_sim_kn):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_kn[idx-3382]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 3382

    return img_indices

def get_recommendations_kt(idx, cosine_sim_kt = cosine_sim_kt):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_kt[idx-3425]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 3425

    return img_indices

def get_recommendations_op(idx, cosine_sim_op = cosine_sim_op):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_op[idx-4270]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 4270

    return img_indices

def get_recommendations_pt(idx, cosine_sim_pt = cosine_sim_pt):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_pt[idx-5250]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 5250

    return img_indices

def get_recommendations_sb(idx, cosine_sim_sb = cosine_sim_sb):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_sb[idx-6196]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 6196

    return img_indices

def get_recommendations_sk(idx, cosine_sim_sk = cosine_sim_sk):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_sk[idx-6244]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 6244

    return img_indices

def get_recommendations_tn(idx, cosine_sim_tn = cosine_sim_tn):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_tn[idx-6533]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 6533

    return img_indices

def get_recommendations_ts(idx, cosine_sim_ts = cosine_sim_ts):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_ts[idx-6614]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 6614

    return img_indices

def get_recommendations_vt(idx, cosine_sim_vt = cosine_sim_vt):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_vt[idx-8114]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 8114

    return img_indices

def get_recommendations_ws(idx, cosine_sim_ws = cosine_sim_ws):
    # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
    sim_scores = list(enumerate(cosine_sim_ws[idx-8504]))
    # 유사도에 따라 이미지들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 5개의 이미지를 받아옵니다.
    sim_scores = sim_scores[1:6]

    # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
    img_indices = [i[0] for i in sim_scores]
    for i in range(len(img_indices)):
        img_indices[i] += 8504

    return img_indices


# cosine_sim_bg = cosine_sim[0:len_bg]
# cosine_sim_bl = cosine_sim[115:len_bl]
# cosine_sim_bn = cosine_sim[1127:len_bn]
# cosine_sim_ca = cosine_sim[1144:len_ca]
# cosine_sim_cp = cosine_sim[1469:len_cp]
# cosine_sim_ct = cosine_sim[1570:len_ct]
# cosine_sim_dp = cosine_sim[1776:len_dp]
# cosine_sim_jk = cosine_sim[2204:len_jk]
# cosine_sim_jp = cosine_sim[2646:len_jp]
# cosine_sim_kn = cosine_sim[3382:len_kn]
# cosine_sim_kt = cosine_sim[3425:len_kt]
# cosine_sim_op = cosine_sim[4270:len_op]
# cosine_sim_pt = cosine_sim[5250:len_pt]
# cosine_sim_sb = cosine_sim[6196:len_sb]
# cosine_sim_sk = cosine_sim[6244:len_sk]
# cosine_sim_tn = cosine_sim[6533:len_tn]
# cosine_sim_ts = cosine_sim[6614:len_ts]
# cosine_sim_vt = cosine_sim[8114:len_vt]
# cosine_sim_ws = cosine_sim[8504:len_ws]

# print(len(cosine_sim_pt))
# cosine_sim_pt2 = []
# for i in range(len(cosine_sim_pt)) :
#     cosine_sim_pt2.insert(i,cosine_sim_pt[i][5250:6196])
# datadir = os.path.join(os.getcwd(), 'content', 'img_test_sorted_A20+21+23')
# dir_list = os.listdir(datadir)

# def checkdir(dir_list, filename) :
#     for i in range(len(dir_list)) :
#         path = os.path.join(os.getcwd(), 'content', 'img_test_sorted_A20+21+23', dir_list[i])
#         if os.path.isfile(os.path.join(path,filename)):
#             return path
# #print(checkdir(dir_list, list_dataset[5464]))

# # def get_recommendations(idx, cosine_sim=cosine_sim):
# #     # 모든 이미지에 대해서 해당 이미지와의 유사도를 구합니다.    
# #     sim_scores = list(enumerate(cosine_sim[idx]))

# #     # 유사도에 따라 이미지들을 정렬합니다.
# #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

# #     # 가장 유사한 5개의 이미지를 받아옵니다.
# #     sim_scores = sim_scores[2:7]

# #     # 가장 유사한 5개의 이미지의 인덱스를 받아옵니다.
# #     img_indices = [i[0] for i in sim_scores]

# #     return img_indices

    
## 텐서 합치는 부분 2분15초 정도 걸려서 아래 코드는 미리 돌리고 view list pickle로 저장! ###
#------------------------------------------------------------------------------------------#
# view = []
# for i, (images, labels) in enumerate(test_dataset):

#     #print(f"Img {i + 1}: {images.size()} - Labels: {labels}")
#     # 이미지를 하나의 텐서로 합치기
#     view.append(images.view(-1, 3, 299, 299))

# print("이미지 텐서로 합치기")
# with open('view_orig.pkl', 'wb') as f:
#     pickle.dump(view, f)
#------------------------------------------------------------------------------------------#
def run():
    torch.multiprocessing.freeze_support()

    # CLASSES = {
    #     0: 'BG',
    #     1: 'BL',
    #     2: 'BN',
    #     3: 'CA',
    #     4: 'CP',
    #     5: 'CT',
    #     6: 'DP',
    #     7: 'JK',
    #     8: 'JP',
    #     9: 'KN',
    #     10: 'KT',
    #     11: 'OP',
    #     12: 'PT',
    #     13: 'SB',
    #     14: 'SK',
    #     15: 'TN',
    #     16: 'TS',
    #     17: 'VT',
    #     18: 'WS'
    # }

    # X = encoded_data.data[:, 0].numpy()
    # Y = encoded_data.data[:, 1].numpy()
    # Z = encoded_data.data[:, 2].numpy() #잠재변수의 각 차원을 numpy행렬로 변환합니다.
    # # print(type(X))
    # labels = test_dataset.targets #레이블도 넘파이행렬로 변환합니다.

    
    # unique_labels = set(labels)  # Get unique labels

    # for label in unique_labels:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    # #     # Filter data for the current label
    # #     label_indices = [i for i, l in enumerate(labels) if l == label]
    # #     label_X = [X[i] for i in label_indices]
    # #     label_Y = [Y[i] for i in label_indices]
    # #     label_Z = [Z[i] for i in label_indices]

    # #     name = CLASSES[label]
    # #     color = cm.rainbow(int(100 * label / 9))
        
    # #     ax.scatter(label_X, label_Y, label_Z, c=color)  # Use scatter() to plot each point with a separate color
    # #     for x, y, z in zip(label_X, label_Y, label_Z):
    # #         ax.text(x, y, z, name, backgroundcolor=color)

    # #     ax.set_xlim(min(X), max(X))
    # #     ax.set_ylim(min(Y), max(Y))
    # #     ax.set_zlim(min(Z), max(Z))
        
    # #     plt.title(f"Label {label} - {name}")
    # #     plt.show()

    # for x, y, z, s in zip(X, Y, Z, labels): #zip()은 같은 길이의 행렬들을 모아 순서대로 묶어줍니다.
    #     name = CLASSES[s]
    #     color = cm.rainbow(int(100*s/9))
    #     ax.text(x, y, z, name, backgroundcolor=color)
    #     ax.set_xlim(X.min(), X.max())
    #     ax.set_ylim(Y.min(), Y.max())
    #     ax.set_zlim(Z.min(), Z.max())
    # #ax.view_init(azim=0, elev=90)

    # plt.show()




    # 플라스크로 추천받을 옷 파일명 받아와서 데이터셋에서 인덱스 찾기
    img_list = ["AIB3SK004",  "AIA3TS022", "AIG3BL022", "AIB3JP002", "AID4VT001","AIB3DP005", "AIC3DP006" ,"AIF3KN001", "AIA7TS003","AIK7CA001"]
    img_idx = []
    for i in range(len(img_list)):
        #print(img_list[i])
        for j in range(len(list_dataset2)):
            if list_dataset2[j]==img_list[i] :
                img_idx.insert(i,j)
        if len(img_idx) == i:
            img_idx.insert(i,0)
    print(img_idx)    


    #추천된 이미지 확인하기
    idx = []    # 추천한 옷 인덱스가 있는 배열
    for i in range(len(img_idx)):
        # Get recommendations
        if img_idx[i] < 115 :
            idx.insert(i,get_recommendations_bg(img_idx[i]))
            continue
        if 115 <= img_idx[i] < 1127 :
            idx.insert(i,get_recommendations_bl(img_idx[i]))
            continue
        if 1127 <= img_idx[i] < 1144 :
            idx.insert(i,get_recommendations_bn(img_idx[i]))
            continue
        if 1144 <= img_idx[i] < 1469 :
            idx.insert(i,get_recommendations_ca(img_idx[i]))
            continue
        if 1469 <= img_idx[i] < 1570 :
            idx.insert(i,get_recommendations_cp(img_idx[i]))
            continue
        if 1570 <= img_idx[i] < 1776 :
            idx.insert(i,get_recommendations_ct(img_idx[i]))
            continue
        if 1776 <= img_idx[i] < 2204 :
            idx.insert(i,get_recommendations_dp(img_idx[i]))
            continue
        if 2204 <= img_idx[i] < 2646 :
            idx.insert(i,get_recommendations_jk(img_idx[i]))
            continue
        if 2646 <= img_idx[i] < 3382 :
            idx.insert(i,get_recommendations_jp(img_idx[i]))
            continue
        if 3382 <= img_idx[i] < 3425 :
            idx.insert(i,get_recommendations_kn(img_idx[i]))
            continue
        if 3425 <= img_idx[i] < 4270 :
            idx.insert(i,get_recommendations_kt(img_idx[i]))
            continue
        if 4270 <= img_idx[i] < 5250 :
            idx.insert(i,get_recommendations_op(img_idx[i]))
            continue
        if 5250 <= img_idx[i] < 6196 :
            idx.insert(i,get_recommendations_pt(img_idx[i]))
            continue
        if 6196 <= img_idx[i] < 6244 :
            idx.insert(i,get_recommendations_sb(img_idx[i]))
            continue
        if 6244 <= img_idx[i] < 6533 :
            idx.insert(i,get_recommendations_sk(img_idx[i]))
            continue
        if 6533 <= img_idx[i] < 6614 :
            idx.insert(i,get_recommendations_tn(img_idx[i]))
            continue
        if 6614 <= img_idx[i] < 8114 :
            idx.insert(i,get_recommendations_ts(img_idx[i]))
            continue
        if 8114 <= img_idx[i] < 8504 :
            idx.insert(i,get_recommendations_vt(img_idx[i]))
            continue
        if 8504 <= img_idx[i] < 8991 :
            idx.insert(i,get_recommendations_ws(img_idx[i]))
        
    print(idx)
    # Initialize recommend_list
    recommend_list = []

    #추천된 옷 파일명 출력
    for i in range(len(idx)) :
        # Initialize inner list for each iteration
        inner_list = []
        for j in range(len(idx[0])) :
            print(list_dataset2[idx[i][j]])
            # Use append to add elements to the inner list
            inner_list.append(list_dataset2[idx[i][j]])
        # Append the inner list to recommend_list
        recommend_list.append(inner_list)
    print(recommend_list)
    # 이미지 확인
    #랜덤으로 이미지를 선택합니다.
    # rand = torch.randint(len(test_dataset), size=(1,)).item()
    rand = 8190
    images, labels = test_dataset[rand]
    print("image:", images)
    print(rand)
    print(list_dataset[rand])
    print(f"Img: {images.size()} - Labels: {labels}")
    view_img = images.view(-1, 3, 299, 299)
    #view_img = view_img.type(torch.FloatTensor)/255.
    out_img = torch.squeeze(view_img.cpu())
    img=out_img.permute(1,2,0)
    fig = plt.figure(figsize=(1, 1))
    #print(type(img))
    plt.imshow(img) #랜덤으로 선택된 이미지를 확인합니다.
    plt.title("Random Image")

    # Get recommendations
    idx = get_recommendations_vt(rand)
    print(idx)
    images = []
    view_img = []
    out_img = []

    for i in range(5) :
        images.insert(i,test_dataset[idx[i]])
        img, label = images[i]
        view_img.insert(i,img.view(-1, 3, 299, 299))
        out_img.insert(i,torch.squeeze(view_img[i].cpu()))
        images.insert(i,out_img[i].permute(1,2,0))

    
    for i in range(5) :
        print(list_dataset[idx[i]])
      
    f, a = plt.subplots(2,5, figsize=(5,2))
    for j in range(5):
        recommended_img = decoded_data.data[idx[j]]
        recommended_img=recommended_img.reshape(-1, 3, 299, 299)
        recommended_img=recommended_img.permute(0,2,3,1)[0]
        a[0][j].imshow(recommended_img)
        a[0][j].set_xticks(()); a[0][j].set_yticks(())
        a[1][j].imshow(images[j])
        a[1][j].set_xticks(()); a[1][j].set_yticks(())

    plt.suptitle('Recommendations',fontsize=20)
    plt.show()


if __name__ == '__main__':
    run()