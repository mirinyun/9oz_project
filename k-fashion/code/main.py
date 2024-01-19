from __future__ import print_function, division

import torch
import torch.utils as utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import train_model

class TestDataset(utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.image_list = os.listdir(root)
            self.transform = transform

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, index):
            image_path = os.path.join(self.root, self.image_list[index])
            image = np.array(Image.open(image_path).convert('RGB'))
            image = self.transform(image)
            return self.image_list[index], image

def run():
    torch.multiprocessing.freeze_support()
   
    #저장된 모델 불러오기
    model_ft = torch.jit.load('./k-fashion/model/my9oz_model.pt')
    model_ft.eval()

    
    # Test Dataset Directory
    test_dir = './test_data/u2net_outputs/A17'
          
    test_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = TestDataset(test_dir,transform=test_transform)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result = []
    for fnames, data in tqdm(test_dataloader):
        data = data.to(device)
        output = model_ft(data)
        _,pred = torch.max(output,1)
        for j in range(len(fnames)):
            result.append(
                {
                    'filename':fnames[j].split(".")[0],
                    'style':pred.cpu().detach().numpy()[j]
                }
        )
    pd.DataFrame(sorted(result,key=lambda x:x['filename'])).to_csv('9oz_A17.csv',index=None)
if __name__ == '__main__':
    run()