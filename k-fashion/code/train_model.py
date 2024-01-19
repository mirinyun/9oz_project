from __future__ import print_function, division

import torch
import torch.utils as utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def run():
    torch.multiprocessing.freeze_support()
    
    data_dir = './Train'
    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_dataset =  datasets.ImageFolder(data_dir, data_transform)

    train_split = 0.9
    split_size = int(len(image_dataset) * train_split)
    batch_size = 64
    num_workers= 8


    train_set, valid_set = torch.utils.data.random_split(image_dataset, [split_size, len(image_dataset) - split_size])
    #추가한 부분
    #train_set = DistributedSampler(dataset=train_set, shuffle=True)

    tr_loader = utils.data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers)
    val_loader = utils.data.DataLoader(dataset=valid_set,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)
    dataloaders = {'train': tr_loader, 'val':val_loader}
    dataset_sizes = {}
    dataset_sizes['train'] = split_size
    dataset_sizes['val'] = len(image_dataset) -split_size
    class_names = image_dataset.classes


    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' # 네 개 device를 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # print(f"Using {device} device")
    # print(torch.version)
    def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(image_dataset.classes))
    # model_ft = torch.nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    # model_ft = nn.DataParallel(model_ft, output_device=1) # gradient를 1번 디바이스에 gather함

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)

    model_scripted = torch.jit.script(model_ft) # TorchScript 형식으로 내보내기
    model_scripted.save('my9oz_model_resnet34.pt') # 저장하기
if __name__ == '__main__':
    run()