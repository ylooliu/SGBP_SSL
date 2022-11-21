import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_imagenet(data_path, network_config):
    print("loading ImageNet")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = network_config['batch_size']
    data_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_dataset =torchvision.datasets.ImageFolder(root='./ILSVRC2012/train',transform=data_transform)
    trainloader =DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=4)

    test_dataset = torchvision.datasets.ImageFolder(root='ILSVRC2012/val',transform=data_transform)
    testloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True,num_workers=4)

    return trainloader,testloader