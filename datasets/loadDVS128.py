class_num = 11
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torchneuromorphic.dvs_gestures import create_hdf5,dvsgestures_dataloaders
import torch
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms

#
def get_DVS128_beida(data_path, network_config):
    batch_size = network_config['batch_size']

    train_set = DVS128Gesture(data_path, train=True, data_type='frame', split_by='number', frames_number=network_config['n_steps'])
    test_set = DVS128Gesture(data_path, train=False, data_type='frame', split_by='number', frames_number=network_config['n_steps'])

    # train_set = DVS128Gesture(data_path, train=True, data_type='event')
    # test_set = DVS128Gesture(data_path, train=False, data_type='event')

    trainloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    return trainloader, testloader

import os

dir = 'F:\pycharmproject\spiking_unet11\spiking_unet\one_more_try\datasets\DvsGesture\DvsGesture\DvsGesture/'
name = 'dvs128_hdf5'

def get_DVS128_torch(root,name):

    cr=create_hdf5.create_events_hdf5(root,name)

    train_dl, test_dl = create_dataloader(
        root='./'+name,
        batch_size=16,
        ds=4,
        num_workers=0)

    return train_dl,test_dl

