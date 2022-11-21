import os
import random
import sys
import torch
import torch.backends.cudnn as cudnn
from utils.network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadFashionMNIST, loadNMNIST_Spiking
import logging
from utils.utils import learningStats
import utils.cnns as cnns
import utils.global_v as glv
from utils.utils import EarlyStopping
import BP.loss_f as loss_f
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import numpy as np

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import argparse

import ssl


ssl._create_default_https_context = ssl._create_unverified_context


max_accuracy = 0
min_loss = 1000


def train(network, trainloader, opti, epoch, states, network_config, err):
    network.train()
    sg_sigma = 2
    global max_accuracy
    global min_loss
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()

    if network_config['loss'] == "kernel":
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps / 10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).cuda()
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        targets = torch.zeros(labels.shape[0], n_class, 1, 1, n_steps).cuda()
        if len(inputs.shape)<5:
            inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)

        # forward pass
        labels = labels.cuda()
        inputs = inputs.cuda()
        inputs.type(torch.float32)
        outputs = network.forward(inputs, True,sg_sigma)  #true表示训练状态 返回的outputs是最后一层产生的psp

        targets.zero_()
        for i in range(len(labels)):
            targets[i, labels[i], ...] = desired_spikes

        loss = err.spike_kernel(outputs, targets, network_config)

        # backward pass
        opti.zero_grad()
        loss.backward()
        clip_grad_norm_(network.get_parameters(), 1)
        opti.step()
        network.weight_clipper()

        spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        train_loss += torch.sum(loss).item()
        labels = labels.cpu().numpy()
        total += len(labels)
        correct += (predicted == labels).sum().item()

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.cpu().data.item()
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > max_accuracy:
        max_accuracy = total_accuracy
    if min_loss > total_loss:
        min_loss = total_loss

    logging.info("Train Accuracy: %.4f (%.3f). Loss: %.4f (%.4f)", 100. * total_accuracy, 100 * max_accuracy, total_loss, min_loss)


def test(network, testloader, epoch, states, network_config, early_stopping):
    network.eval()
    global best_acc
    global best_epoch
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    for batch_idx, (inputs, labels) in enumerate(testloader):

        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)

        # forward pass
        labels = labels.cuda()
        inputs = inputs.cuda()
        outputs = network.forward(inputs,False)

        spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        labels = labels.cpu().numpy()
        y_pred.append(predicted)

        y_true.append(labels)
        total += len(labels)
        correct += (predicted == labels).sum().item()

        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    test_accuracy = correct / total
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cf = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
        df_cm = pd.DataFrame(cf, index = [str(ind*25) for ind in range(n_class)], columns=[str(ind*25) for ind in range(n_class)])
        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.savefig("confusion_matrix.png")
        plt.close()

    logging.info("Train Accuracy: %.4f (%.4f).", 100. * test_accuracy, 100 * best_acc)
    # Save checkpoint.
    acc = 100. * correct / total
    early_stopping(acc, network, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',default='./configs/MNIST_CNN.yaml',action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-gpu', type=int, default=0, help='GPU device to use (default: 0)')
    parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    logging.basicConfig(filename='result/result.log', level=logging.INFO)
    logging.info("start parsing settings")
    params = parse(config_path)
    logging.info("finish parsing settings")
    
    # check GPU
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    # set GPU
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True

    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.manual_seed(args.seed)  #
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    glv.init(params['Network']['n_steps'], params['Network']['tau_s'] )
    
    logging.info("dataset loaded")
    if params['Network']['dataset'] == "MNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadMNIST.get_mnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "NMNIST_Spiking":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "FashionMNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadFashionMNIST.get_fashionmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "CIFAR10":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadCIFAR10.get_cifar10(data_path, params['Network'])

    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    
    net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).cuda()
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
    
    error = loss_f.SpikeLoss(params['Network']).cuda()
    
    optimizer = torch.optim.AdamW(net.get_parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))
    
    best_acc = 0
    best_epoch = 0
    
    l_states = learningStats()
    early_stopping = EarlyStopping()


    for e in range(params['Network']['epochs']):
        l_states.training.reset()

        train(net, train_loader, optimizer, e, l_states, params['Network'], error)
        l_states.training.update()
        l_states.testing.reset()
        print('one epoch train finish!')
        test(net, test_loader, e, l_states, params['Network'], early_stopping)
        l_states.testing.update()


    logging.info("Best Accuracy: %.4f, at epoch: %d \n", best_acc, best_epoch)

