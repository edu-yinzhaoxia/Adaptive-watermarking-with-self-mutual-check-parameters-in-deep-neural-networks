
import random

import torch
from model import Model
import torch.nn as nn

import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import time


from torchsummary import summary
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(device)
# model = Model()
#
# print("%.7f"%model.conv1.weight[0][0][0][0])

# root1 =  './models/mnistcuangai0407'
# root2 =  './models/mnistcuangai0407-jianhuaadaptive.pkl'
# root3 =  './models/mnistcuangai0407-mod3'
# root4 =  './models/mnistcuangai0407-mod3'
# root = ['./models/random_attack.pkl',  #攻击后的
#         './models/random_attack-recovery.pkl',#恢复后的
#         './models/mnistcuangai0407-mod3',]#原始的
#         #'./models/mnistcuangai0407-mod3']

#root = ['./models/alexnet/cifar10/cifar10_0403_0.97.pkl','./models/alexnet/cifar10/alexnet_1_cifar10_0403_0.97-adapt12bit0507.pkl','./models/alexnet/cifar10/alexnet_1_cifar10_0403_0.97.pkl','./models/alexnet/cifar10/alexnet_1_cifar10_adapt','./models/alexnet/cifar10/alexnet_1_cifar10_adapt-mod511-0.05attack']#原始的
        #'./models/mnistcuangai0407-mod3']
#root = ['models/resnet18/cifar10/cifar10_0403_0.7158.pkl','models/resnet18/cifar10/cifar10_0403_0.7158-exchange.pkl','models/resnet18/cifar10/cifar10_0403_0.7158-exchange-adapt.pkl']#原始的
#'./models/resnet18/cifar10/cifar10_0403_0.9839-2-exchange.pkl','./models/resnet18/cifar10/cifar10_0403_0.9839-2-exchange-adapt.pkl','
#'./models/alexnet/cifar10/alexnet_1_cifar10_adapt-mod511-0.05attack'

#root = ['./models/resnet18/cifar10/resnet_1_cifar10_adapt-mod511-0.05attack','./models/resnet18/cifar10/resnet18_1_cifar10_adapt-mod511-0.001recovery']
#root = ['./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu-attack-0.01.pkl','./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu-attack-0.01-recover.pkl'
#        ,'./models/resnet50/cifar10/cifar10_0403_0.7432-exchange.pkl','./models/resnet50/cifar10/cifar10_0403_0.7432.pkl',]

root = ['./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-mod511.pkl']
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     transforms.Resize((32, 32)),
     ])

# transform = transforms.Compose(
#     [transforms.RandomCrop(32, padding=4),
#      transforms.RandomHorizontalFlip(),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#      transforms.Resize((32, 32)),
#      ])

# model = torch.load(root1).to(device)


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


batch_size = 10000
train_dataset = cifar.CIFAR10(root='./cifar/cifar10', train=True, transform=transform)
test_dataset = cifar.CIFAR10(root='./cifar/cifar10', train=False, transform=transform)


# train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
# test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_correct_num = 0
all_sample_num = 0



for i in root:
    print(i)
    model = torch.load(i).to(device)

    # if torch.cuda.device_count() > 1:
    #     print("Using multiple GPUs for training.")
    #     model = nn.DataParallel(model)


    #print(summary(model, (3, 256, 256)))
    start_time = time.time()
    for current_epoch in range(5):
        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_label = test_label.to(device)

            predict_y = model(test_x.to(device).float()).detach()
            predict_y = torch.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            #        all_correct_num += torch.sum(current_correct_num.numpy(), axis=-1)
            all_correct_num += torch.sum(current_correct_num, axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.10f}'.format(acc))
    end_time = time.time()
    runtime = end_time - start_time
    print("运行时间：", runtime, "秒")

#
# for current_epoch in range(1):
#     all_correct_num = 0
#     all_sample_num = 0
#     for idx, (test_x, test_label) in enumerate(test_loader):
#         test_label = test_label.to(device)
#
#         predict_y = model(test_x.to(device).float()).detach()
#         predict_y = torch.argmax(predict_y, axis=-1)
#         current_correct_num = predict_y == test_label
# #        all_correct_num += torch.sum(current_correct_num.numpy(), axis=-1)
#         all_correct_num += torch.sum(current_correct_num, axis=-1)
#
#         all_sample_num += current_correct_num.shape[0]
#     acc = all_correct_num / all_sample_num
#     print('accuracy: {:.10f}'.format(acc))
#
#
# model = torch.load(root2).to(device)
# print("%.7f"%model.conv1.weight[0][0][0][0])
#
# for current_epoch in range(1):
#     all_correct_num = 0
#     all_sample_num = 0
#     for idx, (test_x, test_label) in enumerate(test_loader):
#         test_label = test_label.to(device)
#
#         predict_y = model(test_x.to(device).float()).detach()
#         predict_y = torch.argmax(predict_y, axis=-1)
#         current_correct_num = predict_y == test_label
# #        all_correct_num += torch.sum(current_correct_num.numpy(), axis=-1)
#         all_correct_num += torch.sum(current_correct_num, axis=-1)
#
#         all_sample_num += current_correct_num.shape[0]
#     acc = all_correct_num / all_sample_num
#     print('accuracy: {:.10f}'.format(acc))
#
# #print(model.conv1.weight)
# model = torch.load(root3).to(device)
# print("%.7f"%model.conv1.weight[0][0][0][0])
#
# for current_epoch in range(1):
#     all_correct_num = 0
#     all_sample_num = 0
#     for idx, (test_x, test_label) in enumerate(test_loader):
#         test_label = test_label.to(device)
#
#         predict_y = model(test_x.to(device).float()).detach()
#         predict_y = torch.argmax(predict_y, axis=-1)
#         current_correct_num = predict_y == test_label
# #        all_correct_num += torch.sum(current_correct_num.numpy(), axis=-1)
#         all_correct_num += torch.sum(current_correct_num, axis=-1)
#
#         all_sample_num += current_correct_num.shape[0]
#     acc = all_correct_num / all_sample_num
#     print('accuracy: {:.10f}'.format(acc))
#
