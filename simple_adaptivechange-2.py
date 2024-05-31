import struct
import numpy as np
import torch
from model import Model
import math
import random
from utils import *
import pickle
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
import time
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import gc
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Model().to(device)
# model = torch.load('./models/mnistcuangai0407').to(device)
# model = torch.load('./models/alexnet/cifar10/alexnet_1_cifar10_0403_0.97.pkl').to(device)

# model = torch.load('./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-7.pkl').to(device)
# model1 = torch.load('./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-7.pkl').to(device)
model = torch.load('./models/alexnet/cifar10/alexnet_1_cifar10_exchange-adapt-mod511').to(device)

if torch.cuda.device_count() > 1:
    print(device)
    print("Using multiple GPUs for training.")
    model = nn.DataParallel(model)

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

loss_fn = CrossEntropyLoss().to(device)

batch_size_train = 2000

batch_size_test = 2000

# train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
# test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
# train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     transforms.Resize((256, 256)),
     ])

train_dataset = cifar.CIFAR10(root='./cifar/cifar10', train=True, transform=transform)
test_dataset = cifar.CIFAR10(root='./cifar/cifar10', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test)

# circletimes = 3
# proportion = 16
circletimes = 3
proportion = 64

a = 0
for _, params in model.named_parameters():

    counter = 1
    if a > 1:
        break

    a += 1
    if a < 0:
        continue



    for i in range(circletimes):

        if counter > circletimes:  # 每层循环几次
            continue

        circletimes_compareexchange = 0
        for idx, (train_x, train_label) in enumerate(train_loader):
            model.eval()
            model.float().to(device)
            train_label = train_label.to(device)
            predict_y = model(train_x.to(device).float())
            loss = loss_fn(predict_y, train_label)
            loss.backward()
            break

        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(test_loader):

            test_data_list = []
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_data_list.append((test_x.to(device).float(), test_label.to(device)))

            model.eval()
            model.float().to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.to(device).float())
            current_correct_num = predict_y.argmax(dim=-1) == test_label
            all_correct_num += torch.sum(current_correct_num).item()
            all_sample_num += test_label.size(0)
            acc1 = all_correct_num / all_sample_num
            print('accuracy: {:.10f}'.format(acc1))
            print('第{}层,{}轮训练（微调前）'.format(_, counter))

            del predict_y
            gc.collect()
            break

        if counter == 1:
            best_acc = acc1

        # start_time = time.time()

        # print(params)
        # paramF = params.flatten()
        # print(params.shape)
        paramF, b = flattenM(params)
        # print(params.grad)
        paramG = params.grad.flatten()
        compressPG = list(zip(paramF, paramG))

        compressPG = [list(elem) for elem in compressPG]

        # transform_order = []
        # print(enumerate(compressPG))
        # for i in enumerate(compressPG):
        #     print(i)
        sorted_list = sorted(enumerate(compressPG), key=lambda x: -abs(x[1][1]))
        transform_order = [t[0] for t in sorted_list]  # 仅保留原列表中的索引
        sorted_list = [t[1] for t in sorted_list]  # 仅保留元素值

        # t = []
        # for innerorder in transform_order:
        #     t.append(sorted_list[innerorder][0])

        # for idx, item in enumerate(compressPG):      #记录从小到大的顺序
        #     transform_order.append(sorted_list.index(item))

        for index, (i, j) in enumerate(sorted_list):
            last12 = last12bit_correspondsimple(i.item())  # 取出来中间的一个bit
            sorted_list[index].append(last12)
            sorted_list[index].append(index)

        deal_list = sorted_list
        # deal_list = sorted_list
        lenth = len(deal_list)
        for index, (i, j, k, location) in enumerate(deal_list):

            if index + 1 > lenth / proportion:
                # 4
                break

            tensor = i
            gradoftensor = j
            tail = k

            effmax = 0
            corr_index = index
            if tensor * gradoftensor > 0:
                deal_list[index][2] = 0
                # k = 0


            else:
                # k = 1
                deal_list[index][2] = 1

                # deal_list[index][2],deal_list[corr_index][2] = deal_list[corr_index][2],deal_list[index][2] #交换并记录顺序
                # deal_list[index][3],deal_list[corr_index][3] = deal_list[corr_index][3],deal_list[index][3] #交换并记录顺序

        new_decimal = []
        for index, (i, j, k, location) in enumerate(deal_list):
            # binary_tail = bin(k)
            # binary_tail = binary_tail[2:]
            binary_all = floatToBinary32(i)

            newbinary = binary_all[:11] + str(k) + binary_all[12:]
            # newbinary = binary_all

            dem = Binary32Tofloat(newbinary)
            new_decimal.append(dem)

        ori_order = []
        ori_order = list(range(len(new_decimal)))

        for num, innerorder in enumerate(new_decimal):
            ori_order[transform_order[num]] = innerorder

        ori_order = np.array(ori_order)
        ori_order = ori_order.reshape(tuple(b))

        ori_order = torch.from_numpy(ori_order)

        params.data = ori_order

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print("Elapsed time: {:.4f} seconds".format(elapsed_time))

        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(test_data_list):
            model.eval()
            model.float().to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.to(device).float())

            current_correct_num = predict_y.argmax(dim=-1) == test_label
            all_correct_num += torch.sum(current_correct_num).item()
            all_sample_num += test_label.size(0)
            acc2 = all_correct_num / all_sample_num
            print('accuracy: {:.10f}'.format(acc2))
            print('第{}层,-上面一轮-轮训练（微调后-测试）'.format(_, ))

            del predict_y
            gc.collect()
            break


        # if acc2 > acc1:
        if acc2 > best_acc:

            model1 = model
            model = model1
            best_acc = acc2
        else:

            circletimes_compareexchange += 1

        if circletimes_compareexchange == circletimes:
            model = model1

        # if acc2 > 0.74:
        #     torch.save(model, './models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu.pkl')
    #
    #         break
    #     counter += 1
    #
    # else:
    #     continue
    # break  # 外层循环中触发 break



model.float()

# torch.save(model, './models/mnistcuangai0407-jianhuaadaptive.pkl')
# torch.save(model, './models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-8.pkl')
#torch.save(model, './models/alexnet/cifar10/alexnet_1_cifar10_0403_0.97-adapt12bit0507.pkl')