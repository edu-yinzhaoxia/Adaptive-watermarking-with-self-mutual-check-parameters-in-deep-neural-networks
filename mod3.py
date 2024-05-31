import struct
import numpy as np
import torch
from model import Model
import math
import random
from utils import *
import pickle

# a = [2,3,4]
# b,c,d = (shuffle_and_restore(a,0))
# e = []
# for i in d:
#    e.append(b[i])
# print(e)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
#model = torch.load('./models/mnistcuangai0407-jianhuaadaptive.pkl').to(device)
#model = torch.load('./models/alexnet/cifar10/alexnet_1_cifar10_adapt')


model = torch.load('./models/resnet50/cifar10/cifar10_0403_0.7432.pkl')

# Histogram_0 = []
# Histogram_1 = []
for _,parmeters in model.named_parameters():    #parmeters也就是一层的weight或者bias的参数
    print(parmeters)
    c = []
    huffmanCode = []
    a,b = flattenM(parmeters)  #a属于中间变量，也就是扁平化后的parameter b是size

    aa = a.clone()

    shuffle,ori_list,order_list,lenth = shuffle_and_restore(aa,14586) #乱序，乱序后下来要进行bit的摩二运算,order是记录下的顺序



    for index,i in enumerate(shuffle):
        c.append(floatToBinary32(i.item())) #此时序列c成为2进值浮点数



    for index,i in enumerate(c):
        c[index] = list(str(i))          #此时c中的每个元素成为字符串元素

    #c = c[:2]


    "处理列表中的中间的元素，也就是除去尾部的元素"
    for index,i in enumerate(c):
        if index != lenth-1:   #i = li1
            #for index,i in enumerate(c):           #c是整个数列，i是
            # 一个数列中[0,1,0,0,1,...,1,0]
            li1 = c[index]

            #check_bit = XOR_condense(li1,li3)#横向
            last_colcheck(li1,511)  # 最后的纵向检测
            #print(len(li1))
            #check_bit_2 = colEncoder_2(check_bit,li3,'0000011111') #纵向 之前纵向写的是i不是li3
            # assign_last12_2(check_bit_2, c, index + 1)  # 水印位以及加上去了
            #
            # comparebigxiao(li1,shuffle,index,lenth)

            # check_bit = XOR_condense_2(li1, li3)  # 横向
            # check_bit_2 = colEncoder_2(check_bit,li3) #纵向 之前纵向写的是i不是li3
            # assign_last12_2(check_bit_2, c, index + 1)  # 水印位以及加上去了



            # deci = transDecimal(check_bit_2)
            # huffmanCode.append(deci)


        if index ==  len(c)-1:
            li1 = c[index]

            # check_bit = XOR_condense(li1, li3)  # 横向
            # check_bit_2 = colEncoder(check_bit, li3)  # 纵向
            # assign_last12(check_bit_2, c,0)  # 水印位以及加上去了



            last_colcheck(li1, 511)  # 横向
            # check_bit_2 = colEncoder_2(check_bit,li3,'00000111111') #纵向 之前纵向写的是i不是li3
            # assign_last12_2(check_bit_2, c, 0)  # 水印位以及加上去了
            #
            # comparebigxiao(li1,shuffle,index,lenth)




            # deci = transDecimal(check_bit_2)
            # huffmanCode.append(deci)




    # with open("my_list.pkl", "wb") as f:
    #     pickle.dump(c, f)
    # 
    # # 从文件中加载列表
    # with open("my_list.pkl", "rb") as f:
    #     loaded_list = pickle.load(f)

   # print(loaded_list)

    "处理列表纵向元素"



    c = suminsideStr(c)
    #print(c)
    for index, i in enumerate(c):
        c[index] = Binary32Tofloat(i)


    # Histogram_1 = c + Histogram_1

    "还原顺序"
    # restored_list = [c[i] for i in order_list]
    # ad = list(range(lenth))
    # for index,i in enumerate(a):
    #     ad[i] = oss[index]

    #print(e)

    # print(restored_list)
    # print(ori_list)

    restored_list = []
    restored_list = list(range(lenth))
    for index,i in enumerate(c):
        restored_list[order_list[index]] = i
    #print(restored_list)
    # for index,i in enumerate(restored_list):
    #     restored_list[index] = i.item()
    #
    #
    #
    restored_list = np.array(restored_list)
    restored_list = restored_list.reshape(tuple(b))
    restored_list = torch.from_numpy(restored_list)

    # c = np.array(c)
    # c = c.reshape(tuple(b))
    # c = torch.from_numpy(c)


    parmeters.data = restored_list
    print(parmeters)


# after = Histogram_1
# with open('after.txt', 'w') as f2:
#     for item in after:
#         f2.write("%s\n" % item)

model.float()

torch.save(model, 'models/resnet50/cifar10/cifar10_0403_0.7432-selfembedding.pkl')
