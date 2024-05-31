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
#model = torch.load('./models/random_attack.pkl').to(device)


#model = torch.load('./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu-attack-0.01.pkl').to(device)
model = torch.load('./models/alexnet/cifar10/alexnet_1_cifar10_0403_0.97-adapt12bit0507.pkl').to(device)

for _,parmeters in model.named_parameters():    #parmeters也就是一层的weight或者bias的参数
    print(parmeters)


    c = []
    a,b = flattenM(parmeters)  #a属于中间变量，也就是扁平化后的parameter b是size

    aa = a.clone()

    shuffle,ori_list,order_list,lenth = shuffle_and_restore(aa,14586) #乱序，乱序后下来要进行bit的摩二运算,order是记录下的顺序



    for index,i in enumerate(shuffle):
        c.append(floatToBinary32(i.item())) #此时序列c成为2进值浮点数



    for index,i in enumerate(c):
        c[index] = list(str(i))          #此时c中的每个元素成为字符串元素

    check_state = []
    "处理列表中的中间的元素，也就是除去尾部的元素"
    for index,i in enumerate(c):
        # if index != lenth-1:   #
        #     li1 = c[index]
        #     li2 = c[index+1]
        #     jiaoyan = precise_recovery_1(li2,'00000111111')
        #     jiaoyan_li1 = c[index][0:11]

        if index != 0:   #
            #for index,i in enumerate(c):           #c是整个数列，i是一个数列中[0,1,0,0,1,...,1,0]
                li1 = c[index]



                # check_bit = XOR_condense_2(li1,li3) #纵向相加
                # check_bit_2 = colEncoder_2(check_bit,li3,'00000111111') #纵向 之前纵向写的是i不是li3
                #
                cal_check = li1[0:23]
                cal_check = int(''.join(cal_check),2)
                original_check = li1[23:]
                original_check = int(''.join(original_check),2)
                _,reminder = divmod(cal_check,511)
                if reminder == original_check:
                    check_state.append(0)
                else:
                    check_state.append(1)
                #check_bit = check_XOR(li1, li3)  # 横向

                #assign_last12(check_bit_2,c,index) #水印位以及加上去了



        if index ==  0:
            li1 = c[index]

            # check_bit = XOR_condense_2(li1, li3)  # 纵向相加
            # check_bit_2 = colEncoder_2(check_bit, li3, '00000111111')  # 纵向 之前纵向写的是i不是li3

            cal_check = li1[0:23]
            cal_check = int(''.join(cal_check), 2)
            original_check = li1[23:]
            original_check = int(''.join(original_check), 2)
            _, reminder = divmod(cal_check, 511)
            if reminder == original_check:
                check_state.append(0)  #正常的是0
            else:
                check_state.append(1)




    print(check_state.count(1))

    for xuhao,judgement in enumerate(check_state):

        if xuhao == lenth-1 and judgement != 0:    #judgement不等于0也就是意味着等于1，意味着这个元素需要恢复
            if check_state[0] != 1:  #同时我还需要保证这个元素的前一个元素是良好的，才能去恢复。如果两个都被损毁则将其中前一个设置成为0
                precise_recovery(c[0], '00000111111', c[xuhao])  # 执行准确恢复
                #print(float(Binary32Tofloat(''.join(c[xuhao]))))
            else:

                rough_recovery(c, xuhao, check_state)  # 执行粗糙恢复
        elif judgement != 0:
            if judgement == 1 and check_state[xuhao+1] != 1:
                precise_recovery(c[xuhao + 1], '00000111111', c[xuhao])  # 执行准确恢复
                print((Binary32Tofloat(''.join(c[xuhao]))))

            else:

                rough_recovery(c, xuhao, check_state)  # 执行粗糙恢复

    c = suminsideStr(c)
    #print(c)
    for index, i in enumerate(c):
        c[index] = Binary32Tofloat(i)
    break
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


#restored_list = restored_list.tolist()
restored_list = np.array(restored_list)
restored_list = restored_list.reshape(tuple(b))
restored_list = torch.from_numpy(restored_list)

# c = np.array(c)
# c = c.reshape(tuple(b))
# c = torch.from_numpy(c)


parmeters.data = restored_list
print(parmeters)

model.float()

torch.save(model, './models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu-attack-0.01-recover.pkl')
