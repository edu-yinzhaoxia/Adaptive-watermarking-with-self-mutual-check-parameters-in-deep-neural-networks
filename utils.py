
import struct
import numpy as np
import torch
from model import Model
import math
import random
import  copy


def shuffle_and_restore(original_list,seednum):
    random.seed(seednum)

    lenth = len(original_list)
    a = random.sample(range(0,lenth),lenth)

    #original_list = original_list.tolist()
    # 待随机打乱的列表
    #original_list = [1, 2, 3, 4, 5]
    oss = []
    for i in a:
        oss.append(original_list[i])
    # ad = list(range(lenth))
    # for index,i in enumerate(a):
    #     ad[i] = oss[index]

    # # 用random.shuffle()函数打乱列表顺序
    # random.shuffle(oss)
    #
    # # 记录打乱前每个元素的下标
    # index_list = list(range(len(original_list)))
    #
    # # 将每个元素的下标和值组成元组，存储在一个新列表中
    # tuple_list = list(zip(index_list, original_list))
    #
    # # 将元组列表按照原始列表元素值的顺序排序
    # sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1])
    #
    # # 获取排序后元组列表中每个元素的下标
    # sorted_index_list = [t[0] for t in sorted_tuple_list]
    #
    # # 按照排序后的下标序列重新构造原始列表
    # restored_list = [original_list[i] for i in sorted_index_list]


    return oss,original_list,a,lenth


def floatToBinary32(value):
    val = '{:032b}'.format(struct.unpack('I', struct.pack('f', value))[0])
    return str(val)

def Binary32Tofloat(B):  #B是二进制字符串
    return struct.unpack('<f', struct.pack('<I', int(B,2)))[0]


def flattenM (X): #X是矩阵或者是数组

    a = X.flatten()
    b = X.shape
    return a,b

def reshapeA (L,m,n):    #L是一维数组 ,m,n是长和宽
    L = L.reshape(m,n)
    return L

def deal_matrix(L):

    a,m,n = flattenM(L)


def add4bit (explist):
    explist.insert(0,'0')
    explist.insert(3,'0')
    explist.insert(6,'1')
    explist.insert(9,'1')
    return explist

def sum_faction(li):    #传入一个[0,1,0,1,1,1,..]返回一个00011101的模二序列
    a = li[0:1]
    b = li[9:20]
    newlist = a + b

    newlist = ''.join(newlist)
    #newlist = int(newlist,2)
    #newlist = bin(newlist)[2:]
    #print(newlist)
    return newlist
def sum_factionD(li):
    a = li[0:1]
    b = li[9:14]
    newlist = a + b

    newlist = ''.join(newlist)
    #newlist = int(newlist,2)
    #newlist = bin(newlist)[2:]
    #print(newlist)
    return newlist

def sum_exp(li):        #（插入了4个比特）
    a = li[1:9]
    a = add4bit(a)

    newlist = ''.join(a)
    # newlist = int(newlist,2)
    # newlist = bin(newlist)[2:]
    return newlist

def Dense_exp_fact(li1): #li1 是exp部分包含[1,6]不包含符号位，li1是fraction部分[9，13]，还有另加上一个首位1
    v = li1[1:9]
    a = ''.join(v)
    a = int(a,2)
    #print(a)
    #v = bin(v)[2:].zfill(12)
    if a > 128:
        v = ['1','1','1','1','1']
    elif a < 115:
        v = ['0','0','0','0','0']

    else:
        a -= 113
        a = bin(a)[2:].zfill(5)
        a = str(a)

        v = list(a)


    #a = li1[1:7]

    b = li1[0:1]
    c = li1[9:14]


    d = b + v  + c




    newlist = ''.join(d)
    return newlist




def XOR(li1,li3): #li1为处理序列，l3为校验序列，返回的是12位离散bit
    li11 = copy.deepcopy(li1)

    li1 = sum_faction(li1)
    li1 = int(li1,2)


    # li2 = sum_exp(li11)
    # li2 = int(li2,2)


    li3 = sum_faction(li3)
    li3 = int(li3,2)

    checkbit = li1^li3
    #checkbit = checkbit^li3

    # checkbit = int(checkbit,2)
    # checkbit = bin(checkbit)[2:]
    checkbit = bin(checkbit)[2:].zfill(12)

    checkbit = str(checkbit)
    result = [str(c) for c in checkbit]

    return result

def XOR_condense (li1,li3): #产生横向的校验位

    li1 = Dense_exp_fact(li1)
    li1 = int(li1,2)


    # li2 = sum_exp(li11)
    # li2 = int(li2,2)


    li3 = Dense_exp_fact(li3)
    li3 = int(li3,2)

    checkbit = li1^li3
    #checkbit = checkbit^li3

    # checkbit = int(checkbit,2)
    # checkbit = bin(checkbit)[2:]
    checkbit = bin(checkbit)[2:].zfill(16)

    checkbit = str(checkbit)
    result = [str(c) for c in checkbit]

    return result

def oneonebitcreator(ls):
    v = ls[0:11]
    a = ''.join(v)
    a = int(a, 2)
    #newlist = ''.join(a)
    return a

def onetwo_23_bitcreator(ls):
    v = ls[12:23]
    a = ''.join(v)
    a = int(a, 2)
    #newlist = ''.join(a)
    return a


def XOR_condense_2 (li1,li3): #产生横向的校验位

    li1 = oneonebitcreator(li1)
    #li1 = int(li1,2)
    li3 = oneonebitcreator(li3)
    #li3 = int(li3,2)
    # li2 = sum_exp(li11)
    # li2 = int(li2,2)


    # li3 = Dense_exp_fact(li3)
    # li3 = int(li3,2)

    #checkbit = a^a2
    checkbit = li1^li3

    # checkbit = int(checkbit,2)
    # checkbit = bin(checkbit)[2:]
    checkbit = bin(checkbit)[2:].zfill(11)

    checkbit = str(checkbit)
    result = [str(c) for c in checkbit]

    return result




# def assign_last12(li1,li3,i): #li1是处理12位checkbit，li3是校验list，最后得到全新的li3,
#     li3[-12:] = li1
#     return li3
def assign_last12(li1,c,i): #li1是处理12位checkbit，li3是校验list，最后得到全新的li3,
  c[i][-12:] = li1


def assign_last12_2(li1,c,i): #li1是处理12位checkbit，li3是校验list，最后得到全新的li3,
  c[i][12:23] = li1
  #print(len(c[i]))

def suminsideStr (ls):#对【【1，2,3】，【1,2，】】变成【【123】，【123】】
    for index,i in enumerate(ls):
        a = "".join([str(num) for num in i])
        ls[index] = a
    return ls


def colEncoder (checkOnebit,ls):    #XOR传入的是ls传入的是一个包含32个元素的ls

    exp = sum_exp(ls)
    exp = int(exp,2)

    fac = sum_faction(ls)
    fac = int(fac,2)

    checkOnebit = ''.join(checkOnebit)
    checkOnebit = int(checkOnebit,2)


    checkbit_col = exp^fac

    checkbit_col = checkbit_col^checkOnebit

    checkbit_col = bin(checkbit_col)[2:].zfill(12)

    checkbit_col = str(checkbit_col)

    checkbit_col = list(checkbit_col)
    return checkbit_col

def colEncoder_2 (checkOnebit,ls,seed):    #XOR传入的是ls传入的是一个包含32个元素的ls
    #nexTzongxiang = onetwo_23_bitcreator(ls)
    # nexTzongxiang = ['1','1','1','1','1','1','1','1','1','1','1']
    # nexTzongxiang = ''.join(nexTzongxiang)
    # nexTzongxiang = int(nexTzongxiang,2)



    checkOnebit = ''.join(checkOnebit)#把list拼起来
    checkOnebit = int(checkOnebit,2)

    seed = int(seed,2)

    # checkbit_col = exp^fac

    #checkbit_col = nexTzongxiang^checkOnebit
    #checkbit_col = checkbit_col^seed

    checkbit_col = checkOnebit^seed
    checkbit_col = bin(checkbit_col)[2:].zfill(11)

    #checkbit_col = checkbit_col + '11111'  #之前加上5个1是想要看看有效数字到底多少仍然可以保持精度
    #checkbit_col = checkbit_col


    checkbit_col = str(checkbit_col)

    checkbit_col = list(checkbit_col)
    return checkbit_col


"检查工具"

def check_col(ls):           #先要减去竖向的值
    a = []

    a = ls[-12:]
    stra = ''.join(a)
    stra = int(stra,2)

    exp = sum_exp(ls)
    exp = int(exp, 2)

    fac = sum_faction(ls)
    fac = int(fac, 2)

    sum_EF = exp^fac

    result = sum_EF^stra
    result = bin(result)[2:].zfill(12)
    result = str(result)

    result = list(result)


    return result

def check_XOR(li1,li3): #ls1是处理序列，ls3是下一个序列 ,返回的是一个长度为12的list。用于横向校验
    li1 = Dense_exp_fact(li1)
    li1 = int(li1, 2)

    # li2 = sum_exp(li11)
    # li2 = int(li2,2)

    li3 = Dense_exp_fact(li3)
    li3 = int(li3, 2)

    checkbit = li1 ^ li3
    # checkbit = checkbit^li3

    # checkbit = int(checkbit,2)
    # checkbit = bin(checkbit)[2:]
    checkbit = bin(checkbit)[2:].zfill(12)

    checkbit = str(checkbit)
    result = [str(c) for c in checkbit]

    return result
"以下是哈夫曼编码的工具"

def transDecimal(ls):
    ls = ''.join(ls)
    #hexnum = oct(int(ls, 2))[2:]

    # decimal_num = int(ls, 2)
    #
    octal_str = hex(int(ls, 2))[2:].zfill(3)

    return octal_str

def last12bit_correspond(Flo):
    a = floatToBinary32(Flo)
    b = str(a)
    c = list(b)
    c = c[-21:]
    d = ''.join(c)
    e = int(d,2)
    return e

def last12bit_correspondsimple(Flo):
    a = floatToBinary32(Flo)
    b = str(a)
    c = list(b)
    c = c[11:12]
    d = ''.join(c)
    e = int(d,2)
    return e


"下面是找到他后面位的大小关系"

def comparebigxiao(deal_ls,shuffle,deal_location,shuffle_len):
    bigxiao = []
    if shuffle_len < 7:
        a = shuffle + shuffle + shuffle + shuffle
        for i in range(1,8):
            if a[deal_location] < a[deal_location+i]:
                bigxiao.append('1')         #如果前方的数字大于他自己则置1，
            else:
                bigxiao.append('0')
    else:
        if deal_location <= shuffle_len-1-7:
            for i in range(1,8):
                if shuffle[deal_location] < shuffle[deal_location+i]:
                    bigxiao.append('1')
                else:
                    bigxiao.append('0')
        else:
            a = shuffle + shuffle[0:7]
            for i in range(1,8):
                if a[deal_location] < a[deal_location+i]:
                    bigxiao.append('1')
                else:
                    bigxiao.append('0')
    deal_ls[-9:-2] = bigxiao


# def last_colcheck (ls,divisor):
#     a = ''.join(ls)
#     a = int(a[0:23],2)
#     quotient, remainder = divmod(a, divisor)
#     remainder = bin(remainder)[2:].zfill(2)
#     remainder = list(remainder)
#     ls[-2:] = remainder

def last_colcheck (ls,divisor):
    print(ls)
    a = ''.join(ls)
    a = int(a[0:23],2)
    quotient, remainder = divmod(a, divisor)
    remainder = bin(remainder)[2:].zfill(9)
    remainder = list(remainder)
    ls[-9:] = remainder
    print(ls)


def precise_recovery(ls,key,ls_rec):
    a = ls[0:11]        #ab是正确参数
    b = ls[12:23]
    a = int(''.join(a),2)
    b = int(''.join(b),2)
    key = int(key,2)
    c = a^b
    c = c^key
    c = bin(c)[2:].zfill(11)
    c = list(c)         #c 是计算出来的精确位
    ls_rec[0:11] = c

# def rough_recovery(all,index,check_state):
#     max_jiaoyan = [0.8]
#     min_jiaoyan = [-0.8]
#
#     for i,j in enumerate(check_state[index:index+7]):
#         if j != 1 :
#             if all[index+i][22+i]  == '1':
#                 max_jiaoyan.append(float(Binary32Tofloat(''.join(all[index+i]))))
#             else:
#                 min_jiaoyan.append(float(Binary32Tofloat(''.join(all[index+i]))))
#     youjixian = min(max_jiaoyan)
#     zuojixian = max(min_jiaoyan)
#     mean = (zuojixian + youjixian)/2
#
#     newmean = floatToBinary32(mean)
#     newmean = list(newmean)
#     t = float(Binary32Tofloat(''.join(all[index])))
#     all[index][:] = newmean[:]
#     print(float(Binary32Tofloat(''.join(all[index]))))

def rough_recovery(all,index,check_state):

    all[index] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]




def precise_recovery_1(ls,key):
    a = ls[0:11]        #ab是正确参数
    b = ls[12:23]
    a = int(''.join(a),2)
    b = int(''.join(b),2)
    key = int(key,2)
    c = a^b
    c = c^key
    c = bin(c)[2:].zfill(11)
    # c = list(c)         #c 是计算出来的精确位

    return c
