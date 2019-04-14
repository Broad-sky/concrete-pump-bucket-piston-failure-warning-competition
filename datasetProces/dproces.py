__Author__ = "MEET SHEN"
___Time___ = "2019/1/22 17:26"

import os
import pandas as pd

# class datasetProces(object):
#     def __init__(self,file_dir):
#         self.file_dir = file_dir
#     def readFile(self):
#         for root, dirs, files in os.walk(self.file_dir):
#             print("************")
#             print(root)  # 当前目录路径
#             print(dirs)  # 当前路径下所有子目录
#             print(files)  # 当前路径下所有非目录子文件
# dp = datasetProces("/data_train")
# dp.readFile()

#读取训练集的文件名，并保存为list格式

def data_train_process():
    # 读取训练集的文件名，并保存为list格式
    lis_dir_train = os.listdir("D:\pycharm_program\cpcpv\data_train")
    data_train_label = pd.read_csv("train_labels.csv")#读取训练集的label
    sfn = data_train_label['sample_file_name'].values.tolist()#取出文件名
    l = data_train_label['label'].values.tolist()#取出label标签
    sfn_l = {}
    for i in range(len(sfn)):
        sfn_l[sfn[i]] = l[i]
    print(len(sfn_l))
    n = 0
    for ldt in lis_dir_train:
        # print(ldt)
        if ldt in sfn:
            print("********************n={0}**********".format(n))
            str_ldt = "D:\pycharm_program\cpcpv\data_train\\"+ldt
            print(str_ldt)
            dataset = pd.read_csv(str_ldt)
            print(ldt)#打印train_label的文件名
            print(sfn_l[ldt])#打印train_label的标签
            dataset['label'] = sfn_l[ldt]
            dataset1 = dataset.copy()
            if n == 0:
                dataset1.to_csv("data_train.csv",index=None)
                print(dataset1.head(3))
            if n >= 1:
                dataset1.to_csv("data_train.csv", index=None,header=None, mode='a')
                print(dataset1.head(3))
            n = n + 1
# data_train_process()

def data_test_process():
    # 读取测试集的文件名，并保存为list格式
    lis_dir_test = os.listdir("D:\pycharm_program\cpcpv\data_test")
    n = 0
    for ldt in lis_dir_test:
        print("********************n={0}**********".format(n))
        print(ldt)
        str_ldt = "D:\pycharm_program\cpcpv\data_test\\" + ldt
        dataset = pd.read_csv(str_ldt)
        if n == 0:
            dataset.to_csv("data_test.csv", index=None)
            print(dataset.head(3))
        if n >= 1:
            dataset.to_csv("data_test.csv", index=None, header=None, mode='a')
            print(dataset.head(3))
        n = n + 1
# data_test_process()