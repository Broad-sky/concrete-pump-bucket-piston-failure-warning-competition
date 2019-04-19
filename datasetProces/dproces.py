__Author__ = "MEET SHEN"
___Time___ = "2019/1/22 17:26"

import os
import pandas as pd

root_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.split(root_path)[0]

def data_train_process():
    lis_dir_train = os.listdir(data_path+"\data_train")
    data_train_label = pd.read_csv("train_labels.csv")#读取训练集的label
    sfn = data_train_label['sample_file_name'].values.tolist()#取出文件名
    l = data_train_label['label'].values.tolist()#取出label标签
    sfn_l = {}
    for i in range(len(sfn)):
        sfn_l[sfn[i]] = l[i]
    n = 0
    for ldt in lis_dir_train:
        if ldt in sfn:
            str_ldt = root_path+ldt
            dataset = pd.read_csv(str_ldt)
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
    lis_dir_test = os.listdir(data_path+"\data_test")
    n = 0
    for ldt in lis_dir_test:
        str_ldt = data_path+"\data_test\\" + ldt
        dataset = pd.read_csv(str_ldt)
        if n == 0:
            dataset.to_csv("data_test.csv", index=None)
            print(dataset.head(3))
        if n >= 1:
            dataset.to_csv("data_test.csv", index=None, header=None, mode='a')
            print(dataset.head(3))
        n = n + 1
# data_test_process()
