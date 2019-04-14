__Author__ = "MEET SHEN"
___Time___ = "2019/1/23 8:52"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import xgboost
from sklearn.metrics import classification_report
import datetime
import os
import lightgbm
from sklearn.feature_selection import RFE
import warnings

warnings.filterwarnings("ignore")

dataset = pd.read_csv("data_train.csv")
# dummies = pd.get_dummies(dataset['设备类型'])
map_t = {"ZV7e8e3":1,"ZVe0672":2,"ZV75a42":3,"ZV41153":4,"ZV90b78":5,"ZV55eec":6,"ZVc1d93":7}
dataset['设备类型'] = dataset['设备类型'].map(map_t)
# dataset = dataset.drop(columns = ['发动机转速','流量档位','低压开关',
#                                    '高压开关', '搅拌超压信号', '正泵', '反泵']).copy()

dataset = dataset.drop(columns = ['设备类型'])
# print("dataset.isnull().sum():",dataset.isnull().sum())
col_list = dataset.columns.tolist()
array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]

# transformer1 = MinMaxScaler(feature_range=(0, 1))
# x_train_std = transformer1.fit_transform(X)  # 训练集x_train标准化

# transformer2 = MinMaxScaler(feature_range=(0, 1)))
# y_train_std = transformer2.fit_transform(y_train)  # 训练集y_train标准化
# transformer3 = MinMaxScaler(feature_range=(0, 1))
# x_validation_std = transformer3.fit_transform(x_validation)  # 测试集x_validation标准化
# transformer4 = MinMaxScaler(feature_range=(0, 1))
# y_validation_std = transformer4.fit_transform(y_validation)  # 测试集y_validation标准化

validation_size = 0.3#划分30%为验证集
seed = 7
n_split = 10#十折交叉验证

# x_train0, x_test, y_train0, y_test \
#     = train_test_split(x_train_std,Y,test_size=validation_size,random_state=seed)#得到测试集

validation_size1 = 0.2

x_train,x_validation,y_train,y_validation \
    = train_test_split(X,Y,test_size=validation_size1,random_state=seed)#得到训练集与验证集

# print("y_validation:\n", y_validation)

# model = xgboost.XGBClassifier()
'''
    boosting_type='gbdt', num_leaves=120, reg_alpha=0, reg_lambda=0.01,
    max_depth=-1, n_estimators=3000,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1
'''
model = lightgbm.LGBMClassifier(
    boosting_type='gbdt',
    class_weight=None,
    colsample_bytree=1.0,#feature_fraction小于1, LightGBM会在每次迭代中随机选择部分特征,0-1
    importance_type='split',
    learning_rate=0.1,
    max_depth=-1,#Maximum tree depth for base learners, -1 means no limit.
    min_child_samples=20,#一个叶子上数据的最小数量. 可以用来处理过拟合.
    min_child_weight=0.001,#一个叶子上的最小 hessian 和. 类似于 min_data_in_leaf, 可以用来处理过拟合.
    min_split_gain=0.0,#执行切分的最小增益
    n_estimators=8000,
    n_jobs=-1,
    num_leaves=500,
    objective=None,
    random_state=None,
    reg_alpha=0,
    reg_lambda=0,
    silent=True,
    subsample=0.8,#类似于 feature_fraction,它将在不进行重采样的情况下随机选择部分数据
    subsample_for_bin=200000,
    subsample_freq=1)

start_time = datetime.datetime.now()
print("***********************开始训练模型及验证模型{0}*********************".format(str(start_time)))
model.fit(x_train,y_train,
          eval_set=[(x_validation,y_validation)],
          early_stopping_rounds=50,#如果一个验证集的度量在 early_stopping_round 循环中没有提升, 将停止训练
          eval_metric="auc")
prediction = model.predict(x_validation)
print(prediction)
cr = classification_report(y_validation,prediction)
print(cr)
# Keras/Tensorflow+python+faster RCNN训练自己的数据集
end_test_model_time = datetime.datetime.now()
print("***********************结束模型测试集预测{0}*********************".format(str(end_test_model_time-start_time)))

# 保存模型
def save_model(model1,cr1):
    with open("model.txt",mode='a') as f:
        f.write("\n")
        f.write(str(datetime.datetime.now()))
        f.write("\n")
        f.write(str(model1))
        f.write("\n")
        f.write(str(cr1))
save_model(model,cr)

def predict_test(model):
    print("***********************开始预测测试集{0}*********************".format(str(datetime.datetime.now())))
    lis_dir_test = os.listdir("D:\pycharm_program\cpcpv\data_test")
    n = 0
    data_sample_file = pd.DataFrame(columns={"Id", "Label"})
    for ldt in lis_dir_test:
        print(ldt)
        str_ldt = "D:\pycharm_program\cpcpv\data_test\\" + ldt
        data = pd.read_csv(str_ldt)
        data = data.drop(columns=['设备类型'])
        # map_t = {"ZV7e8e3": 1, "ZVe0672": 2, "ZV75a42": 3, "ZV41153": 4, "ZV90b78": 5, "ZV55eec": 6, "ZVc1d93": 7}
        # data['设备类型'] = data['设备类型'].map(map_t)
        data0 = data.values
        # transformer = MinMaxScaler(feature_range=(0, 1))
        # data1 = transformer.fit_transform(data0)
        pre_value = model.predict(data0)
        one_amount = np.count_nonzero(pre_value)#预测值中为1的个数
        all_amount = len(pre_value)#预测值所有的个数
        # print(all_amount)
        zero_amount = all_amount - one_amount
        # print(one_amount,zero_amount)
        if zero_amount >= one_amount:
            data_sample_file = data_sample_file.append({'Id': ldt, 'Label': 0}, ignore_index=True)
        else:
            data_sample_file = data_sample_file.append({'Id': ldt, 'Label': 1}, ignore_index=True)
        # print(pre_value)
        # print(data_sample_file)
        # if n%10000 == 0:
        #     with open('pre_value.txt', 'a') as f:
        #         f.write(str(ldt))
        #         f.write(str(pre_value))
        #         f.write("\n")
        #         f.write("预测值总数: " + str(all_amount))
        #         f.write("\n")
        #         f.write("预测值为0: " + str(zero_amount) + "  预测值为1: " + str(one_amount))
        #         f.write("\n")
        #     # data_sample_file.to_csv("submit{0}.csv".format(str(n)),index = None)
        #     # with open('data_submit.txt','a') as f:
        #     #     f.write(str(data_sample_file))
        n = n+1
        if n % 100 == 0:
            print(n)
    data_sample_file.to_csv("submit 20190216 v1.csv",index=None)
predict_test(model)
