### 混凝土泵车砼活塞故障预警竞赛

|| 赛题背景

    对生产设备的维护，传统的做法主要有两类，一种是等故障发生后再维修，但这会导致非计划性的停产，经济损失大；第二种是以固定计划进行维护，但维修成本高，停机时间长。预测性维护，则通过分析故障历史数据和实时监测数据，对设备关键部件的剩余寿命或故障进行提前预测预警，并据此进行维护维修，从而减少设备非计划停机时间、降低维护成本。

    （Traditional approaches to production equipment maintenance remain mainly in corrective maintenance and routine-based maintenance. Corrective maintenance, which is no fail no repair, implies unscheduled downtime of equipment, severe security risk and unacceptable economic loss, while routine or time-based maintenance is time consuming and costly. By data analysis and modeling, predictive maintenance forecasts the residual life or possible failure of key components, and schedules maintenance accordingly，in order to avoid unscheduled downtime and to save cost.）

    砼活塞是混凝土泵车的关键部件，也是消耗性部件，活塞故障将导致泵车无法正常工作，同时可能导致整个工地其他配套设备无法正常施工，从而带来相当大的经济损失。活塞寿命与设备的具体工况等密切相关，通过物联网将泵车的实时工况数据等上传至工业互联网云平台，基于积累的数据建立合适的模型，有望对砼活塞在未来一定工作任务期间内可能出现的故障做出有效的预测预警，从而提醒作业人员在施工前进行必要的维护，避免因计划外停机而带来的经济损失。

    （Concrete pistons are key and consumptive components of concrete pump vehicles. Failure of the piston will cause malfunction of the pump vehicle, which may even halt the whole construction site and bring unacceptable economic loss. Piston life is closely related to the specific operation conditions of the equipment. First, we upload floor data of pump vehicles through Industrial Internet of Things (IIoT) to the cloud platform. Second, we train appropriate models based on the accumulated data. And then we predict the possible failure of the concrete piston, and accordingly remind the operators to carry out necessary maintenance before construction, which is promising to avoid economic loss caused by unscheduled downtime.）

|| 文件清单和使用说明

文件中包含以下内容：

    1.train：本文件夹中存放用于训练的采集数据，每个csv文件为泵车工作n小时内采样得到的一个数据样本。

    2.test：本文件夹中存放用于测试的采集数据，每个csv文件为泵车工作n小时内采样得到的一个数据样本。

    3.train_labels.csv：用于训练的标注信息。

|| 训练数据说明

训练数据集包括：

    1.train文件夹下的训练特征数据，每个csv文件为一个样本，utf-8编码格式；

    2.train_labels.csv文件包含的标注信息，utf-8编码格式，其中每一行对应train文件下的一个训练特征数据文件。

注：

sample_file_name为train文件夹下csv样本文件的名称，label为该文件对应的标注信息，0表示该样本对应活塞在未来泵送2000方混凝土任务期间内未发生故障，1表示该样本对应的活塞在未来泵送2000方混凝土任务期间内发生了故障。

特征数据字段包括：活塞工作时长，发动机转速，油泵转速，泵送压力，液压油温，流量档位，分配压力，排量电流，低压开关，高压开关，搅拌超压信号，正泵，反泵，设备类型。其中：

    活塞工作时长：新换活塞后，累积工作时长，数值型。

    发动机转速、油泵转速、泵送压力、液压油温、流量档位、分配压力、排量电流：均为泵车的对应工况值，数值型。

    低压开关、高压开关、搅拌超压信号、正泵、反泵：开关量。

    设备类型：该泵车的类型，类别型。

除了开关量以外，上述设备类型、工况数据的具体值都经过了一定的脱敏处理，即不完全是实际测量值，但已考虑尽量不影响数据蕴含的关系等信息。

注：label只能为1或0，1表示该样本对应的活塞在未来2000方工作量内，会出现故障；0标识在未来2000方工作量内不会故障。具体参考提交样例

|| 评分方式

评分算法采用Macro-F1-Score的计算方式，计算方式如下：

计算作品的召回率（recall）和准确率（precision）：

    R（召回率）= 检测正确的目标数量/(检测正确的目标数量+漏检的目标数量)

    P（准确率）= 检测正确的目标数量/(检测正确的目标数量+检测错误的目标数量)

计算F1-Score：

    F1 = 2PR / (R + R)

线上Score得分为：

    Score=分类0的F1Score+分类1的F1Score/2

排名：Score值越高，排名越靠前。

|| 文件说明：

    主要包含两个python package：mainPredict、datasetProces
    datasetProces:主要用于对整理训练集与测试集

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

简单的对训练集进行处理，便能得到一个较好的分数，外加对模型进行细微的调参工作，能在线上得到0.8108793分！

后期由于主办方更换数据集，并且个人时间有限，便没有持续比赛！

最大的收获是：有时候一上来就是各种复杂的特征工程，反而不会取得很好的结果，定要相信大道至简！

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### 注：

    由于数据量较大，因此训练集、测试集的数据文件，please connect me : 1745379960@qq.com！！！
