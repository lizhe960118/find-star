# 2019 未来杯高校AI挑战赛 区域赛作品

* 战队编号：{330}
* 战队名称: {light}
* 战队成员：{李哲， 王阔}

## 概述
我们使用RetinaNet检测的思路，使用github上关于RetinaNet公开开源的[模型](https://github.com/kuangliu/pytorch-retinanet)进行优化。
超新星检测主要属于小物体检测，并且每张图片只有一个标注框，我们做了以下改进：
1. 将RetinaNet的p3-p7改为p2-p5,仅使用前几层加强小物体的检测
2. 在训练时，由于正类的框很少，取锚框和实际框的iou大于0.1为正类框，防止模型预测能力过低。
3. 使用三张图叠加的图训练网络

## 系统要求

### 硬件环境要求

* CPU: Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
* GPU: 1080Ti
* 内存: 20.00GB
* 硬盘: 1.8T

### 软件环境要求

* 操作系统: {Ubuntu} {16.04.1-Ubuntu}
* {python} {3.6.5}
* {Anaconda} {3.0}
* {cuda} {9.0.176}
* {pytorch} {0.4.0}


### 数据集
未使用官方提供的数据集之外的数据。

## 数据预处理
### 方法概述
1. 标签的转化：将给定的csv文件转化为可以处理的包含预测框和类别的txt文件。
将训练文件转化为：find_star_train_bbx_gt.txt，每两行的定义如下：  
- 第一行为图片的路径：这里我们传入img的前缀路径，但是在处理的时候处理三张图，在神经网络的输入的时候也输入三张图。
- 第二行为图片的中的gt_bbx,{cx, cy, w, h, classes}。这里的classes有8种{noise，ghost, pity, new_target, isstar, asteriod, isnova, known}。w，h我们根据class的类别加以指定。  
这里我们主要做了两方面，一是将路径放在第一行，而是将类别标签进行转化，三是给不同类数据指定大小不同的w，h.
2. 数据的清洗：
如果样本点的坐标标注超出了图片的范围，则不会选择此数据,也就是不进行处理。
3. 观察数据，对每个中点确定的星座，我们选取一个15 * 15 大小的框
4. 想办法将在训练集中让少样本多选取几次，这里对于new_target， isnova类的数据，让他们重复40次；对于ghost重复20次，对于asteroid，known，重复5；对于isstar，重复两次；对于noise，pity设置0.5的概率选取一半。所有结果保存下来，最后打乱写入到train_bbx_gt.txt
6. 在加载数据集时，设置random_flip, random_crop操作进行增强，最后resize到input_size，目前设计网络输入为600
### 操作步骤
将训练的csv转化为train_bbx_gt.txt
在csv2train_txt.py的test方法中修改文件路径，然后执行
```
python csv2train_txt.py
```

### 模型
训练后的模型存储地址：
./checkpoint/ckpt.pth
模型文件大小：
66.46M

## 训练
### 训练方法概述
可直接按照默认参数训练
```
python train.py
```

### 训练操作步骤
可以指定以下参数：
- 学习率：lr
- 训练数据路径：train_dataset（$DATASET_DIR）
- 训练回合数：train_epoch
- 存储模型的路径：model

示例如下：
```
python train.py --lr=1e-3 --train_dataset='./data/af2019-cv-training-20190312'  --train_epoch=200  --model=checkpoint
```
### 训练结果保存与获取
每一次的训练结果保存到文件夹‘./checkpoint/’中, 也可以通过以下命令自己定义：
```
python --model=$MODEL_DIR
```

在程序中，使用如下代码加载参数到网络中：
```
checkpoint = torch.load('./checkpoint/ckpt.pth')   
net.load_state_dict(checkpoint['net'])
```

## 测试
可直接按照默认参测试
```
python test.py
```
### 方法概述
可以指定以下参数：
- 测试数据路径：test_dataset（$TESTSET_DIR）
- 使用训练完成的模型进行预测：model ($MODEL_DIR)
- 保存结果的文件：prediction_file ($PREDICTION_FILE)


### 操作步骤
示例如下：
```
python test.py --test_dataset='./data/af2019-cv-testA-20190318' --model='checkpoint' --prediction_file='./data/find_star_txt_folder/submit.csv'
```

## 其他
无# find-star
