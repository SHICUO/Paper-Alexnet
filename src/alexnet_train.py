# coding:utf-8
# @Time:2022/3/20 12:59
# @Author:LHT
# @File:alexnet_train
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm
import numpy as np
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader
from torchvision.models import alexnet
import torch.optim as optim
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
import sys

sys.path.append(BASE_DIR)
from tools.my_dataset import CatDogDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NUM = 2  # 分类总数
MAX_EPOCH = 20  # 训练迭代次数
BATCH_SIZE = 256  # 批次大小
LR = 0.001  # 学习率
tra_log_interval = 1    # train损失日志的记录间隔(每隔多少个batch记录一次)
val_log_interval = 1
start_epoch = 0
lr_decay_step = 1   # 学习率衰减步长，每过lr_decay_step个epoch衰减


def get_model(path, vis_model=False):
    """
    加载模型
    :param path: 预训练模型路径
    :param vis_model: 是否可视化模型
    :return:
    """
    model = alexnet()
    pro_state_dict = torch.load(path)
    model.load_state_dict(pro_state_dict)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device='cpu')

    return model


if __name__ == '__main__':
    # ----------------------------- step 1 初始化 ----------------------------------
    path_state_dict = os.path.join(BASE_DIR, "model/alexnet-owt-4df8aa71.pth")
    path_data_dir = os.path.join(BASE_DIR, "data/train")

    # 设置tra和val集的transform
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    """
    Resize()  
        将输入图像的大小调整为给定的大小
        此操作是遵循短边先变为给定大小，然后再（按比例）调整长边。     区别直接（256,256）
    CenterCrop() 
        截取中心图片
    CenterCrop() 
        随机截取图片
    RandomHorizontalFlip() 
        以概率P随机进行水平翻转
    TenCrop() 
        将给定的图像裁剪成四个角和中央裁剪加上翻转的版本这些(水平翻转默认使用),总共10张
    torch.stack(sequence, dim=0)
        沿一个新维度对输入张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，而stack（）函数指定的dim参数，就是新增维度的（下标）位置。
        当dim = -1时默认最后一个维度；[1][2][3] dim=0,为[1]所在维度 dim=1,为[2]所在维度
    """
    # 对应paper里面的first数据增强1->2048
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 先把图片变为短边为256，长边按比例（256*？）
        transforms.CenterCrop(256),  # 截取中间256，（256*256）
        transforms.RandomCrop(224),  # 使用随机裁剪得到224*224图片，数量变为1024倍[(256-224)**2]
        transforms.RandomHorizontalFlip(p=0.5),  # 1024*2倍
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    # paper里的test阶段1->10
    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.TenCrop(224, vertical_flip=False),  # 水平翻转
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops]))
    ])

    # 构造dataset实例       # 返回的每张图片数据为tuple类型，要用DataLoder转换为tensor张量
    train_data = CatDogDataset(data_dir=path_data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=path_data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle为True打乱数据
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ----------------------------- step 2 加载模型 ----------------------------------
    alexnet_model = get_model(path_state_dict, vis_model=False)

    # 把连接层最后一层改为(4096,2)
    in_features = alexnet_model.classifier[6].in_features
    alexnet_model.classifier[6] = nn.Linear(in_features, CLASS_NUM)
    alexnet_model.to(device)

    # ----------------------------- step 3 设置损失函数 -------------------------------
    """
    CrossEntropyLoss(输入张量,目标值)
    """
    criterion = nn.CrossEntropyLoss()

    # ----------------------------- step 4 设置优化器 ---------------------------------
    """
    map() 会根据提供的函数对指定序列做映射。
        第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
    id() 函数返回对象的唯一标识符，标识符是一个整数。
        CPython 中 id() 函数用于获取对象的内存地址。
    StepLR() 步长衰减（每过lr_decay_step个epoch衰减一次）
        step_size 衰减步长  gamma 学习率的衰减率
    ReduceLROnPlateau() 当指标停止提高时，降低学习速度
    """
    # 设置优化器，随机梯度下降SGD
    # 设置flag，控制是否冻结卷积层，==1时冻结，==0默认训练所有
    flag = 1
    if flag:
        fc_params = alexnet_model.classifier.parameters()
        fc_params_id = list(map(id, fc_params))  # 返回参数的内存地址列表
        conv_params = filter(lambda x: id(x) not in fc_params_id, alexnet_model.parameters())    # 卷积层的参数
        optimizer = optim.SGD([
            {'params': conv_params, 'lr': LR * 0.1},  # 使卷积层学习率小。如果乘以0相当于不学习，使用预训练参数
            {'params': fc_params, 'lr': LR}], momentum=0.9)  # 设置动量为0.9，即原来参数向量为0.1
    else:
        optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

    # 学习率调度器，设置学习率下降策略
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)

    # ----------------------------- step 5 开始训练 ------------------------------------
    # 用于存储loss值，方便训练完后显示曲线
    train_curve_loss = list()
    valid_curve_loss = list()
    time_sta = time.time()

    for epoch in range(start_epoch, MAX_EPOCH):
        loss_total_num = 0.  # 损失值总数
        correct_num = 0.     # 预测正确数
        total = 0.           # 样本数

        """
        alexnet_model.train()
            训练时设置为train,测试时设置为eval
            目的：符合paper中测试时要将输出乘P（dropout概率）的设计
        enumerate() 返回迭代器的每个值和对应的下标
        loss.item() 返回每张图片训练完后计算出的损失值
        """
        alexnet_model.train()
        # 把25000*0.9=22500张图片分为128一个batch，总共176个（128*176=22528）批次
        for batch_num, batch_data in enumerate(train_loader):
            # 以每批次中的128张图片为整体进行训练
            inputs, labels = batch_data     # [128,3,224,224]  [128]
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape, labels.shape)
            # 喂入数据
            outputs = alexnet_model(inputs)     # [128,2]

            # backward
            optimizer.zero_grad()   # 设置所有优化的梯度趋向于0
            loss = criterion(outputs, labels)
            loss.backward()     # 计算梯度

            # update weights
            optimizer.step()

            # 统计预测分类情况
            _, predicted = torch.max(outputs, dim=1)    # 第一个返回值为真实的数据值，第二个返回值为对应的索引，1为横向压缩
            total += labels.size(0)     # 每次加128
            correct_num += (predicted == labels).squeeze().cpu().sum().numpy()  # 通过计算True总数求得预测正确的总数

            # 打印训练信息
            loss_total_num += loss.item()
            train_curve_loss.append(loss.item())    # 添加后面可视化曲线的y值
            if (batch_num+1) % tra_log_interval == 0:
                loss_mean = loss_total_num / tra_log_interval
                print("Training: Epoch[{:0>3}/{:0>3}] Batch[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH-1, batch_num, len(train_loader)-1, loss_mean, correct_num / total))
                loss_total_num = 0.

        # scheduler.step()    # 更新学习率
        scheduler.step(loss)

        # 验证集验证
        if (epoch+1) % val_log_interval == 0:
            # 设置初始量
            loss_num_val = 0.        # 损失值总数
            correct_num_val = 0.     # 预测正确数
            total_val = 0.           # 样本数

            alexnet_model.eval()
            with torch.no_grad():
                for batch_num, batch_data in enumerate(valid_loader):
                    # 加载数据
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)   # inputs[4,10,3,224,224]

                    # 进行维度变换   [4,10,3,224,224]->[40,3,224,224]
                    b, ncrops, c, h, w = inputs.shape
                    outputs = alexnet_model(inputs.view(-1, c, h, w))   # [40,2]
                    # [40,2]->[4,10,2]->[4,2] 取10张变换图片中的最高值，即最可能的
                    outputs_avg = outputs.view(b, ncrops, -1).mean(1)

                    # 计算损失值
                    loss = criterion(outputs_avg, labels)

                    # 统计验证集分类情况
                    _, predicted_val = torch.max(outputs_avg, dim=1)
                    correct_num_val += (predicted_val == labels).squeeze().cpu().sum().numpy()
                    total_val += labels.size(0)

                    loss_num_val += loss.item()

                # 打印训练信息
                loss_mean_val = loss_num_val / len(valid_loader)
                valid_curve_loss.append(loss_mean_val)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH-1, loss_mean_val, correct_num_val / total_val))

        time_end = time.time()
        print("Cost of time form epoch0 to epoch{}: {:.2f}min".format(epoch, (time_end-time_sta)/60))

    # 保存模型网络和权重
    print("保存网络及权重中========>")
    save_path_weight = os.path.join(BASE_DIR, "model/T_cat_and_dog.pth")
    torch.save(alexnet_model, save_path_weight)
    print("保存成功========>>位置：", save_path_weight)

    # range() 返回的是一个可迭代对象（类型是对象），而不是列表类型
    train_x = range(len(train_curve_loss))
    train_y = train_curve_loss

    tra_batch_num = len(train_loader)
    # 由于valid中记录的是epochloss，需要对记录点进行转换到每个batch
    valid_x = np.arange(1, len(valid_curve_loss)+1) * tra_batch_num * val_log_interval  # np.arange()在给定间隔内返回均匀间隔的值
    valid_y = valid_curve_loss

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.title('BATCH OF LOSS CURVE')
    plt.legend(loc='upper right')   # 设置图例
    plt.xlabel('Batch')
    plt.ylabel('loss value')
    plt.show()




