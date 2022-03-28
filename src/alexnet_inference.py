# coding:utf-8
# @Time:2022/3/13 17:36
# @Author:LHT
# @File:alexnet_inference
# Contact:1042528352@qq.com
# @Softwar:PyCharm

import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models
import time

BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labels_file(path_labels):
    """
    读取标签文件
    :param path_labels: 文件路径
    :return: 全部标签
    """
    with open(path_labels, "r", newline='') as file:
        labels = json.load(file)
    return labels


def img_transform(img_r, config_transform=None):
    """
    设置转换函数
    :param img_r: 输入图片 PIL
    :param config_transform: 转换格式
    :return: tensor
    """
    if config_transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")
    img_t = config_transform(img_r)  # img->transforms
    return img_t


def img_preprocess(path_img):
    """
    预处理
    :param path_img: 图片路径
    :return: Tensor张量
    """
    # 设置transform
    norm_mean = [0.485, 0.456, 0.406]  # 根据ILSVRC-2012数据集计算得出，RGB均值
    norm_std = [0.229, 0.224, 0.225]  # RGB标准差
    config_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    img_r = Image.open(path_img)  # 加convert()可改变图像通道（多了A透明度）eg:convert('RGBA')
    img_t = img_transform(img_r, config_transform)
    # chw->bchw
    img_t.unsqueeze_(0)  # 0:在0轴之前增加一个维度，1:在1轴增加一个维度
    img_t = img_t.to(device)  # 放在gpu上计算

    return img_t, img_r


def get_model(path_weight, vis_model=False):
    """
    加载模型
    :param path_weight: 模型权重文件
    :param vis_model: 布尔值，设置是否可视化网络结构
    :return: 返回模型
    """
    # 权重文件不包含网络
    # model = models.alexnet()  # 加载torch内置模型
    # pro_state_dict = torch.load(path_weight)  # 加载预训练权重
    # model.load_state_dict(pro_state_dict)

    # 权重文件包含网络
    model = torch.load(path_weight)
    """
    在train模式下，dropout网络层会按照设定的参数p，设置保留激活单元的概率（保留概率=p)。BN层会继续计算数据的mean和var等参数并更新。
    在eval模式下，dropout层会让所有的激活单元都通过，而BN层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
    训练用train(),测试用eval()
    """
    model.eval()  # paper中提到的test时要加*p
    # model.to(device)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224))  # 如果想在gpu上运行，需要先运行model.to(device)把type变成gpu的

    model.to(device)
    return model


if __name__ == '__main__':
    # ------------------------ step1 初始化 -------------------------------------------
    # 设置路径
    # 权重文件
    # path_weight_file = os.path.join(BASE_DIR, "model\\alexnet-owt-4df8aa71.pth")
    path_weight_file = os.path.join(BASE_DIR, "model/T_cat_and_dog.pth")
    # 输入图片
    path_img_file = os.path.join(BASE_DIR, "data/cat from baidu3（false）.jpg")
    # 标签文件
    # path_labels_file = os.path.join(BASE_DIR, "data/imagenet1000.json")
    path_labels_file = os.path.join(BASE_DIR, "data/T_cat_dog.json")

    # 加载标签文件
    class_labels = load_labels_file(path_labels_file)

    # 图片预处理, 返回tensor数据
    img_tensor, img_rgb = img_preprocess(path_img_file)

    # 加载预训练模型
    alexnet_model = get_model(path_weight_file, True)

    # ------------------------ step2 inference开始推理 --------------------------------
    with torch.no_grad():   # 不使用梯度计算
        time_sta = time.time()
        outputs = alexnet_model(img_tensor)
        time_end = time.time()

    # ------------------------ step3 根据index得到类别名称 ------------------------------
    # 得到predict和top5的类别索引
    _, pre_index = torch.max(outputs, dim=1)    # 第一个返回值为真实的数据值，第二个返回值为对应的索引
    # _, top5_index = torch.topk(outputs, k=5, dim=1)  # dim为0纵向压缩，1为横向

    # 根据索引输出对应类别名称
    pre_index = int(pre_index.cpu())
    pre_name = class_labels[pre_index]
    print("img:{}, predict is {}".format(os.path.basename(path_img_file), pre_name))
    # print("cost the time:%.2fs" % float(time_end - time_sta))
    print("cost the time:{:.2f}s".format(time_end - time_sta))

    # ------------------------ step4 可视化top5个类别 -----------------------------------
    # top5_index = top5_index.squeeze(0).cpu().numpy()
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pre_name))
    # top5_name = [class_labels[item] for item in top5_index]
    # for i in range(len(top5_name)):
    #     """
    #     text参数：x, y, str, ***
    #     bbox加边框，需要字典参数，默认是'square'方形边框，facecolor填充，alpha透明度
    #     """
    #     plt.text(5, 25+i*40, "top {}:{}".format(i+1, top5_name[i]), bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()

