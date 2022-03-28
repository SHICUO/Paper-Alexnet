# coding:utf-8
# @Time:2022/3/17 16:38
# @Author:LHT
# @File:alexnet_visualizaton
# Contact:1042528352@qq.com
# @Softwar:PyCharm

import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as model
import torch
import torch.nn as nn
import torchvision.utils as tvutils
import PIL.Image as Image
from torchvision.transforms import transforms

BASE_DIR = os.path.join(os.path.dirname(__file__), "../")

if __name__ == '__main__':
    # 创建可视化文件存放目录
    log_dir = os.path.join(BASE_DIR, "log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 加载模型
    path_weight_file = os.path.join(BASE_DIR, "model/alexnet-owt-4df8aa71.pth")
    alexnet_model = model.alexnet()
    pro_state_dict = torch.load(path_weight_file)
    alexnet_model.load_state_dict(pro_state_dict)

    # -------------------------------step 1 kernel visualization ----------------------------------
    # filename_suffix为存储log文件名的后缀
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_kernel')

    kernel_num = 1  # 当前核种类，1为第一个卷积层的卷积核
    vis_kernel = 2  # 要显示的卷积核种类数
    for each_layer in alexnet_model.modules():  # 遍历模型，得到子模型，里面会遍历每层
        # isinstance对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
        if not isinstance(each_layer, nn.Conv2d):  # 如果each_layer不是卷积层退出本次循环
            continue
        if kernel_num > vis_kernel:  # 如果当前卷积核超过要显示的数（一种卷积核对应一个卷积层），退出循环
            break

        kernels = each_layer.weight
        # print(kernels.shape)    # [64, 3, 11, 11]第一层卷积层的卷积核形状
        c_out, c_inp, k_high, k_wide = kernels.shape  # 输出通道，输入通道，核高，核宽

        for c_out_idx in range(c_out):
            # 遍历每个输出通道，形成[3, 11, 11]的形状chw      unsqueeze_是在替换原来数据操作
            kernel_co_idx = kernels[c_out_idx, :, :, :].unsqueeze(1)  # chw->bchw[3,1,11,11], make_grad直接收此形状数据
            """
                torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                功能：制作网格图像
                • tensor：图像数据, B*C*H*W形式
                • nrow：每行的图片数（列数自动计算）
                • padding：图像间距（像素单位）
                • normalize：是否将像素值标准化
                • range：标准化范围
                • scale_each：是否单张图维度标准化
                • pad_value：padding的像素值
            """
            kernel_grid = tvutils.make_grid(kernel_co_idx, nrow=c_inp, normalize=True, scale_each=True, padding=1)
            # global_step代表现在是第几张图片
            writer.add_image("No.{}_Convlayer_split_in_channel".format(kernel_num), kernel_grid, global_step=c_out_idx)

        # 显示每个核的RGB图像
        kernel_all = kernels.view(-1, 3, k_high, k_wide)  # 3, h, w     把不是rgb3通道的改为3通道
        kernel_grid_all = tvutils.make_grid(kernel_all, nrow=8, normalize=True, scale_each=True, padding=1)
        writer.add_image("No.{}_Convlayer_all_kernel".format(kernel_num), kernel_grid_all, global_step=1)

        print("No.{}_Convlayer shape: {}".format(kernel_num, tuple(kernels.shape)))
        kernel_num += 1

    # -------------------------------step 2 feature visualization ----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_feature')

    # 加载图片，并把图片转换成tensor
    img_path = os.path.join(BASE_DIR, "data/Golden Retriever from baidu.jpg")
    img_rgb = Image.open(img_path).convert('RGB')
    norm_mean = [0.485, 0.456, 0.406]  # 根据ILSVRC-2012数据集计算得出，RGB均值
    norm_std = [0.229, 0.224, 0.225]  # RGB标准差
    config_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img_tensor = config_transform(img_rgb).unsqueeze_(0)  # chw->bchw  有_代表在原来数据上处理，结果原数据发送改变

    # 取第一层卷积后的特征图
    convlayer1 = alexnet_model.features[0]
    f_map1 = convlayer1(img_tensor)  # [1,64,55,55]

    f_map1.transpose_(0, 1)  # [64,1,55,55]
    f_map1_grid = tvutils.make_grid(f_map1, nrow=8, padding=1, normalize=True, scale_each=True)
    writer.add_image("feature map in conv1", f_map1_grid, global_step=1)

    print("feature map in conv1 of shape: {}".format(f_map1.transpose(0, 1).shape))
    writer.close()
