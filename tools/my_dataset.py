# coding:utf-8
# @Time:2022/3/19 11:29
# @Author:LHT
# @File:my_dataset
# Contact:1042528352@qq.com
# @Software:PyCharm

from torch.utils.data import Dataset
import PIL.Image as Image
import os
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个读取猫狗数据集的类
class CatDogDataset(Dataset):  # 为Dataset子类
    # 构造函数
    def __init__(self, data_dir, mode="train", train_proportion=0.9, transform=None, rand_seed=520):
        """
        :param data_dir: str,数据集所在路径
        :param mode: train/valid
        :param train_proportion: 默认训练集占0.9，验证集占0.1
        :param transform: 数据预处理方式 torch.transform
        :param rand_seed: 随机种子，用于打乱数据集
        """
        self.data_dir = data_dir
        self.mode = mode
        self.train_portion = train_proportion
        self.transform = transform
        self.r_seed = rand_seed
        # 调用_get_img_info函数返回数据集的每张图片路径和标签（path，label），在DataLoader中通过index读取样本
        self.data_info = self._get_img_info()

    # 索引函数，Dataset中为抽象      DataLoader中使用索引时会访问此函数
    def __getitem__(self, index):
        """
        Python的__getitem__(self,n)方法为拦截索引运算
            当实例s出现s[i]这样的索引运算时，Python会调用这个实例s继承的__getitem__(self,n)方法，并把s作为第一个参数传递（self）,将方括号内的索引值 i 传递给第二个参数 n
        :param index:
        :return:
        """
        # 获取图片路径和标签
        path_img, label = self.data_info[index]
        # 读取图片
        img = Image.open(path_img).convert('RGB')

        # 转换为tensor张量
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # 统计数据集数量
    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    # 获取数据集每张图片的路径和标签形成一个元组，并存放在一个列表里面
    def _get_img_info(self):
        """
        -endswith(suffix, start=None, end=None)
            方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False。
            可选参数"start"与"end"为检索字符串的开始与结束位置。
        -startswith()
            方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查。
        filter()
            函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
        list(a)会对a中的全部元素进行遍历，之后组成一个list,耗时多
        []仅仅是将a这个整体当成list的第一个元素
        label：cat为0，dog为1
        :return: img_path, label
        """
        # 读取文件夹，获取文件夹中的每个文件名称
        img_names = os.listdir(self.data_dir)  # 返回指定的文件夹包含的文件或文件夹的名字的列表
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        # 打乱文件顺序
        random.seed(self.r_seed)
        # shuffle() 方法将序列的所有元素随机排序
        random.shuffle(img_names)

        # 根据文件名形成一个标签列表
        img_labels = [0 if name.startswith('cat') else 1 for name in img_names]

        # 根据获得的文件名列表和标签列表进行train/valid分类
        split_idx = int(len(img_names) * self.train_portion)  # 25000*0.9=22500
        if self.mode == "train":
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("\nself.mode 无法识别，仅支持(train, valid)")

        # 组合路径与标签
        path_img_set = [os.path.join(self.data_dir, file_name) for file_name in img_set]
        data_info = [(name, label) for name, label in zip(path_img_set, label_set)]

        return data_info