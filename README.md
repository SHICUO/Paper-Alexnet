# ALEXNET

### The Architecture

![image-20220327211041286](.\notes\alexnet_ar.png)

<img src=".\notes\alexnet_code.png">

5个卷积层加3个全连接

各卷积层进行pad，目的在于使特征图分辨率不变，只有pool时才变

#### conv1-->ReLU-->Pool-->LRN

conv.11×11，stride=4，96 kernels，

可计算出参数为3×11×11×96+96=34,944

max-pool.3×3，stride=2

#### conv2-->ReLU-->Pool-->LRN

conv.5×5，stride=1，256 kernels，padding=2

可计算出参数为96×5×5×256+256=614,656

max-pool.3×3，stride=2

#### conv3-->ReLU

conv.3×3，stride=1，384 kernels，padding=1

可计算出参数为256×3×3×384+384=885,120

#### conv4-->ReLU

conv.3×3，stride=1，384 kernels，padding=1

可计算出参数为384×3×3×384+384=1,327,488

#### conv5-->ReLU-->Pool

conv.3×3，stride=1，256 kernels，padding=1

可计算出参数为384×3×3×256+256=884,992

max-pool.3×3，stride=2

#### ==torch的代码中间加了一个平均池化，

是为了在输出入图片size不能卷积到全连接需要的6×6时进行自适应处理。==

#### fully-connected-->ReLU

可计算出参数为256×6×6×4096+4096=37,752,832

#### fully-connected-->ReLU

可计算出参数为4096×4096+4096=16,781,312

#### fully-connected-->ReLU-->softmax

可计算出参数为4096×1000+1000=4,097,000

total_parameter=62,378,344 ，全连接占据的参数超过50%。

全连接后面也可用1×1卷积替代，效果一样

## **Reducing Overfifitting**

### **Data Augmentation**

#### train（horizontal reflections水平翻转）

![image-20220327222104970](.\notes\data_.png)

1. 图片按比例把最短变缩放为256\*256

2. 对随机位置裁剪224\*224区域

   通过对图片进行256里面取224可得到32\*32的数据集规模增强

3. 对图片以P=0.5的概率进行水平翻转

   又可得到\*2的数据集规模增强

   （256-224）\*\*2\*2=2048。

#### validation、test阶段

1. 图片按比例把最短变缩放为256\*256
2. 对256\*256的图片进行左上，左下，右上，右下，中间，裁剪出5张图片
3. 对5张图片均进行水平翻转，得到10张

两个阶段也都使用了PCA方法修改RGB通道的像素值，实现颜色扰动

#### Dropout

全连接层进行随机失活P=0.5（随机失活概率）

训练和测试两阶段的数据尺度不同，在测试时要将神经元输出值乘以P

##### 为什么只有输出乘P？

训练时是训练所有权重（即所有参数都被训练出来，并没有减少参数）

尽管随机失活，但共享所有权重，只是一些在训练时为了训练其他权重而变为0，训练完成后既减少了过拟合（权重之间相互依赖减少）又得到所有权重。

测试时因为是进行此张输入图片的测试，此时的权重经过dropout只剩下一半，对应的输出也就要\*0.5，才体现了数学上等式两边的合理近似。

这是train的100%的权重与对应的output
$$
\sum^{100\%}W_iX_i=M
$$
这是test的50%的权重与对应的output要加上*0.5
$$
\sum^{50\%}W_iX_i=M*0.5
$$

## 论文摘抄

- Their capacity can be controlled by varying their depth and breadth.（1 Introduction p2） 

- Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then cropped out the central 256×256 patch from the resulting image.(2 Dataset p3)

- ReLUs have the desirable property that they do not require input normalization to prevent them from saturating(3.3 LRN p1)

- The network has learned a variety of **frequency**- and **orientation**-selective kernels, as well as various **colored** blobs.(6.1 p1)

  卷积核学习到频率、方向、颜色的特征

- If two images produce feature activation vectors with a small Euclidean separation, we can say that **the higher levels of** the neural network consider them **to be similar**.(6.1 p3)

- This should produce a much **better image retrieval method** than applying autoencoders to the raw pixels.(6.1 p4)

  卷积后的高级特征可用于图片检索，并且效果更好

卷积可提取图片大量特征
