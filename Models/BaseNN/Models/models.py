# -*- coding:utf-8 -*-

"""
基础神经网络模块, 所有神经网络都由该函数生成, 支持 pytorch 1.0 以上,
但建议 pytorch 1.8 以上, 否则无法转为 c 项目工程

Note:
    forward 函数等关键函数已经进行了程序加速处理

可供使用的参数:
    note : (可选) 指可不使用, 否则为必须参数, 不使用会报错

全局参数:
    "channel" : 输入数据 [batch size, channel, height, width]
    "height"  : 
    "width"   : 

网络参数:
    "convolution"
        "filters" : 卷积输出数据的通道数 (int)
        "size"    : 卷积核尺寸 (int)
        "stride"  : 卷积步长 (int)
        "pad"     : 是否需要零填充, 程序会自动选择填充多少, 不需人为设置 (0 or 1)

        "activation"    : 激活函数类型 (str) (可选)
        "normalization" : norm 类型 (str) (可选)
        "bias"          : 卷积是否加上 bias (0 or 1) (可选)
        "dilation"      : 空洞卷积间隔 (int) (可选)

        Note:
            考虑到 net 层数过多导致难以维护, 因此将 norm 和 active 并入 conv 层;
            如果不需要 norm 或 active, 在 cfg 文件中不声明即可, 程序会自动忽略值为 None 的声明;

    "DSconvolution" (depthwise separable convolution)
        "filters" : 卷积输出数据的通道数 (int)
        "size"    : 卷积核尺寸 (int)
        "stride"  : 卷积步长 (int)
        "pad"     : 是否需要零填充, 程序会自动选择填充多少, 不需人为设置 (0 or 1)

        "activation"    : 激活函数类型 (str) (可选)
        "normalization" : norm 类型 (str) (可选)
        "bias"          : 卷积是否加上 bias (0 or 1) (可选)
        "use_se"        : 是否使用 se 注意力机制, 如果使用, 则值为 divide (可选)

    "route"
        "layers"     : 数据分叉或拼接
        "separation" : 数据分叉时可用, "front" or "rear" (可选)

    "shortcut"
        "from" : 与 -1 层相加的层数

    "attention" (修改自: swin transformer v1)
        "heads"       : 多头注意力机制中的头数 (int)
        "window_size" : 每个窗口的大小 (int)

        "roll" : 是否使用 shift (0 or 1) (可选)
        "relative_pos_embedding" : 是否使用位置编码 (0 or 1) (可选)

    "channelMLP" (channel linear)
        "hidden_dim" : linear 隐藏层层数 (int)
        "activation" : 激活函数, 使用方法和上述一样 (str)

    "upsample" (上采样)
        "size"         : 输出的大小 (可选)
        "scale_factor" : 上采样的倍数 (可选)

        Note:
            either size or scale_factor should be defined

        "mode"          : 'nearest'(默认), 'linear' (可选)
        "align_corners" : True or False (可选)

    "activation"
        "activation" : 'swish', 'hardSwish', 'mish', 'ReLU', 'LeakyReLU', 'GELU', 'RELU6', 'ELU'

    "normalization"
        "normalization" : 'BatchNorm', "LayerNormTrans", 'LayerNorm1d', 'LayerNorm2d', 'LayerNorm3d'

    "focus"

    "head"
"""

import torch
import torch.nn as nn

from Models.BaseNN.Models.roll_transformer import Attention, ChannelMLP
from Models.BaseNN.Models.layers import *


class BASENN(nn.Module):
    """
    An implementation of a neural network;
    BASENN : base neural network

    : parameter args   : network argparse

    几个重要的属性:
        self.output_filters = []  # 每一层的 feature map channel
        self.output_size = []     # 每一层的 feature map height or width
        self.outputs = {}         # 每一层的神经网络输出值, 但是不包括输入
    """

    def __init__(self, cfg_file):
        super(BASENN, self).__init__()

        nn_cfg = str(cfg_file)  # 网络参数

        self.A_N = False  # 是否 激活函数(activation) 在 归一化(normalization) 前面; True or False
        self.affine_en = False  # BatchNorm2d =? False
        self.He_weights_init = False  # 是否针对 relu 使用 he init; True or False

        self.blocks = self._parse_cfg(nn_cfg)
        self._create_modules(self.blocks)  # 创建网络
        print("A network initialization completed")

    def _parse_cfg(self, cfgfile):
        """
        获取 cfg 参数
        output:
            一个 list: blocks; blocks 的每个元素都是一个字典, 字典里面储存了一个模块的全部信息
        """
        file = open(cfgfile, 'r')
        lines = file.read().split('\n')  # lines 的每个元素是 cfg 的每一行
        lines = [x for x in lines if len(x) > 0]  # 删除空行
        lines = [x for x in lines if x[0] != '#']  # 删除注释
        lines = [x.rstrip().lstrip() for x in lines]  # 删除字符串左右两边的空格

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":  # 找到一个模块的初始位置
                if len(block) != 0:  # 如果 block 不为 0, 说明 block 已经储存了上一个模块的数据
                    blocks.append(block)  # 则将上一个模块的数据添加入 blocks
                    block = {}  # 重置 block
                block["type"] = line[1:-1].rstrip()  # 储存每个 block 的 type 参数
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()  # 储存每个 block 的 key 和 value 参数
        blocks.append(block)

        return blocks

    def _create_convolution(self, prev_filters, prev_size, index, x):
        """
        convolution block = convolution + Norm + activation or convolution + activation + Norm
        :param prev_filters : 上一层 feature map 的 channel
        :param prev_size    : 上一层 feature map 的 height, width; 需要 height = width
        :param index        : 该模块在神经网络中处于第几层
        :param x            : 该模块的 cfg 参数, 根据这些参数构建网络

        声明:
            在 cfg 中声明必要参数, 可选参数不声明即代表不使用

        note:
            1. 如果卷积后面有 BN, 则建议不加 bias, 因为没有用;
            2. 有的时候 activation 放在 normalization 前面更有用.
        """
        module = nn.Sequential()

        # 必需参数
        filters = int(x["filters"])  # 卷积输出数据的通道数
        kernel_size = int(x["size"])  # 卷积核尺寸
        stride = int(x["stride"])  # 卷积步长
        padding = int(x["pad"])  # 是否零填充

        # 可选参数 -- start
        try:
            activation = str(x["activation"])  # 激活函数
        except:
            activation = None

        try:
            normalization = str(x["normalization"])  # 归一化
        except:
            normalization = None

        try:
            bias = True if int(x["bias"]) else False  # 偏移量
        except:
            bias = False

        try:
            dilation = int(x["dilation"])  # 空洞卷积的空洞间隔
        except:
            dilation = 1
        # 可选参数 -- end

        if dilation == 1:
            pad = ((kernel_size - 1) // 2) if padding else 0  # 卷积补零
        else:
            pad = ((((dilation - 1) * (kernel_size - 1) + kernel_size) - 1) // 2) if padding else 0  # 卷积补零

        fea_size = (prev_size + (stride - 1)) // stride  # 本层输出 feature map 的大小

        # add the convolution layer(卷积)
        conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, dilation=dilation, bias=bias)
        module.add_module("conv_{0}".format(index), conv)

        # add the Norm layer(归一化) and the activation(激活函数)
        if bool(self.A_N):  # active 在 norm 前面
            if activation != None:
                module = self._create_activation(index, x, module)
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
        else:
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
            if activation != None:
                module = self._create_activation(index, x, module)

        return module, filters, fea_size

    def _create_depthwise_separable_convolution(self, prev_filters, prev_size, index, x):
        """
        深度可分离卷积 (depthwise separable convolution) : 参考 mobile net v3
        depthwise convolution + (a / n) + (se) + pointwise convolution + (a / n)

        Note:
            depthwise conv 和 pointwise conv 后面是否需要 activation 或 norm , 仍未有定论;
            部分论文认为不使用 a/n 会有影响, 部分论文认为没有影响, 部分论文认为最后一步需要使用;

            对于 depthwise conv 和 pointwise conv 后面是否需要 bias , 也仍未有定论;
            有说法认为 depthwise conv 后不用 bias 而 pointwise conv 需要 bias;

            所以目前该实现全部保留为可选参数, 如需要可在 cfg 中声明, 若不声明, 则默认不使用;
            也可以选择 a,n 前后位置
        """
        module = nn.Sequential()

        filters = int(x["filters"])  # 卷积输出数据的通道数
        kernel_size = int(x["size"])  # 卷积核尺寸
        stride = int(x["stride"])  # 卷积步长
        padding = int(x["pad"])  # 零填充

        # 可选参数 -- start
        try:
            activation = str(x["activation"])  # 激活函数
        except:
            activation = None

        try:
            normalization = str(x["normalization"])  # 归一化
        except:
            normalization = None

        try:
            bias = True if int(x["bias"]) else False  # 偏移量
        except:
            bias = False

        try:
            use_se = True if int(x["use_se"]) else False  # 是否使用 se 注意力机制
            divide = int(x["use_se"])
        except:
            use_se = False
            divide = None
        # 可选参数 -- end

        pad = ((kernel_size - 1) // 2) if padding else 0  # 卷积补零
        fea_size = (prev_size + (stride - 1)) // stride  # 本层输出 feature map 的大小

        # depthwise convolution: out_channels = in_channels, groups = in_channels
        depthwise_conv = nn.Conv2d(prev_filters, prev_filters, kernel_size, stride, pad, groups=prev_filters, bias=bias)
        module.add_module("depthwise_conv_{0}".format(index), depthwise_conv)

        # add the Norm layer(归一化) and the activation(激活函数)
        if bool(self.A_N):  # active 在 norm 前面
            if activation != None:
                module = self._create_activation(index, x, module)
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
        else:
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
            if activation != None:
                module = self._create_activation(index, x, module)

        # 可加 se 注意力机制
        if use_se and divide != None:
            SEblock = SqueezeAndExcite(prev_filters, prev_filters, divide)
            module.add_module("SEblock_{0}".format(index), SEblock)

        # pointwise convolution: kernel_size=1, stride=1
        pointwise_conv = nn.Conv2d(prev_filters, filters, kernel_size=1, stride=1, bias=bias)
        module.add_module("pointwise_conv_{0}".format(index), pointwise_conv)

        # add the Norm layer(归一化) and the activation(激活函数)
        if bool(self.A_N):  # active 在 norm 前面
            if activation != None:
                module = self._create_activation(index, x, module)
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
        else:
            if normalization != None:
                module = self._create_normalization(filters, fea_size, index, x, module)
            if activation != None:
                module = self._create_activation(index, x, module)

        return module, filters, fea_size

    def _create_route(self, index, x):
        """
        # route layer
        # layers 参数仅一个, 提取某一层的 feature map, 即数据分叉出去
          但是如果在 cfg 文件中声明 separation, 则分叉数据的前一半或者后一半, 具体可见 _route 方法
        # layers 参数有两个, 将该两层 feature map 拼接(concat 模块), 不是相加
        # layers 参数有三个, 将该三层 feature map 拼接(concat 模块)
        # layers 参数有四个, 将该四层 feature map 拼接(concat 模块)
        """
        module = nn.Sequential()

        x["layers"] = x["layers"].split(',')
        if len(x["layers"]) == 1:
            start = int(x["layers"][0])
            if start > 0:  # 如果 start 是绝对 index
                start = start - index
            filters = self.output_filters[index + start]
            fea_size = self.output_size[index + start]

            # 是否只取其中的一部分
            try:
                separation = str(x["separation"])
            except:
                separation = None

            if separation != None:
                filters = filters // 2

        elif len(x["layers"]) == 2:
            start = int(x["layers"][0])
            end = int(x["layers"][1])
            if start > 0:  # 如果 start 是绝对 index
                start = start - index
            if end > 0:
                end = end - index
            filters = self.output_filters[index + start] + self.output_filters[index + end]
            fea_size = self.output_size[index + start]

        elif len(x["layers"]) == 3:
            start = int(x["layers"][0])
            mid = int(x["layers"][1])
            end = int(x["layers"][2])
            if start > 0:  # 如果 start 是绝对 index
                start = start - index
            if mid > 0:
                mid = mid - index
            if end > 0:
                end = end - index
            filters = self.output_filters[index + start] + self.output_filters[index + mid] + self.output_filters[
                index + end]
            fea_size = self.output_size[index + start]

        elif len(x["layers"]) == 4:
            start = int(x["layers"][0])
            front = int(x["layers"][1])
            rear = int(x["layers"][2])
            end = int(x["layers"][3])
            if start > 0:  # 如果 start 是绝对 index
                start = start - index
            if front > 0:
                front = front - index
            if rear > 0:
                rear = rear - index
            if end > 0:
                end = end - index
            filters = self.output_filters[index + start] + \
                      self.output_filters[index + front] + \
                      self.output_filters[index + rear] + \
                      self.output_filters[index + end]
            fea_size = self.output_size[index + start]

        else:
            print("route layer cfg parameter is wrong")

        route = EmptyLayer()
        module.add_module("route_{0}".format(index), route)
        return module, filters, fea_size

    def _create_shortcut(self, index):
        """
        # shortcut layer, 作用于残差模块(rest unit)中的 add
        # 两层 feature map 相加, 数值相加
        # 数据输入输出通道数不变, filters 不变
        """
        module = nn.Sequential()
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(index), shortcut)
        return module

    def _route(self, i, module, x):
        layers = module["layers"]
        layers = [int(a) for a in layers]

        if len(layers) == 1:
            if (layers[0]) > 0:
                layers[0] = layers[0] - i
            x = self.outputs[i + (layers[0])]

            try:
                separation = str(module["separation"])
            except:
                separation = None

            if separation != None:
                sep_index = (self.output_filters[i + (layers[0])] // 2)
                if separation == "front":
                    x = x[:, :sep_index, :]
                elif separation == "rear":
                    x = x[:, sep_index:, :]
                else:
                    print(" error: there is not this separation type!")

        elif len(layers) == 2:
            if (layers[0]) > 0:
                layers[0] = layers[0] - i
            if (layers[1]) > 0:
                layers[1] = layers[1] - i
            map1 = self.outputs[i + layers[0]]
            map2 = self.outputs[i + layers[1]]
            x = torch.cat((map1, map2), 1)

        elif len(layers) == 3:
            if (layers[0]) > 0:
                layers[0] = layers[0] - i
            if (layers[1]) > 0:
                layers[1] = layers[1] - i
            if (layers[2]) > 0:
                layers[2] = layers[2] - i
            map1 = self.outputs[i + layers[0]]
            map2 = self.outputs[i + layers[1]]
            map3 = self.outputs[i + layers[2]]
            x = torch.cat((map1, map2, map3), 1)

        elif len(layers) == 4:
            if (layers[0]) > 0:
                layers[0] = layers[0] - i
            if (layers[1]) > 0:
                layers[1] = layers[1] - i
            if (layers[2]) > 0:
                layers[2] = layers[2] - i
            if (layers[3]) > 0:
                layers[3] = layers[3] - i
            map1 = self.outputs[i + layers[0]]
            map2 = self.outputs[i + layers[1]]
            map3 = self.outputs[i + layers[2]]
            map4 = self.outputs[i + layers[3]]
            x = torch.cat((map1, map2, map3, map4), 1)
        return x

    def _shortcut(self, i, module, x):
        from_ = int(module["from"])
        x = self.outputs[i - 1] + self.outputs[i + from_]
        return x

    def _create_attention(self, prev_filters, prev_size, index, x):
        """
        该方法的实现思路来自 swin transformer v1 及其变体
        note:
            attention 输入输出大小不变
        """
        module = nn.Sequential()

        heads = int(x["heads"])  # 多头注意力机制中的头数
        window_size = int(x["window_size"])  # 每个窗口的大小

        try:
            shifted = True if int(x["roll"]) else False  # 是否移动数据以获得跨区域视野
        except:
            shifted = False

        try:
            relative_pos_embedding = True if int(x["relative_pos_embedding"]) else False  # 是否位置编码
        except:
            relative_pos_embedding = False

        att = Attention(heads, prev_filters, prev_size, window_size, shifted, relative_pos_embedding)
        module.add_module("attention_{0}".format(index), att)
        return module

    def _create_upsample(self, prev_size, index, x):
        module = nn.Sequential()
        try:
            stride = int(x["scale_factor"])
            fea_size = int(prev_size * stride)
        except:
            stride = None

        try:
            fea_size = size = int(x["size"])
        except:
            size = None

        try:
            mode = str(x["mode"])
        except:
            mode = "nearest"

        try:
            if x["align_corners"] != None:
                align_corners = True if int(x["align_corners"]) == 1 else False
            else:
                align_corners = None
        except:
            align_corners = None

        module.add_module("upsample_{}".format(index),
                          Interpolate(size=size, scale_factor=stride, mode=mode, align_corners=align_corners))
        return module, fea_size

    def _create_channel_mlp(self, prev_filters, index, x):
        module = nn.Sequential()
        hidden_dim = int(x["hidden_dim"])
        activation = str(x["activation"])

        cmlp = ChannelMLP(prev_filters, hidden_dim, activation)
        module.add_module("channel_MLP_{0}".format(index), cmlp)
        return module

    def _create_activation(self, index, x, module=None, enInplace=True):
        '''
        如果 activation 作为卷积模块中的一部分, 则 inplace=True;
        当 activation 作为独立一个模块时, inplace=False, 防止其上一层数据被覆盖
        '''
        if module == None:
            module = nn.Sequential()
            enInplace = False

        activation = str(x["activation"])
        if activation == "ReLU":
            module.add_module("ReLU_{0}".format(index), nn.ReLU(inplace=enInplace))
        elif activation == "LeakyReLU":
            activn = nn.LeakyReLU(0.1, inplace=enInplace)
            module.add_module("LeakyReLU_{0}".format(index), activn)
        elif activation == "GELU":
            module.add_module('GELU_{0}'.format(index), GELU())
        elif activation == "RELU6":
            module.add_module("RELU6_{0}".format(index), nn.ReLU6(inplace=enInplace))
        elif activation == "swish":
            module.add_module('swish_{0}'.format(index), Swish())
        elif activation == "hardSwish":
            module.add_module('hardSwish_{0}'.format(index), HardSwish(enInplace))
        elif activation == "mish":
            module.add_module('mish_{0}'.format(index), Mish())
        elif activation == "ELU":
            module.add_module("ELU_{0}".format(index), nn.ELU(inplace=enInplace))
        elif activation != None:
            print(activation + " : there is not this type of activation!")

        return module

    def _create_normalization(self, prev_filters, prev_size, index, x, module=None):
        if module == None:
            module = nn.Sequential()

        normalization = str(x["normalization"])
        if normalization == "BatchNorm":
            module.add_module("batch_norm_{0}".format(index), nn.BatchNorm2d(prev_filters, affine=self.affine_en))
        elif normalization == "LayerNormTrans":
            module.add_module("layer_norm_trans_{0}".format(index), LayerNormTrans(prev_filters))
        elif normalization == "LayerNorm1d":
            module.add_module("layer_norm_1d_{0}".format(index), LayerNorm1d(prev_size))
        elif normalization == "LayerNorm2d":
            module.add_module("layer_norm_2d_{0}".format(index), LayerNorm2d(prev_size, prev_size))
        elif normalization == "LayerNorm3d":
            module.add_module("layer_norm_3d_{0}".format(index), LayerNorm3d(prev_filters, prev_size, prev_size))
        elif normalization != None:
            print(normalization + " : there is not this type of normalization!")

        return module

    def _create_focus(self, prev_filters, prev_size, index, x):
        module = nn.Sequential()
        module.add_module("focus_{0}".format(index), Focus())
        # x(b,c,h,w) -> y(b,4*c,h/2,w/2)
        filters = 4 * prev_filters
        size = prev_size // 2
        return module, filters, size

    def create_head(self):
        self.head = nn.Sequential()
        self.head.add_module("head", EmptyLayer())
        print("Head is called but it has not been redefined by user.")

    def head_forward(self, *x):
        print("Head Forward has not been redefined by user, so it return itself.")
        return x

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _create_modules(self, blocks):
        self.module_list = nn.ModuleList()

        self.output_filters = []  # 每一层的 feature map channel
        self.output_size = []  # 每一层的 feature map height or width
        self.outputs = {}  # 每一层的神经网络输出, 但是不包括输入

        # 避免在循环中使用 '.' 这种操作以提高运行速度; 因为每使用一次 '.', 都会暗中调用特定函数
        module_append = self.module_list.append
        filters_append = self.output_filters.append
        size_append = self.output_size.append

        # net 全局参数
        net_info = blocks[0]
        if int(net_info["height"]) == int(net_info["width"]):
            prev_size = int(net_info["height"])
        else:
            prev_size = min(int(net_info["height"]), int(net_info["width"]))
        prev_filters = int(net_info["channels"])  # 输入数据的通道数, 初始化值为图片通道数

        self.input_size = [prev_filters, prev_size]  # 输入图像的大小

        # net 局部参数
        for index, x in enumerate(blocks[1:]):
            # 检查 block 的 type
            # 为每个 block 创建 module
            # append each block to module_list

            if (x["type"] == "convolution"):
                module, filters, fea_size = self._create_convolution(prev_filters, prev_size, index, x)
            elif (x["type"] == "DSconvolution"):
                module, filters, fea_size = self._create_depthwise_separable_convolution(prev_filters, prev_size, index,
                                                                                         x)
            elif (x["type"] == "attention"):
                module = self._create_attention(prev_filters, prev_size, index, x)
            elif (x["type"] == "channelMLP"):
                module = self._create_channel_mlp(prev_filters, index, x)
            elif (x["type"] == "route"):
                module, filters, fea_size = self._create_route(index, x)
            elif x["type"] == "shortcut":
                module = self._create_shortcut(index)
            elif x["type"] == "upsample":
                module, fea_size = self._create_upsample(prev_size, index, x)
            elif x["type"] == "activation":
                module = self._create_activation(index, x)
            elif x["type"] == "normalization":
                module = self._create_normalization(prev_filters, prev_size, index, x)
            elif x["type"] == "focus":
                module, filters, fea_size = self._create_focus(prev_filters, prev_size, index, x)
            elif x["type"] == "head":
                self.create_head()
            else:
                print(str(x["type"]), " : there is not this type of block !")

            # 每遍历一个 block, module_list 添加一个 module
            module_append(module)

            prev_filters = filters  # prev_filters 是 block 输入数据的通道数
            prev_size = fea_size  # prev_size 是输入 feature map 的长宽

            filters_append(filters)  # filters 是 block 输出数据的通道数
            size_append(fea_size)  # fea_size 是输出 feature map 的长宽

        if self.He_weights_init:
            self.weights_initialization()  # 参数初始化

    def forward(self, x):
        outputs = self.outputs  # 这样做是为了在循环中减少变量使用 '.' 操作
        modules = self.blocks[1:]

        type_list = ["convolution", "DSconvolution", "attention", "channelMLP", "upsample", "activation",
                     "normalization", "focus"]

        # 各层输出
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type in type_list:
                x = self.module_list[i](x)
            elif module_type == "route":
                x = self._route(i, module, x)
            elif module_type == "shortcut":
                x = self._shortcut(i, module, x)
            elif module_type == "head":
                x = self.head_forward(x)
            else:
                print("Warning: there is no this kind of module type!")
            outputs[i] = x

        self.outputs = outputs
        return x

    def show_net_info(self):
        print("net module structure:\n")
        print("input size:")
        c, l = self.input_size[0], self.input_size[1]
        print("channel:" + str(c) + "  size of input:" + str(l) + "\n")

        for index, module in enumerate(self.module_list):
            print("layer " + str(index) + ":")
            print(module)

            c = str(self.output_filters[index])
            temp = str(self.output_size[index])
            print("filter num:" + c + "  size of filter:" + temp + "\n")

    def show_layer_output(self, index):
        # 将某一层的结果输出
        print(self.outputs[index].size())
        return self.outputs[index]

    def load_weights(self, directory, i):
        # 载入 network 权重, directory 是保存路径, i 是迭代次数
        self.load_state_dict(torch.load(directory + "/" + str(i) + '.pth'))

    def save_weights(self, directory, i):
        # 保存 network 权重, directory 是保存路径, i 是迭代次数
        torch.save(self.state_dict(), directory + "/" + str(i) + '.pth')
        # torch.save(self.state_dict(), directory + "\\" + str(i) + '.pth', _use_new_zipfile_serialization=False)
