#-*- coding:utf-8 -*-

"""
    一些在神经网络中与 layer(层) 有关的实现
    这些实现支持 pytorch 1.2 及以上
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    """ upsample """
    """
    上采样, 默认采样率为 2
    
    : Input and Output :
    if input = (N, C, W_in), output = (N, C, W_out);
    if input = (N, C, H_in, W_in), output = (N, C, H_out, W_out);
    if input = (N, C, D_in, H_in, W_in), output = (N, C, D_out, H_out, W_out);
    
    : mode : 可供选择的上采样模式, 默认使用'nearest'
    'nearest', 'linear', 'bilinear', 'bicubic' , 'trilinear' , 'area'. 
    '最近邻', '线性'(3D-only), '双插值线性'(4D-only), '双三次'(4D-only), '三插值线性'(5D-only), 'area 算法'
    
    : align_corners :
    几何上, 我们认为输入和输出的像素是正方形, 而不是点;
    如果设置为 True,  则输入和输出张量由其角像素的中心点对齐, 从而保留角像素处的值;
    如果设置为 False, 则输入和输出张量由它们的角像素的角点对齐, 插值使用边界外值的边值填充;
    当 scale_factor 保持不变时, 使该操作独立于输入大小;
    仅当 mode == 'linear', 'bilinear' or 'trilinear' 时可以使用
    """
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        x = self.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class LayerNormTrans(nn.Module):
    """
        transformer 常用的 layer norm
        本质上和 LayerNorm2d 原理相近, 无太大区别
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class LayerNorm1d(nn.Module):
    """ 对 input 的最后一层进行 normalization """
    """
    eg: input.size = [b, c, h, w], 则对 w 层进行 normalization
    
    :param weight: w 通道值
    
    note: elementwise_affine 参数是否需要学习, False 表示 norm 参数不需学习
    """
    def __init__(self, weight):
        super().__init__()
        self.norm = nn.LayerNorm(weight, elementwise_affine=False)
    
    def forward(self, x):
        return self.norm(x)


class LayerNorm2d(nn.Module):
    """ 对 input 的最后两层进行 normalization """
    """
    eg: input.size = [b, c, h, w], 则对 h, w 层进行 normalization, 即对整张图片进行 norm
    
    :param height: h 通道值
    :param weight: w 通道值
    
    note: elementwise_affine 参数是否需要学习, False 表示 norm 参数不需学习
    """
    def __init__(self, height, weight):
        super().__init__()
        self.norm = nn.LayerNorm([height, weight], elementwise_affine=False)
    
    def forward(self, x):
        return self.norm(x)


class LayerNorm3d(nn.Module):
    """ 对 input 的最后三层进行 normalization """
    """
    eg: input.size = [ba, a, b, g], 则对 a, b, g 层进行 normalization
    
    :param alpha : alpha 通道
    :param beta  : beta  通道
    :param gamma : gamma 通道
    """
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.norm = nn.LayerNorm([alpha, beta, gamma], elementwise_affine=False)
    
    def forward(self, x):
        return self.norm(x)


class GELU(nn.Module):
    # 激活函数 GELU : GELU 效果和 mish 差不多, 但是计算资源少
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Swish(nn.Module):
    # 激活函数 Swish
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class HardSwish(nn.Module):
    # 激活函数 HardSwish
    def __init__(self, enInplace):
        super().__init__()
        self.re = nn.ReLU6(inplace=enInplace)
    
    def forward(self, x):
        return x * self.re(x + 3) / 6


class Mish(nn.Module):
    # 激活函数 mish
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class EmptyLayer(nn.Module):
    # 空的层, 被 route layer 与 shortcut layer 调用
    def __init__(self):
        super(EmptyLayer, self).__init__()


class SqueezeAndExcite(nn.Module):
    """ se 注意力机制 """
    def __init__(self, in_channels, out_channels, divide):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels, bias=False),
            HardSwish(True),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x)
        out = out.view(b, c)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return x * out


class Focus(nn.Module):
    # 数据切片 x(b,c,h,w) -> y(b,4*c,h/2,w/2)
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    """ ------------- test focus ------------- """
    input = torch.rand([1,2,8,8])
    print(input)
    focus = Focus()
    print(focus(input))
    print(focus(input).size())
    
    """ ------------- test se ------------- """
    input = torch.rand([2,16,32,32])
    print(input.size())
    se = SqueezeAndExcite(16,16,4)
    print(se(input).size())
    
    """ ------------- test norm ------------- """
    input = torch.rand([1,2,4,4])
    print(input)
    norm1d = LayerNorm1d(4)
    print(norm1d(input))
    norm2d = LayerNorm2d(4, 4)
    print(norm2d(input))
    norm3d = LayerNorm3d(2, 4, 4)
    print(norm3d(input))
    
    input = torch.rand([1,2,4,4])
    print(input)
    norm = LayerNormTrans(2)
    print(norm(input))
    
    """ ------------- test upsample ------------- """
    input = torch.tensor([[[1., 2., 3., 4.]]])
    print(input.size())
    upsample = Interpolate(scale_factor=2)
    print(upsample(input))
    upsample = Interpolate(scale_factor=2, mode='linear', align_corners=True)
    print(upsample(input))
    
    """ ------------- test activation ------------- """
    input = torch.from_numpy(np.arange(-10, 10, 0.1)).float()
    #acti = GELU()
    #acti = Swish()
    #acti = HardSwish()
    acti = Mish()
    plt.plot(input, acti(input))
    plt.show()
    
