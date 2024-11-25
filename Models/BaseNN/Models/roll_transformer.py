#-*- coding:utf-8 -*-

"""
    roll transformer 注意力机制
    该方法的实现思路来自 swin transformer v1 及其变体
"""

import torch
from torch import nn, einsum

import numpy as np
from einops import rearrange

from Models.BaseNN.Models.layers import *
#from layers import *


def get_relative_distances(window_size):
    """ 位置编码 """
    indices = torch.LongTensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class CyclicShift(nn.Module):
    """ 对 height, weight 进行滚动操作 """
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement
    
    def forward(self, x):
        # size of x : b c h w; 因此对 2, 3 维进行操作
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(2, 3))


def create_mask(window_size, displacement, zero_dimension=False, one_dimension=False):
    """ roll transformer 需要对计算结果的无效部分进行剔除; 函数提供最多两个维度方向上的 mask """
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    
    if zero_dimension:  # 第 0 维度 mask
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    
    if one_dimension:  # 第 1 维度 mask
        template = torch.zeros(window_size, window_size)
        template[-displacement:, :-displacement] = float('-inf')
        template[:-displacement, -displacement:] = float('-inf')
        
        for i in range(window_size ** 2)[0::window_size]:
            for j in range(window_size ** 2)[0::window_size]:
                mask[i:i+window_size, j:j+window_size] = template
    
    return mask


class Attention(nn.Module):
    """
    原理:
    conv encoding -> softmax(((q * k) / sqrt(d)) + mask) * v -> conv encoding
    
    注意力机制组成:
    1. depthwise separable convolution (to qkv encoding)
    2. MSA
    3. depthwise separable convolution (to output encoding)
    
    Note:
        to qkv 和 to output 的 dsconv 都是卷积后不带 norm 和 acti, 但是 dsconv 最后带了 bias;
    
    : input : size of input = [batch_size, feature_num(channel), len_signal]
    : parameter heads        : 多头注意力机制中的头数
    : parameter feature_num  : feature_num(channel)
    : parameter feature_size : len_signal
    : parameter window_size  : 数据将 len_signal 平均分成 window_num 份, 共 window_num 份, 每份大小 window_size
    
    : parameter shifted                : True / False; 是否移动数据以获得跨区域视野, default = False
    : parameter relative_pos_embedding : True / False; 是否设置位置编码; 因为卷积自带隐形位置编码, 因此 default = False
    """
    def __init__(self, heads, feature_num, feature_size, window_size, shifted=False, relative_pos_embedding=False):
        super().__init__()
        
        self.heads = heads
        self.feature_num = feature_num
        self.feature_size = feature_size
        self.window_size = window_size
        self.window_num = self.feature_size // self.window_size  # 窗口数
        
        self.shifted = shifted
        self.relative_pos_embedding = relative_pos_embedding
        
        #print(feature_size)
        assert self.feature_size % self.window_size == 0  # feature map 必须能被 window size 整除才能被切分
        assert self.feature_num % self.heads == 0         # feature map num 必须要被 head 整除, 各个 head 各取一部分 feature map
        
        # =========== to qkv encoding ===========
        self.qkv_dconv = nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, groups=feature_num, bias=False)
        self.qkv_pconv = nn.Conv2d(feature_num, feature_num*3, kernel_size=1, stride=1, bias=True)  # * 3 是因为有 q k v 三个张量
        
        # =========== to output encoding ===========
        self.out_dconv = nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, groups=feature_num, bias=False)
        self.out_pconv = nn.Conv2d(feature_num, feature_num, kernel_size=1, stride=1, bias=True)
        
        # =========== MSA ===========
        self.dk = (self.feature_num / self.heads) ** -0.5  # 缩放, 防止数据过大; 即 1/根号dk
        
        if self.shifted:
            displacement = self.window_size // 2
            
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            
            # 如果使用 nn.Parameter 则不用单独放入 device, 但是这样 net 会把它当成参数的一部分(虽然不求导)
            self.zero_dim_mask = nn.Parameter(create_mask(self.window_size, displacement, zero_dimension=True), requires_grad=False)
            self.one_dim_mask  = nn.Parameter(create_mask(self.window_size, displacement, one_dimension=True), requires_grad=False)
            #self.zero_dim_mask = create_mask(self.window_size, displacement, zero_dimension=True).to(self.device)
            #self.one_dim_mask = create_mask(self.window_size, displacement, one_dimension=True).to(self.device)
        
        if self.relative_pos_embedding:  # + B
            # relative_distances 每个维度的取值范围: -(window_size-1) ~ +(window_size+1), 一共 2*self.window_size-1 个值
            # relative_distances 后面要作为索引值, 因此 +(window_size-1) 使得所有元素均 > 0
            self.relative_indices = get_relative_distances(self.window_size) + self.window_size - 1
            # 两个维度, 每个维度的长度和 relative_distances 每个维度的长度一样, 即 2*self.window_size-1
            self.pos_embedding = nn.Parameter(torch.randn(2*self.window_size-1, 2*self.window_size-1))
    
    def forward(self, x):
        #print(x.size())
        if self.shifted:
            x = self.cyclic_shift(x)
        
        # =========== to qkv encoding ===========
        x = self.qkv_dconv(x)
        qkv = self.qkv_pconv(x).chunk(3, dim=1) # channel 通道分割成 3 份
        
        # =========== MSA ===========
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) (nw_h w_h) (nw_w w_w) -> b h (nw_h nw_w) (w_h w_w) c',
                                h=self.heads, w_h=self.window_size, w_w=self.window_size), qkv)
        
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.dk
        
        if self.relative_pos_embedding:
            # 索引: relative_indices[:,:,0] 是横坐标, relative_indices[:,:,1] 是纵坐标, 以此类推
            dots += self.pos_embedding[self.relative_indices[:,:,0], self.relative_indices[:,:,1]]
        
        # window_num*window_num 个 token 按照 0,1,2... 的顺序排列, 只有其中一部分需要 mask
        if self.shifted:
            dots[:, :, -self.window_num:] += self.zero_dim_mask  # 最下边的 patch
            dots[:, :, self.window_num - 1::self.window_num] += self.one_dim_mask  # 最右边的 patch
        
        attn = dots.softmax(dim=-1)
        
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) c -> b (h c) (nw_h w_h) (nw_w w_w)',
                        h=self.heads, w_h=self.window_size, nw_h=self.window_num)
        
        if self.shifted:
            out = self.cyclic_back_shift(out)
        
        # =========== to output encoding ===========
        out = self.out_dconv(out)
        out = self.out_pconv(out)
        
        #print(out.size())
        return out


class ChannelMLP(nn.Module):
    """
    针对通道的 MLP, 只对通道进行线性变换, 输入和输出大小一样
    : param dim        : 输入 channel 数
    : param hidden_dim : linear 隐藏层层数
    """
    def __init__(self, dim, hidden_dim, activation):
        super().__init__()
        
        if activation == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            activation = GELU()
        elif activation == "RELU6":
            activation = nn.ReLU6(inplace=True)
        elif activation == "swish":
            activation = Swish()
        elif activation == "hardSwish":
            activation = HardSwish(True)
        elif activation == "mish":
            activation = Mish()
        elif activation == "ELU":
            activation = nn.ELU(inplace=True)
        elif activation != None:
            print(activation + " : there is not this type of activation!")
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.net(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.size())
        return x


if __name__ == "__main__":
    # test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('using the GPU...')
    else:
        print('using the CPU...')
    #device = 'cpu'
    
    heads=4
    feature_num=8
    feature_size=16
    window_size=4
    shifted=False
    relative_pos_embedding=False
    
    model = ChannelMLP(8, 32, 'ReLU')
    #model = Attention(heads, feature_num, feature_size, window_size, shifted, relative_pos_embedding)
    model = model.to(device)
    
    # show param
    print(model)
    
    for param in model.parameters():
        print(param.shape)
        #print(param)
    
    # 输入初始化
    inputs = torch.rand((2, 8, 16, 16))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)
    
