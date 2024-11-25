#-*- coding:utf-8 -*-


def addPath():
    """ 添加路径至系统路径 """
    import os, sys
    this_dir = os.path.dirname(os.path.abspath(__file__)) # 获得此程序地址, 即 tools 地址
    this_dir = os.path.dirname(this_dir) # 获取项目地址
    sys.path.append(this_dir)
    return this_dir
    #print(sys.path)


def show_parameters_num(model):
    """ 神经网络的参数总量 """
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print('Number of params(million): %.2f million.' % (total / 1e6))  # 每一百万为一个单位
    print('Number of params(number): ' + str(total))


def show_parameters_size(model):
    """ 神经网络每一层的参数大小 """
    for param in model.parameters():
        print(param.shape)
        #print(param)

