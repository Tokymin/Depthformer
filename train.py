# -*- coding:utf-8 -*-
import csv
import datetime
import shutil

import torchvision
from path import Path
import time
import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 设定项目地址
from Models.BaseNN.Models.DepthNet import DepthNet
from Models.BaseNN.Models.Depthformer.My_Swin_DP import My_DepthFormer
from loss.sigloss import SigLoss

this_dir = os.path.dirname(os.path.abspath(__file__))
this_dir = os.path.dirname(this_dir)
sys.path.append(this_dir)
basic_path = this_dir

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

# 数据加载
from dataset.train_folders import TrainFolder

# 损失函数
from Models.BaseNN.Models.models import BASENN
from utils.util import *
from tensorboardX import SummaryWriter


def arg_parse():
    parser = argparse.ArgumentParser(description='SEMG: Main transformer')

    # for ablation study
    parser.add_argument('--no_ssim', action='store_true',
                        help='use ssim in photometric loss')
    parser.add_argument('--no_auto_mask', action='store_true',
                        help='masking invalid static points')
    parser.add_argument('--no_dynamic_mask',
                        action='store_true', help='masking dynamic regions')
    parser.add_argument('--no_min_optimize', action='store_true',
                        help='optimize the minimum loss')

    # 数据集加载
    parser.add_argument('--dataset_dir', default="/home/toky/Datasets/Endo_colon_unity/train_dataset", type=str)
    parser.add_argument('--dataset_name', type=str,
                        default='kitti', choices=['kitti', 'nyu', 'ddad'])
    parser.add_argument('--sequence_length', type=int,
                        default=3, help='number of images for training')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='jump sampling from video')
    # 神经网络结构设置
    parser.add_argument('--A_N', default=False, type=bool)  # 是否 激活函数(activation) 在 归一化(normalization) 前面

    # 训练设置

    parser.add_argument('--num_work', default=0, type=int)  # data loader num work
    parser.add_argument('--batch_size', default=256, type=int)  # batch size
    parser.add_argument('--subdivisions', default=32, type=int)  # 当显存不够时, 把 batch size / subdivisions 后分批运算, 统一反向传播 Resnet: 2
    parser.add_argument('--max_episode', default=3000, type=int)  # 重复训练多少次
    # 训练参数

    parser.add_argument('--photo_weight', type=float,
                        default=1.0, help='photometric loss weight')
    parser.add_argument('--geometry_weight', type=float,
                        default=0.5, help='geometry loss weight')
    parser.add_argument('--smooth_weight', type=float,
                        default=0.1, help='smoothness loss weight')
    # 学习率
    parser.add_argument('--learning_rate', default=1e-4, type=float)  # init learning rate

    # 动态学习率
    parser.add_argument('--patience', default=250, type=int)  # 连续 patience 次迭代 loss 仍没有明显下降时降低学习率
    parser.add_argument('--show_info', default=True, type=bool)  # 自动改变学习率时是否提示信息

    # 加载计算好的权重
    parser.add_argument('--load_depth', default=None, type=bool)  # 使用 True 载入之前的权重并在上一次训练的基础上继续训练
    parser.add_argument('--weight_depth', default=3000, type=int)  # 如果使用之前的权重, 权重路径
    parser.add_argument('--load_pose', default=None, type=bool)  # 使用 True 载入之前的权重并在上一次训练的基础上继续训练
    parser.add_argument('--weight_pose', default=3000, type=int)  # 如果使用之前的权重, 权重路径

    # 测试设置
    parser.add_argument('--test_batch_size', default=1, type=int)  # test batch size

    # 输出信息设置
    parser.add_argument('--print_log', default=10, type=int)  # 每隔 print_log 在终端上输出一次
    parser.add_argument('--save_interval_depth', default=500, type=int)  # 每隔 save_interval 保存一次权重
    parser.add_argument('--save_interval_pose', default=500, type=int)  # 每隔 save_interval 保存一次权重

    # 参数文件路径设置
    parser.add_argument('--cfg_file_depth', default=r"cfg/depthnet_v3.cfg", type=str)  # network cfg 文件路径
    parser.add_argument('--save_weight_depth', default=r"backup/depthnet_v1", type=str)  # 保存权重的路径
    parser.add_argument('--cfg_file_pose', default=r"cfg/posenet_v2.cfg", type=str)  # network cfg 文件路径
    # 数据文件路径设置
    parser.add_argument('--train_file', default=r"data/train/train.txt", type=str)  # train txt 路径
    parser.add_argument('--test_file', default=r"data/test/test.txt", type=str)  # test txt 路径

    # 可视化保存
    parser.add_argument('--show_path', default=r"backup/show", type=str)  # 可视化储存路径

    return parser.parse_args()


class DEPTHNN(BASENN):
    def __init__(self, args):
        super(DEPTHNN, self).__init__(args)

    def create_head(self):
        pass

    def head_forward(self, x):
        x = torch.sigmoid(x)
        return x


def save_model(save_path, model_state, file_prefixes, epoch, filename='big_save_checkpoint.pth.tar'):
    # 保存模型
    torch.save(model_state, save_path / '{}_{}_{}_{}'.format(file_prefixes, file_prefixes, epoch, filename))


def write_metric_to_csv(metric_csv_path, data, test_dataset_index):
    csv_file = open(metric_csv_path, "a+", encoding='utf-8', newline='')  # a+追加写入
    csv_writer = csv.writer(csv_file)
    metrc_data_list = [test_dataset_index]
    for item in data:
        metrc_data_list.append(item)
    csv_writer.writerow(metrc_data_list)
    csv_file.close()


def train():
    model_depth.train()
    subdivisions = int(args.subdivisions)
    loss_show = 0
    writer = writers["train"]
    for index, (tgt_img, tgt_depth) in enumerate(train_loader):
        tgt_img = tgt_img.to(device)

        # input = torch.rand((4, 3, 320, 320)).to(device)
        outputs = model_depth(tgt_img)
        loss_temp = siglos(outputs, tgt_depth.to(device), tgt_img)
        loss = loss_temp / subdivisions
        loss.backward()

        loss_show += loss.item()

        if (index + 1) % subdivisions == 0 or (index + 1) == len(train_loader):
            optimizer_depth.step()
            optimizer_depth.zero_grad()
            scheduler_depth.step(loss)
            writer.add_image(
                "tgt_depth",
                visualize_depth(normalize_image(tgt_depth[0, :, :, :])), index)
            writer.add_image(
                "predic_depth",
                visualize_depth(normalize_image(outputs[0, :, :, :])), index)
            writer.add_image(
                "img",
                tgt_img[0, :, :, :], index)
            writer.add_scalar("loss", loss_show, index + (epoch * len(train_loader)))
            # 输出信息
            print(
                'Train epoch: {}\tTraining Schedule: [{}/{} ({:.0f}%)]\n train/total_loss: {:.6f}, lr: {:.8f}'.format(
                    epoch, index, (len(train_loader)), 100. * index / (len(train_loader)),
                    loss_show, optimizer_depth.state_dict()['param_groups'][0]['lr']))
            write_metric_to_csv(
                big_save_path + "/train_loss_" + model_prefix + ".csv",
                [loss_show], epoch)
            loss_show = 0


class proccess_loss(nn.Module):
    def __init__(self):
        super(proccess_loss, self).__init__()

    def forward(self, glove_outputs, glove_label):
        assert glove_outputs.size()[-1] == 3
        loss_glove = F.mse_loss(input=glove_outputs, target=glove_label, reduction='sum')
        return loss_glove


def load_data(args):
    # 加载数据
    batch = int(args.batch_size)
    subdivisions = int(args.subdivisions)
    batch_size = batch // subdivisions
    # 加载数据

    train_dataset = TrainFolder(
        args.dataset_dir,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               num_workers=7,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               drop_last=True)
    return train_loader


if __name__ == "__main__":
    args = arg_parse()
    model_prefix = "CDPDepthformer_only_sigloss"  # 模型的前缀标注
    num_work = int(args.num_work)
    batch = int(args.batch_size)
    subdivisions = int(args.subdivisions)
    batch_size = batch // subdivisions
    max_episode = int(args.max_episode)
    learning_rate = float(args.learning_rate)
    patience = int(args.patience)
    patience = int(args.patience)
    show_info = bool(args.show_info)

    print_log = int(args.print_log)
    save_interval_depth = int(args.save_interval_depth)
    save_interval_pose = int(args.save_interval_pose)

    train_file = str(args.train_file)
    test_file = str(args.test_file)
    show_path = str(args.show_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('using the GPU...')
    else:
        print('using the CPU...')
    # model_depth = DEPTHNN(args.cfg_file_depth).to(device) # 使用自己创建的attention网络
    # model_depth = DepthNet().cuda()  # 使用ResNet网络
    model_depth = My_DepthFormer().cuda()  # 使用修改后的DepthFormer网络
    timestamp = datetime.datetime.now().strftime("%m-%d-%H")
    big_save_path = Path("/home/toky/Projects/Depth_Prediction/ckpt/" + model_prefix)
    print("-" * 80 + "\n  save model to path {} every 100 epoch".format(big_save_path))
    big_save_path.makedirs_p()  # 若目录不存在就创建

    train_loader = load_data(args)  # 加载数据

    # 模型初始化
    print(args)
    if not (args.load_depth is None):
        print('load model from %s ...' % args.load_depth)
        model_depth = torch.load(args.load_depth)['state_dict']
        print('load weight success!')
        # loss
        siglos = SigLoss()
        # ===== show info =====
        model_depth.show_net_info()
        show_parameters_num(model_depth)
        show_parameters_size(model_depth)

    optimizer_depth = torch.optim.Adam(model_depth.parameters(), lr=learning_rate)
    scheduler_depth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_depth, 'min', patience=patience,
                                                                 verbose=show_info)

    small_save_path = '/home/toky/Projects/Depth_Prediction/ckpt/' + model_prefix
    if os.path.exists(small_save_path) == False:
        os.mkdir(small_save_path)
    siglos = SigLoss()
    # writer
    writers = {}
    writers["train"] = SummaryWriter(os.path.join("/home/toky/Projects/Depth_Prediction/logs", model_prefix))

    for epoch in range(0, args.max_episode + 1):
        # 训练
        train()
        if epoch % 2 == 0 and epoch != 0:
            torch.save(model_depth.state_dict(), small_save_path + "/" + str(epoch) + '.pth')
        # 验证 ,暂时不进行验证
        if epoch % 100 == 0 and epoch != 0:
            save_model(
                big_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': model_depth
                }, model_prefix + 'pose', epoch + 1)
