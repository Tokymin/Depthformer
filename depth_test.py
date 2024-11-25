# -*- coding:utf-8 -*-
import csv
import datetime
import shutil
import pandas as pd
import torchvision
import time
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# 设定项目地址
from Models.BaseNN.Models.DepthNet import DepthNet
from Models.BaseNN.Models.Depthformer.My_Swin_DP import My_DepthFormer
from dataset.test_folders import TestFolder
from loss.sigloss import SigLoss
from utils.get_name import get_name
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
# 加载配置参数
from config.config_test import hyperparameters


class DEPTHNN(BASENN):
    def __init__(self, args):
        super(DEPTHNN, self).__init__(args)
    def create_head(self):
        pass

    def head_forward(self, x):
        x = torch.sigmoid(x)
        return x

def write_metric_to_excel(excel_path, data, model_prefix):  # 新增
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name=model_prefix)
    writer.save()

def save_depth_tensor_to_png(depth_tensor, file_path):  # 新增
    depth_array = depth_tensor.squeeze().cpu().detach().numpy()
    plt.imsave(file_path, depth_array, cmap='gray')

# 测试函数
def test(model_index, model_prefix, load_depth_gt, nums_limit,is_save_depth):
    all_errs = []
    model_depth.eval()
    writer = writers["test"]
    for index, (tgt_img, tgt_depth, img_name) in enumerate(test_loader):
        if index >= nums_limit:
            break
        tgt_img = tgt_img.to(device)
        outputs = model_depth(tgt_img)
        # 计算误差
        if load_depth_gt:
            errs = compute_errors(tgt_depth.squeeze(0).cuda(), normalize_image(outputs[0, :, :, :]), tgt_img)
            errs_names = ["abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"]
            all_errs.append(np.array(errs))
            for (i, errs_name) in zip(range(0, 9), errs_names):
                writer.add_scalar(errs_name, errs[i], index)
        writer.add_image("img" + str(model_index), tgt_img[0, :, :, :], index)
        writer.add_image("predic_depth" + str(model_index), visualize_depth(normalize_image(outputs[0, :, :, :])), index)
        root = "Saved_depth_data"
        predict_depth = os.path.join(root, model_prefix, f"depth_{img_name[0]}")
        if not os.path.exists(os.path.join(root, model_prefix)):
            os.makedirs(os.path.join(root, model_prefix))
        if is_save_depth:
            save_depth_tensor_to_png(outputs[0], predict_depth)
        print('Test model index: {}\t Test Schedule: [{}/{} ({:.0f}%)]'.format(model_index, index, len(test_loader), 100. * index / len(test_loader)))
        if index > nums_limit:
            break
    if load_depth_gt == True:  # 当加载了深度图的GT时才计算误差
        all_errs_values = np.array(all_errs)
        errs_names = ["abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"]
        metrics = {errs_name: np.mean(all_errs_values[:, index]) for index, errs_name in enumerate(errs_names)}
        metrics['model_prefix'] = model_prefix
        write_metric_to_excel(root + "/" + model_prefix + "/metrics.xlsx", [metrics], model_prefix)

# 加载数据
def load_data(args):
    batch_size = args['batch_size']
    test_dataset = TestFolder(args['dataset_dir'], transform=[torchvision.transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=args['num_work'], batch_size=batch_size, pin_memory=True, drop_last=True)
    return test_loader


if __name__ == "__main__":
    args = hyperparameters
    device = 'cuda' if torch.cuda.is_available() and args['CUDA'] else 'cpu'
    if args['network'] == 'ResNet':
        model_depth = DepthNet().to(device) # 使用DepthNet网络 ResnetEncoder+DepthDecoder
    elif args['network'] == 'Depthformer':
        model_depth = My_DepthFormer().cuda() # 使用修改后的DepthFormer网络
    else:
        print("请指定模型类型！！")
    test_loader = load_data(args)  # 加载数据
    for model_index in range(42, 44, 2):
        if args['load_ckpt']:
            print('load model from %s ...' % args['load_ckpt'])
            model_depth.load_state_dict(torch.load(os.path.join(args['load_ckpt'], str(model_index) + '.pth')))
            print('load weight success!')
        else:
            print("please input model path to test")
            exit(0)
        model_prefix=args['model_prefix']
        load_depth_gt=args['load_depth_gt']
        writers = {}
        writers["test"] = SummaryWriter(os.path.join("logs/Saved_depth_data", args['model_prefix']))
        test(model_index, model_prefix, load_depth_gt, args['nums_limit'],args['is_save_depth'])