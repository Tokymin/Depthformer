# -*- coding:utf-8 -*-
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from torch.autograd.grad_mode import F


def normalization(data, data_max, data_min):
    """
    根据 max value 和 min value 对 data 进行归一化;
    若 data 可以是二维矩阵 [m,n], data max / min 为 [m,1]
    """
    # 归一化至 -1 ~ +1
    return (data - data_min) / (data_max - data_min)


def inverse_normalization(data, data_max, data_min):
    # 反归一化
    return ((data * (data_max - data_min)) + data_min)


def GenerateRMSE(joint_pre, joint_real):
    return np.sqrt(np.mean(np.square(joint_pre - joint_real), axis=0))


def GenerateR_2(joint_pre, joint_real):
    # return r2_score(joint_pre, joint_real)
    return r2_score(joint_real, joint_pre)


def GenerateCC(joint_pre, joint_real):
    cc = np.corrcoef(joint_pre, joint_real)
    return cc[0, 1]


def evaluate(pre_glo, grt_glo):
    CC_ls, RMSE_ls, R2_ls = [], [], []
    # -------- 评价回归 --------
    for i in range(pre_glo.shape[0]):
        CC = GenerateCC(pre_glo[i], grt_glo[i])
        RMSE = GenerateRMSE(pre_glo[i], grt_glo[i])
        R2 = GenerateR_2(pre_glo[i], grt_glo[i])

        CC_ls.append(CC)
        RMSE_ls.append(RMSE)
        R2_ls.append(R2)

    return CC_ls, RMSE_ls, R2_ls


def evaluate_new(pre_glo, grt_glo):
    CC_ls = []
    # -------- 评价回归 --------
    for i in range(pre_glo.shape[0]):
        CC = GenerateCC(pre_glo[i], grt_glo[i])
        CC_ls.append(CC)

    RMSE = GenerateRMSE(pre_glo, grt_glo)
    R2 = GenerateR_2(pre_glo, grt_glo)
    return np.mean(np.array(CC_ls)), np.mean(RMSE), R2


# -*- coding:utf-8 -*-


def addPath():
    """ 添加路径至系统路径 """
    import os, sys
    this_dir = os.path.dirname(os.path.abspath(__file__))  # 获得此程序地址, 即 tools 地址
    this_dir = os.path.dirname(this_dir)  # 获取项目地址
    sys.path.append(this_dir)
    return this_dir
    # print(sys.path)


def show_parameters_num(model):
    """ 神经网络的参数总量 """
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print('Number of params(million): %.2f million.' % (total / 1e6))  # 每一百万为一个单位
    print('Number of params(number): ' + str(total))


def show_parameters_size(model):
    """ 神经网络每一层的参数大小 """
    for param in model.parameters():
        print(param.shape)
        # print(param)


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def normalize_image_numpy(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = x.max()
    mi = x.min()
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def visualize_depth(dis, cmap=cv2.COLORMAP_HOT):
    """
    depth: (H, W)
    """
    x = dis.detach().cpu().numpy().reshape(dis.shape[1], dis.shape[2])
    # x = np.nan_to_num(x)  # change nan to 0

    # x = 5.4 / (x + 1e-8)
    # x = np.nan_to_num(x)  # change nan to 0
    # depth = np.clip(depth, 0, 1)  # 从80改为了10
    # depth = np.uint16(depth * 256)

    # mi = np.min(x)  # get minimum depth
    # ma = np.max(x)
    # x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint16)  # 255 *
    x_ = Image.fromarray(cv2.applyColorMap(cv2.convertScaleAbs(x), cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def visualize_to_gray(dis, cmap=cv2.COLORMAP_HOT):
    """
    depth: (H, W)
    """
    x = dis.detach().cpu().numpy()
    x_ = Image.fromarray(x).convert('L')
    return x_
def visualize_depth_PIL(dis, cmap=cv2.COLORMAP_HOT):
    """
    depth: (H, W)
    """
    x = dis.detach().cpu().numpy().reshape(dis.shape[1], dis.shape[2])
    x = (255 * x).astype(np.uint16)  # 255 *
    x_ = Image.fromarray(cv2.applyColorMap(cv2.convertScaleAbs(x), cmap))
    return x_
def visualize_depth_opencv(dis, cmap=cv2.COLORMAP_MAGMA):
    """
    depth: (H, W)
    """
    x = dis.detach().cpu().numpy().reshape(dis.shape[2], dis.shape[2])
    # x = np.nan_to_num(x)  # change nan to 0

    # x = 5.4 / (x + 1e-8)
    # x = np.nan_to_num(x)  # change nan to 0
    # depth = np.clip(depth, 0, 1)  # 从80改为了10
    # depth = np.uint16(depth * 256)

    # mi = np.min(x)  # get minimum depth
    # ma = np.max(x)
    # x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint16)  # 255 *
    x_ = cv2.applyColorMap(cv2.convertScaleAbs(x), cmap)

    return x_

@torch.no_grad()
def compute_errors(gt, pred, tgt_img):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()
    
    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='bilinear', align_corners=False)
    pred = pred.view(batch_size, h, w)

    max_depth = 1
    min_depth = 0.1
    a = torch.where(tgt_img > 0, tgt_img, 0 * tgt_img)[:, 0, :, :]
    valid_mask = torch.logical_and(gt <= max_depth,
                                   a > 0)  # 过滤掉那些0,0,1的像素点)
    valid = (gt > min_depth) & (gt < max_depth)
    valid = valid & valid_mask
    valid_gt = gt[valid]
    valid_pred = pred[valid]
    # align scale
    valid_pred = valid_pred * \
                 torch.median(valid_gt) / torch.median(valid_pred)
    valid_pred = valid_pred.clamp(min_depth, max_depth)

    thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
    a1 += (thresh < 1.25).float().mean()
    a2 += (thresh < 1.25 ** 2).float().mean()
    a3 += (thresh < 1.25 ** 3).float().mean()

    diff_i = valid_gt - valid_pred
    abs_diff += torch.mean(torch.abs(diff_i))
    abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
    sq_rel += torch.mean(((diff_i) ** 2) / valid_gt)
    rmse += torch.sqrt(torch.mean(diff_i ** 2))
    rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                                       torch.log(valid_pred)) ** 2))
    log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                                   torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]]
