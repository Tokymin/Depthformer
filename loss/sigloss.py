# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from Models.BaseNN.Models.Depthformer.builder import LOSSES

device = 'cuda'


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)

    if mask.sum() > 100:
        # print("mask.sum() > 100")
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        print("mask.sum() < 100")
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 max_depth=1,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask

        self.max_depth = max_depth
        self.eps = 0.1  # avoid grad explode
        self.valid_mask_data = None
        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

        self.loss_weight = 0.5
        self.diff_loss_weight = 0.5
        self.smooth_loss_weight = 0.3

    def sigloss(self, input, target, tgt_img):
        if self.valid_mask:
            valid_mask = target > 0
            a = torch.where(tgt_img > 0, tgt_img, 0 * tgt_img)[:, 0, :, :].unsqueeze(1)
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, a > 0)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def compute_smooth_loss(self, disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        img_mask = torch.where(img > 0, True, False)
        disp_mask = torch.logical_and(disp > 0, img_mask[:, 0, :, :].unsqueeze(1) > 0)
        # img = img * img_mask
        disp = disp * disp_mask
        # mean_disp = disp.mean(2, True).mean(3, True)
        # norm_disp = disp / (mean_disp + 1e-7)
        # disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(grad_img_x)
        grad_disp_y *= torch.exp(grad_img_y)
        loss = 1 - (grad_disp_x.mean() + grad_disp_y.mean())
        return loss

    def compute_diff_loss(self, computed_depth, gt_depth):
        diff_depth = (computed_depth - gt_depth).abs() / \
                     (computed_depth + gt_depth)
        diff_loss = mean_on_mask(diff_depth, self.valid_mask_data)
        return diff_loss

    def forward(self,
                depth_pred,
                depth_gt, tgt_img):
        """Forward function."""
        a = torch.where(tgt_img > 0, tgt_img, 0 * tgt_img)[:, 0, :, :].unsqueeze(1)
        valid_mask = torch.logical_and(depth_gt <= self.max_depth,
                                       a > 0)  # 过滤掉那些0,0,1的像素点)
        self.valid_mask_data = valid_mask
        # loss_depth = self.loss_weight * self.sigloss(depth_pred,
        #                                              depth_gt, tgt_img)
        # smooth_loss = self.smooth_loss_weight * self.compute_smooth_loss(
        #     depth_pred, tgt_img)
        #
        diff_loss = self.diff_loss_weight * self.compute_diff_loss(depth_pred, depth_gt)
        return diff_loss  # smooth_loss + loss_depth +
