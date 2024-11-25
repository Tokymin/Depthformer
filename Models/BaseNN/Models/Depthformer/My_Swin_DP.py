from torch import nn

from .backbones import DepthFormerSwin
from .decode_heads import DenseDepthHead
from .necks import HAHIHeteroNeck


class My_DepthFormer(nn.Module):
    def __init__(self):
        super(My_DepthFormer, self).__init__()
        self.model_encode_depth_swin_net = DepthFormerSwin()  # Swintransformer提取的特征四层
        self.model_neck = HAHIHeteroNeck(positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=64),
            in_channels=[64, 96, 192, 384, 768],
            out_channels=[64, 96, 192, 384, 768],
            embedding_dim=64,
            scales=[1, 1, 1, 1, 1])
        self.model_decode_depth_net = DenseDepthHead(
            in_channels=[64, 96, 192, 384, 768],
            up_sample_channels=[64, 96, 192, 384, 768],
            channels=64,
            align_corners=True,
            loss_decode=dict(type='SigLoss', valid_mask=True),
            act_cfg=dict(type='LeakyReLU', inplace=True),
            min_depth=0.0001,
            max_depth=1)

    def inference_depth(self, img):
        # 原始模型
        encode_feature_list_F = self.model_encode_depth_swin_net(img)
        temp = self.model_neck(encode_feature_list_F)
        pred_depth = self.model_decode_depth_net(temp)
        return pred_depth

    def forward(self, tgt_img):
        tgt_depth = self.inference_depth(tgt_img)
        return tgt_depth
