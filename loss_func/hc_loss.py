"""
计算预测图像的头围与GT的头围的差值
"""
import numpy as np
from utils.FitEllipse import *
from torchvision import transforms as T
from PIL import Image
import torch


def hc_loss(pre, pix, hc, device):
    transform = [T.Resize((540, 800), interpolation=Image.NEAREST)]    # 转回原始图像大小
    transform = T.Compose(transform)
    bz = pre.size()[0]  # 获取batch_size
    hc_pre = torch.tensor(np.zeros([bz, ]))
    for i in range(bz):
        pre_every = pre.data.cpu().numpy()[i, :, :, :].squeeze().astype(np.uint8)  # 获取单独一张图像
        pre_every_i = Image.fromarray(pre_every)
        pre_every_a = np.array(transform(pre_every_i))
        # gt_every = gt.data.cpu().numpy()[0, :, :, :].squeeze().astype(np.uint8)  # 获取单独一张图像
        if C_elipse(np.array(pre_every_a)) == 0:
            loss_hc = torch.tensor(0, device=device)
            return loss_hc
        hc_pre[i] = torch.tensor(C_elipse(np.array(pre_every_a)) * pix[i] - pix[i]*5.4)    # 后面的是矫正值,矫正后GT结果才与文档基本一致
    loss_hc = torch.mean((torch.abs(hc_pre-hc)**2))       # MSE_loss
    return loss_hc
