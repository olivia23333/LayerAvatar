import torch
import torch.nn as nn
import numpy as np
from mmgen.models.builder import MODULES
import lpips
import cv2


@MODULES.register_module()
class PerLoss(nn.Module):

    def __init__(self, loss_weight=1.0, height=1024, width=1024):
        super().__init__()
        self.loss_vgg = torch.jit.load('work_dirs/cache/vgg16.pt').eval()
        # self.loss_vgg = lpips.LPIPS(net='vgg')
        # for param in self.loss_vgg.parameters():
        #     param.requires_grad = False
        self.loss_weight = loss_weight
        self.height, self.width = height, width

    def forward(self, pred, target):
        # select_idx = np.random.choice(num_imgs, 1, replace=False)
        rand_start = torch.randint(256, (1,))
        # rand_start_h = ((torch.randn(1) * 256 / 3) + 256).int().clip(0, 512)
        rand_start_h = ((torch.randn(1) * 119 / 3) + 119).int().clip(0, 238)
        if pred.shape[-2] > 512:
            pred_imgs = pred.permute(0, 3, 1, 2).clamp(min=0, max=1)[:, :, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
            target_imgs = target.permute(0, 3, 1, 2).clamp(min=0, max=1)[:, :, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
            # pred_imgs = pred.permute(0, 1, 4, 2, 3).clamp(min=0, max=1)[:, :, :3, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
            # target_imgs = target.permute(0, 1, 4, 2, 3).clamp(min=0, max=1)[:, :, :3, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
        else:
            pred_imgs = pred.permute(0, 3, 1, 2).clamp(min=0, max=1)
            target_imgs = target.permute(0, 3, 1, 2).clamp(min=0, max=1)
            # pred_imgs = pred.permute(0, 1, 4, 2, 3).clamp(min=0, max=1)
            # target_imgs = target.permute(0, 1, 4, 2, 3).clamp(min=0, max=1)
        pred_imgs = pred_imgs.reshape(-1, 3, 512, 512)
        # pred_imgs = pred_imgs[:, :3] * 0.6 + pred_imgs[:, 3:] * 0.4
        # cv2.imwrite('/mnt/sdb/zwt/SSDNeRF/debug.jpg', cv2.cvtColor((pred_imgs[0]*255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        target_imgs = target_imgs.reshape(-1, 3, 512, 512)

        # target_masks = target_rgba[:, [3]].bool().repeat(1, 3, 1, 1)
        # target_imgs = target_rgba[:, :3]
        # pred_imgs = pred_imgs * target_masks
        # pred_imgs[~target_masks] = 0.5
        # target_imgs = target_imgs * target_masks
        # target_imgs[~target_masks] = 0.5

        # target_imgs = target_imgs[:, :3] * 0.6 + target_imgs[:, 3:] * 0.4
        # cv2.imwrite('/mnt/sdb/zwt/SSDNeRF/debug_real.jpg', cv2.cvtColor((target_imgs[0]*255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        # assert False
        # pred_imgs = pred_imgs * 2. - 1.
        # target_imgs = target_imgs * 2. - 1.
        pred_imgs = pred_imgs * 255
        target_imgs = target_imgs * 255
        pred_feat = self.loss_vgg(pred_imgs, resize_images=False, return_lpips=True)
        target_feat = self.loss_vgg(target_imgs, resize_images=False, return_lpips=True)
        # pred_imgs = pred_imgs[:, :, :256, :256]
        # target_imgs = target_imgs[:, :, :256, :256]
        # print(self.loss_vgg(pred_imgs*2-1, target_imgs*2-1).shape)
        # assert False
        dist = (target_feat - pred_feat).square().sum()
        dist_weighted = dist * self.loss_weight
        return dist_weighted
        # return self.loss_vgg(pred_imgs, target_imgs).mean() * self.loss_weight