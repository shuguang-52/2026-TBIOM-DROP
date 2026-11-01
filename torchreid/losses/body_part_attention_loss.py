from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from torchreid.utils.constants import PIXELS

class BodyPartAttentionLoss(nn.Module):
    """ A body part attention loss as described in our paper
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    """

    def __init__(self, loss_type='cl', label_smoothing=0.1, with_ss=False, use_gpu=False):
        super().__init__()
        self.pred_accuracy = Accuracy(top_k=1)
        self.loss_type = loss_type
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()
        if loss_type == 'cl':
            self.part_prediction_loss_1 = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError("Loss {} for part prediction is not supported".format(loss_type))

        self.with_ss = with_ss
        self.loss_h = torch.nn.L1Loss(reduction='mean')  # 之前为reduction='mean'
        self.loss_v = torch.nn.L1Loss(reduction='mean')

    def forward(self, pixels_cls_scores, targets):
        """ Compute loss for body part attention prediction.
            Args:
                pixels_cls_scores [N, K, H, W]
                targets [N, H, W]
            Returns:
        """
        loss_summary = {}
        loss_summary[PIXELS] = OrderedDict()
        pixels_cls_loss, pixels_cls_accuracy, pixels_regul = self.compute_pixels_cls_loss(pixels_cls_scores, targets)
        loss_summary[PIXELS]['c'] = pixels_cls_loss
        loss_summary[PIXELS]['a'] = pixels_cls_accuracy
        loss_summary[PIXELS]['r'] = pixels_regul
        return pixels_cls_loss, loss_summary


    def compute_pixels_cls_loss(self, pixels_cls_scores, targets):
        targets = targets.to(pixels_cls_scores.device)
        spatial = 1
        b, k, h, w = pixels_cls_scores.shape
        h_target = pixels_cls_scores.new_zeros(b, k, h-1, w)
        v_target = pixels_cls_scores.new_zeros(b, k, h, w-1)

        contra_h = pixels_cls_scores[:, :, spatial:, :] - pixels_cls_scores[:, :, 0:-spatial, :]
        contra_v = pixels_cls_scores[:, :, :, spatial:] - pixels_cls_scores[:, :, :, 0:-spatial]
        l_h = self.loss_h(contra_h, h_target)
        l_v = self.loss_v(contra_v, v_target)

        pixels_cls_score_targets = targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)  # [N*Hf*Wf, M]

        if self.with_ss:
            loss = self.part_prediction_loss_1(pixels_cls_scores, pixels_cls_score_targets) + 0.35*(l_h + l_v)
        else:
            loss = self.part_prediction_loss_1(pixels_cls_scores, pixels_cls_score_targets)
        accuracy = self.pred_accuracy(pixels_cls_scores, pixels_cls_score_targets)
        return loss, accuracy.item(), (l_h + l_v)
