# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import cv2
import numpy as np
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # combined = to_image_list(combined)

        # input_image = np.ones(shape=[images.tensors.shape[2], images.tensors.shape[3], 3], dtype=np.uint8)
        # newimages = np.swapaxes(np.swapaxes(images.tensors[0,:,:,:].cpu().data.numpy(), 0, 2), 0, 1)
        # for i in range(input_image.shape[2]):
        #     input_image[:,:,i] = newimages[:,:,i]

        features = self.backbone(images.tensors)
        # features_depth = self.backbone1(combined.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        # print(targets[0].bbox)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
            # for j in range(len(result)):
            #     temp = result[j]
            #     for i in range(len(temp)):
            #         cv2.rectangle(input_image, (temp.bbox[i][0],temp.bbox[i][1]), (temp.bbox[i][2], temp.bbox[i][3]), (0, 0, 255), 2)
            # # temp1 = targets[0].bbox
            # # for i in range(len(temp1)):
            # #     cv2.rectangle(input_image, (temp1[i][0], temp1[i][1]), (temp1[i][2], temp1[i][3]), (0, 255, 0), 2)
            #
            #
            # cv2.imshow('image', input_image)
            # cv2.waitKey(10)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
