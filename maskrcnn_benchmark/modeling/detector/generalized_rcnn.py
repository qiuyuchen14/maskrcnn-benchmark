# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone

from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import cv2
import numpy as np
from PIL import Image


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


        if cfg.MODEL.DEPTH_ON and cfg.MODEL.RGB_ON:
            self.rpn = build_rpn(cfg, 2*self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, 2*self.backbone.out_channels)
        else:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.depth_on = cfg.MODEL.DEPTH_ON
        self.rgb_on = cfg.MODEL.RGB_ON
        self.coarse_on = cfg.MODEL.COARSE_ON
        self.dcoarse_on = cfg.MODEL.DCOARSE_ON
        self.cfg = cfg



    def forward(self, images, depths=None, targets=None):
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
        features = None
        images_coarse = None
        depth_feature = None
        depth_coarse = None

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")


        if self.rgb_on:
            images = to_image_list(images)
            if self.coarse_on:
                images_coarse = F.interpolate(images.tensors, size=(128, 256))

            rgb_features = self.backbone(images.tensors)
            features = rgb_features

        if self.depth_on:
            depths = to_image_list(depths)
            if self.dcoarse_on:
                depth_coarse = F.interpolate(images.tensors, size=(128, 256))

            depth_feature = self.backbone(depths.tensors)
            features = depth_feature
            # proposals, proposal_losses = self.rpn(depths, depth_feature, targets)

        if self.rgb_on and self.depth_on:
            rgbd_features = []
            for i in range(len(features)):
                rgbd_features.append(torch.cat((rgb_features[i], depth_feature[i]), dim=1))

            rgbd_features = tuple(rgbd_features)
            features = rgbd_features
            images = to_image_list(torch.cat((images.tensors, depths.tensors), dim=1))
            images.image_sizes = depths.image_sizes


        proposals, proposal_losses = self.rpn(images, features, targets)


        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, images_coarse, depth_feature, depth_coarse, proposals, targets)

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
