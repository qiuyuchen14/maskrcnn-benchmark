# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch

@registry.ROI_POSE_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pose_predictor = nn.Linear(num_inputs, num_classes)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        poses = self.pose_predictor(x)
        return poses


@registry.ROI_POSE_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_bins = cfg.MODEL.ROI_POSE_HEAD.NUM_POSE_BINS
        representation_size = in_channels


        self.pose_predictor = nn.Linear(representation_size, num_classes)
        # self.pose_cls = nn.Linear(num_classes, num_bins)

        nn.init.normal_(self.pose_predictor.weight, std=0.01)
        nn.init.constant_(self.pose_predictor.bias, 0)

        # nn.init.normal_(self.pose_cls.weight, std=0.01)
        # nn.init.constant_(self.pose_cls.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        poses = self.pose_predictor(x)
        poses = torch.sigmoid(poses)
        return poses

def make_roi_pose_predictor(cfg, in_channels):
    func = registry.ROI_POSE_PREDICTOR[cfg.MODEL.ROI_POSE_HEAD.PREDICTOR]
    return func(cfg, in_channels)
