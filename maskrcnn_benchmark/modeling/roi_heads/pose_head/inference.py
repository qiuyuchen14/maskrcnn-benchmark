# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate
import torch.nn.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO check if want to return a single BoxList or a composite
# object
class PosePostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self):
        super(PosePostProcessor, self).__init__()

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        # pose_prob = F.softmax(x, -1)
        pose_prob = x
        # select masks coresponding to the predicted classes
        num_poses = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_poses, device=labels.device)

        # mask = torch.ones(pose_prob.size())
        # mask[:, 3] = 0
        # pose_prob = pose_prob * mask.float()

        pose_prob = pose_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        pose_prob = pose_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, box in zip(pose_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("pose", prob)
            results.append(bbox)

        return results


def make_roi_pose_post_processor():
    pose_post_processor = PosePostProcessor()
    return pose_post_processor

























