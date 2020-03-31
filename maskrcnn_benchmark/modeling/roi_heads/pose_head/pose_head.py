# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_pose_feature_extractors import make_roi_pose_feature_extractor
from .roi_pose_predictors import make_roi_pose_predictor
from .inference import make_roi_pose_post_processor
from .loss import make_roi_pose_loss_evaluator
import numpy as np
def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIPoseHead(torch.nn.Module):
    """
     Generic Box Head class.
     """

    def __init__(self, cfg, in_channels):
        super(ROIPoseHead, self).__init__()
        self.feature_extractor = make_roi_pose_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_pose_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_pose_post_processor()
        self.loss_evaluator = make_roi_pose_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # # change proposals to 4 proposals
        # outputs = []
        # proposals1 = proposals.copy()
        # proposals2 = proposals.copy()
        # proposals3 = proposals.copy()
        # proposals4 = proposals.copy()

        # for j in range(len(proposals)):
        #     w1 = proposals[j].bbox[:,2]-proposals[j].bbox[:,0]
        #     h1 = proposals[j].bbox[:,3]-proposals[j].bbox[:,1]
        #     cx = 0.5*(proposals[j].bbox[:,0] + proposals[j].bbox[:,2])
        #     cy = 0.5*(proposals[j].bbox[:,1] + proposals[j].bbox[:,3])
        #     # proposals1[j] = torch.tensor([0.5, 2, 0.5, 2]).to("cuda")*(proposals[j].bbox)+torch.cat((cx.unsqueeze(-1), torch.cat((-cy.unsqueeze(-1), torch.cat((cx.unsqueeze(-1), -cy.unsqueeze(-1)), 1)), 1)), 1)
        #     # proposals2[j] = torch.tensor([2.0, 2.0, 2.0, 2.0]).to("cuda")*(proposals[j].bbox).to("cuda")+torch.cat((-cx.unsqueeze(-1), torch.cat((-cy.unsqueeze(-1), torch.cat((-cx.unsqueeze(-1), -cy.unsqueeze(-1)), 1)), 1)), 1).to("cuda")
        #     # proposals3[j] = torch.tensor([2, 0.5, 2, 0.5]).to("cuda")*(proposals[j].bbox)+torch.cat((cx.unsqueeze(-1), torch.cat((-cy.unsqueeze(-1), torch.cat((cx.unsqueeze(-1), -cy.unsqueeze(-1)), 1)), 1)), 1)
        #     # proposals4[j] = proposals[j]
        #     proposals2[j] = torch.tensor([-0.3, -0.3, 0.3, 0.3]).to("cuda")*torch.cat((w1.view(w1.shape[0], 1), torch.cat((h1.view(h1.shape[0], 1), torch.cat((w1.view(w1.shape[0], 1), h1.view(h1.shape[0], 1)), 1)), 1)), 1)+(proposals[j].bbox)
        #     proposals2[j][proposals2[j]<0]=0
        #     proposals2[j][proposals2[j][:, 0] > proposals[j].size[0]] = proposals[j].size[0]
        #     proposals2[j][proposals2[j][:, 2] > proposals[j].size[0]] = proposals[j].size[0]
        #     proposals2[j][proposals2[j][:, 1] > proposals[j].size[1]] = proposals[j].size[1]
        #     proposals2[j][proposals2[j][:, 3] > proposals[j].size[1]] = proposals[j].size[1]
        #     # blank_image = np.ones(shape=[proposals[j].size[0], proposals[j].size[0], 3], dtype=np.uint8)
        #     # import cv2
        #     #
        #     # # for j in range(len(proposals)):
        #     # temp = proposals2[j][0]
        #     # temp1 = proposals[j].bbox[0]
        #     # for i in range(len(temp)):
        #     #     cv2.rectangle(blank_image, (temp[0],temp[1]), (temp[2], temp[3]), (0, 0, 255), 2)
        #     #     cv2.rectangle(blank_image, (temp1[0], temp1[1]), (temp1[2], temp1[3]), (255, 255, 255), 2)
        #     # cv2.imshow('image', blank_image)
        #     # cv2.waitKey(10)
        #
        #     # final_proposals = torch.cat((proposals1[j].view(a, b, 1), torch.cat((proposals2[j].view(a, b, 1), torch.cat((proposals3[j].view(a, b, 1), proposals4[j].bbox.view(a, b, 1)), 2)), 2)), 2)
        #     outputs.append(BoxList(proposals2[j], proposals[j].size))


        # x = self.feature_extractor(features, proposals)

        x = self.feature_extractor(features, proposals)

        # final regressor that converts the features into predictions
        poses = self.predictor(x)
        if not self.training:
            result = self.post_processor(poses, proposals)
            return x, result, {}

        loss_pose = self.loss_evaluator(proposals, poses, targets)
        return x, proposals, dict(loss_pose = loss_pose)



def build_roi_pose_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIPoseHead(cfg, in_channels)
