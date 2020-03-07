# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import torch
# from torch.nn import functional as F
# import numpy as np
# from maskrcnn_benchmark.layers import smooth_l1_loss
# from maskrcnn_benchmark.layers import l1_loss
# from maskrcnn_benchmark.layers import huber_loss_pose
# from maskrcnn_benchmark.modeling.box_coder import BoxCoder
# from maskrcnn_benchmark.modeling.matcher import Matcher
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
# from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
#     BalancedPositiveNegativeSampler
# )
# from maskrcnn_benchmark.modeling.utils import cat
#
#
# class FastRCNNLossComputation(object):
#     """
#     Computes the loss for Faster R-CNN.
#     Also supports FPN
#     """
#
#     def __init__(
#         self,
#         proposal_matcher,
#         fg_bg_sampler
#     ):
#         """
#         Arguments:
#             proposal_matcher (Matcher)
#             fg_bg_sampler (BalancedPositiveNegativeSampler)
#             box_coder (BoxCoder)
#         """
#         self.proposal_matcher = proposal_matcher
#         self.fg_bg_sampler = fg_bg_sampler
#
#     def match_targets_to_proposals(self, proposal, target):
#         match_quality_matrix = boxlist_iou(target, proposal)
#         matched_idxs = self.proposal_matcher(match_quality_matrix)
#         # Fast RCNN only need "labels" field for selecting the targets
#         # target = target.copy_with_fields(["labels", "pose"])
#         target = target.copy_with_fields(["labels", "pose"])
#
#         # get the targets corresponding GT for each proposal
#         # NB: need to clamp the indices because we can have a single
#         # GT in the image, and matched_idxs can be -2, which goes
#         # out of bounds
#         matched_targets = target[matched_idxs.clamp(min=0)]
#         matched_targets.add_field("matched_idxs", matched_idxs)
#         return matched_targets
#
#     def prepare_targets(self, proposals, targets):
#         labels = []
#         poses = []
#         for proposals_per_image, targets_per_image in zip(proposals, targets):
#             matched_targets = self.match_targets_to_proposals(
#                 proposals_per_image, targets_per_image
#             )
#             matched_idxs = matched_targets.get_field("matched_idxs")
#
#             labels_per_image = matched_targets.get_field("labels")
#             labels_per_image = labels_per_image.to(dtype=torch.int64)
#
#             # Label background (below the low threshold)
#             bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
#             labels_per_image[bg_inds] = 0
#
#             neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
#             labels_per_image[neg_inds] = 0
#
#             positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
#
#             # articulated_poses = matched_targets.get_field("pose")
#             # articulated_poses = articulated_poses[positive_inds]
#
#             articulated_poses = matched_targets.get_field("pose")
#             articulated_poses = articulated_poses.to(dtype=torch.int64)
#             articulated_poses[bg_inds]=0
#             labels_per_image[neg_inds] = 0
#
#
#             labels.append(labels_per_image)
#             poses.append(articulated_poses)
#
#         return labels, poses
#
#     def subsample(self, proposals, targets):
#         """
#         This method performs the positive/negative sampling, and return
#         the sampled proposals.
#         Note: this function keeps a state.
#
#         Arguments:
#             proposals (list[BoxList])
#             targets (list[BoxList])
#         """
#
#         labels, poses = self.prepare_targets(proposals, targets)
#         sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
#
#         proposals = list(proposals)
#         # add corresponding label and regression_targets information to the bounding boxes
#         # for labels_per_image, proposals_per_image in zip(
#         #     labels, proposals
#         # ):
#         #     proposals_per_image.add_field("labels", labels_per_image)
#         for labels_per_image, proposals_per_image in zip(
#             poses, proposals
#         ):
#             proposals_per_image.add_field("poses", labels_per_image)
#
#         # distributed sampled proposals, that were obtained on all feature maps
#         # concatenated via the fg_bg_sampler, into individual feature map levels
#         for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
#             zip(sampled_pos_inds, sampled_neg_inds)
#         ):
#             img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
#             proposals_per_image = proposals[img_idx][img_sampled_inds]
#             proposals[img_idx] = proposals_per_image
#
#         self._proposals = proposals
#         return proposals
#
#     def __call__(self, proposals, pose_logits, targets):
#         """
#         Arguments:
#             proposals (list[BoxList])
#             mask_logits (Tensor)
#             targets (list[BoxList])
#
#         Return:
#             mask_loss (Tensor): scalar tensor containing the loss
#         """
#         labels, pose_targets = self.prepare_targets(proposals, targets)
#
#         labels = cat(labels, dim=0)
#         pose_targets = cat(pose_targets, dim=0)
#         #
#         positive_inds = torch.nonzero(labels > 0).squeeze(1)
#         labels_pos = labels[positive_inds]
#         #
#         # # torch.mean (in binary_cross_entropy_with_logits) doesn't
#         # # accept empty tensors, so handle it separately
#         if pose_targets.numel() == 0:
#             return pose_logits.sum() * 0
#
#         pose_logits = cat([pose_logits], dim=0)
#         device = pose_logits.device
#
#         if not hasattr(self, "_proposals"):
#             raise RuntimeError("subsample needs to be called before")
#
#         # # proposals = self._proposals
#         #
#         # # mask = torch.zeros(pose_logits.size())
#         # # mask[:, 12] = 1  # The first pose class
#         # # mask[:, 13] = 1  # The second pose class
#         # # mask[:, 14] = 1  # The third pose class
#         # # mask[:, 15] = 1  # The forth pose class
#         # # mask[:, 16] = 1  # The fifth pose class
#         # # pose_logits = pose_logits * mask.cuda().float()
#         # # pose_loss = F.cross_entropy(pose_logits[positive_inds], pose_targets)
#         # proposals = self._proposals
#         # # pose_targets = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
#         # pose_targets = cat([proposal.get_field("poses") for proposal in proposals], dim=0)
#         #
#         # pose_loss = F.cross_entropy(pose_logits, pose_targets)
#
#         mask = torch.zeros(pose_logits.size())
#         mask[:, 1] = 1 # The first one is drawer, and the pose gradient should be 1.
#         mask[:, 2] = 1  # The second one is door, and the pose gradient should be 1.
#         pose_logits = pose_logits * mask.cuda().float()
#         pose_loss = smooth_l1_loss(pose_logits[positive_inds, labels_pos], pose_targets, 0.1)
#
#
#         return pose_loss
#
#
# def make_roi_pose_loss_evaluator(cfg):
#     matcher = Matcher(
#         cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
#         cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
#         allow_low_quality_matches=False,
#     )
#
#     fg_bg_sampler = BalancedPositiveNegativeSampler(
#         cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
#     )
#
#     loss_evaluator = FastRCNNLossComputation(
#         matcher,
#         fg_bg_sampler
#     )
#
#     return loss_evaluator

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "pose"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        poses = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            articulated_poses = matched_targets.get_field("pose")
            articulated_poses = articulated_poses[positive_inds]

            labels.append(labels_per_image)
            poses.append(articulated_poses)

        return labels, poses


    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, poses = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposals_per_image in zip(
            labels, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, proposals, pose_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, pose_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        pose_targets = cat(pose_targets, dim=0)
        #
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        #
        # # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # # accept empty tensors, so handle it separately
        if pose_targets.numel() == 0:
            return pose_logits.sum() * 0

        pose_logits = cat([pose_logits], dim=0)
        device = pose_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        mask = torch.ones(pose_logits.size())
        mask[:, 3] = 0 # The third one is handle, and the pose should be zero.
        pose_logits = pose_logits * mask.cuda().float()
        # pose_logits[:, 3] = torch.tensor(torch.zeros([1, pose_logits.shape[0]], dtype=torch.float64), requires_grad=True)
        pose_loss = smooth_l1_loss(pose_logits[positive_inds, labels_pos].float(), pose_targets.float(), 0.1)
        # pose_loss = l1_loss(pose_logits[positive_inds, labels_pos], pose_targets)

        return pose_loss


def make_roi_pose_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler
    )

    return loss_evaluator