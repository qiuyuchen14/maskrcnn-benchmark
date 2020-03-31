# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

## fc layers
def fc(batch_norm, nc_inp, nc_out):
    if batch_norm:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out, bias=True),
            nn.BatchNorm1d(nc_out),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Linear(nc_inp, nc_out),
            nn.LeakyReLU(0.1,inplace=True)
        )

def fc_stack(nc_inp, nc_out, nlayers, use_bn=True):
    modules = []
    for l in range(nlayers):
        modules.append(fc(use_bn, nc_inp, nc_out))
        nc_inp = nc_out
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            #n = m.out_features
            #m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
            #n = m.in_features
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #xavier
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #this modified initialization seems to work better, but it's very hacky
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class RoiEncoder(nn.Module):
    def __init__(self, nc_inp_fine, nc_inp_coarse, use_context=True, nz_joint=1024, nz_roi=1024, nz_coarse=1024, nz_box=1024):
        super(RoiEncoder, self).__init__()
        self.encoder_coarse = fc_stack(nc_inp_coarse, nz_coarse, 2)
        self.encoder_bbox = fc_stack(4, nz_box, 3)
        self.encoder_joint = fc_stack(nz_joint, nz_joint, 2)
        self.use_context = use_context

    def forward(self, feats, bbox_flag):
        roi_img_feat, img_feat_coarse, rois_inp = feats
        feat_fine = roi_img_feat
        feat_coarse = self.encoder_coarse.forward(img_feat_coarse)

        if len(rois_inp)==2:

            feat_coarse = torch.cat((feat_coarse[0].repeat(1, rois_inp[0].bbox.shape[0]).view(-1, 1024), feat_coarse[1].repeat(1, rois_inp[1].bbox.shape[0]).view(-1, 1024)), dim=0)

            feat_bbox = self.encoder_bbox.forward(torch.cat((rois_inp[0].bbox / 480, rois_inp[1].bbox / 480), dim=0))
        else:
            feat_coarse = feat_coarse[0].repeat(1, rois_inp[0].bbox.shape[0]).view(-1, 1024)
            feat_bbox = self.encoder_bbox.forward(rois_inp[0].bbox / 480)
        if bbox_flag:
            img_feat = torch.cat((torch.cat((feat_fine, feat_coarse), dim=1), feat_bbox), dim=1)
        else:
            img_feat = torch.cat((feat_fine, feat_coarse), dim=1)
        img_feat = self.encoder_joint.forward(img_feat)
        return img_feat


import torchvision
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks=n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.encoder_coarse = fc_stack(512 * (128 // 32) * (256 // 32), 300, 2)
        self.coarse = cfg.MODEL.COARSE_ON
        self.bbox = cfg.MODEL.BBOXFeature_ON
        self.rgb_on = cfg.MODEL.RGB_ON
        self.dcoarse = cfg.MODEL.DCOARSE_ON
        self.depth_on = cfg.MODEL.DEPTH_ON
        self.dbbox = cfg.MODEL.DBBOXFeature_ON
        roi_size = 2
        nc_inp_coarse = 512 * (128 // 32) * (256 // 32)
        if self.rgb_on:
            if cfg.MODEL.COARSE_ON and not cfg.MODEL.BBOXFeature_ON:
                nz_joint_value =2048
            if cfg.MODEL.COARSE_ON and cfg.MODEL.BBOXFeature_ON:
                nz_joint_value = 3072
            if cfg.MODEL.COARSE_ON:
                self.roi_encoder = RoiEncoder(256 * roi_size * roi_size, nc_inp_coarse, nz_joint=nz_joint_value)

        if self.depth_on:
            if cfg.MODEL.DCOARSE_ON and not cfg.MODEL.DBBOXFeature_ON:
                nz_joint_value = 2048
            if cfg.MODEL.DCOARSE_ON and cfg.MODEL.DBBOXFeature_ON:
                nz_joint_value = 3072
            if cfg.MODEL.DCOARSE_ON:
                self.roi_encoder = RoiEncoder(256 * roi_size * roi_size, nc_inp_coarse, nz_joint=nz_joint_value)
        if self.depth_on and self.rgb_on:
            self.encoder_rgbd = fc_stack(3072*2, 3072*2, 2)

    def forward(self, features, images_coarse, depth_features, depth_coarse, proposals, targets=None):
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

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        # x_rgb = self.feature_extractor(features, proposals)
        # x_depth = self.feature_extractor(features, proposals)
        x = self.feature_extractor(features, proposals)
        if self.rgb_on:
            if self.coarse:
                if self.training:
                    resnet_conv_coarse = ResNetConv(n_blocks=4).cuda()
                else:
                    resnet_conv_coarse = ResNetConv(n_blocks=4).cuda()
                coarse_features = resnet_conv_coarse.forward(images_coarse)
                coarse_features = coarse_features.view(coarse_features.size(0), -1)
                # roi_size = 2
                # nc_inp_coarse = 512 * (128 // 32) * (256 // 32)
                # roi_encoder = RoiEncoder(256 * roi_size * roi_size, nc_inp_coarse, use_context=True, nz_joint=1024, nz_roi=1024, nz_coarse=1024, nz_box=proposals[0].bbox.shape[0] + proposals[1].bbox.shape[0]).cuda()

                x_rgb = self.roi_encoder.forward((x, coarse_features, proposals), self.bbox)

        if self.depth_on:
            if self.dcoarse:
                if self.training:
                    resnet_conv_coarse = ResNetConv(n_blocks=4).cuda()
                else:
                    resnet_conv_coarse = ResNetConv(n_blocks=4).cuda()
                coarse_features = resnet_conv_coarse.forward(depth_coarse)
                coarse_features = coarse_features.view(coarse_features.size(0), -1)
                x_depth = self.roi_encoder.forward((x, coarse_features, proposals), self.dbbox)

        if self.rgb_on:
            x = x_rgb
        if self.depth_on:
            x = x_depth
        if self.rgb_on and self.depth_on:
            x_rgbd = torch.cat((x_rgb, x_depth), dim=1)
            x = self.encoder_rgbd.forward(x_rgbd)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
