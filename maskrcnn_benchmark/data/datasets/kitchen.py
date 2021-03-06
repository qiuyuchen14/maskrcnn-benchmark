# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import matplotlib.pyplot as plt

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import numpy as np
from torchvision.transforms import functional as F
min_keypoints_per_image = 10
from PIL import Image
import math
import scipy.misc

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class KitchenDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None, depth_on=True):
        super(KitchenDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.depth_on = depth_on

    def __getitem__(self, idx):
        img, anno = super(KitchenDataset, self).__getitem__(idx)
        # loading depth images:

        depth = np.repeat(np.zeros((img.size[1], img.size[0], 1)), 3, axis=2)

        if self.depth_on:
            depth = np.load("/home/zoey/ssds/data/Kitchen/data1/depth/frame{0}.npy".format(idx+1))
            # fov, aspect, nearplane, farplane = 45, imgWidth / imgHeight, 0.01, 100
            # depth = farplane * nearplane / (farplane - (farplane - nearplane) *depth1)
            # TODO: fix the max value
            max_depth = 5
            min_depth = 0.01
            depth = (depth-min_depth)/(max_depth - min_depth)
            depth = np.repeat(depth, 3, axis=2)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]


        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]

        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)


        target.add_field("labels", classes)


        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')

            target.add_field("masks", masks)

        if anno and "pose" in anno[0]:
            pose = [obj["pose"] for obj in anno]
            pose = torch.tensor(pose)
            target.add_field("pose", pose)

        # if anno and "pose" in anno[0]:
        #     # poses = [obj["pose"]+1 for obj in anno]
        #     poses = [obj["pose"] for obj in anno]
        #
        #     poses = [self.json_category_id_to_contiguous_id[c] for c in poses]
        #     poses = torch.tensor(poses)
        #
        #     target.add_field("pose", poses)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:

            img, depth, target = self._transforms(img, depth, target)


        if self.depth_on:
            depth = depth * 255 - [102.9801, 115.9465, 122.7717]

            depth = torch.from_numpy(depth.copy())
            depth = depth.permute(2, 0, 1)
            
        if not self.depth_on:
            depth = torch.from_numpy(depth.copy())
            depth = depth.permute(2, 0, 1)

        return img, target, depth, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
