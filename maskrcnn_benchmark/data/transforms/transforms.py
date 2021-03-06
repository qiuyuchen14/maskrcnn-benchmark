# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import torch.nn.functional as F1
import numpy
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, target):
        for t in self.transforms:
            image, depth, target = t(image, depth, target)
        return image, depth, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, depth, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        x = np.arange()
        depth = scipy.interpolate.interp2d(depth)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, depth, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, depth, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            depth = numpy.flip(depth, 1)#F.hflip(depth)
            target = target.transpose(0)
        return image, depth, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, depth, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            depth = numpy.flip(depth, 0)#F.vflip(depth)
            target = target.transpose(1)
        return image, depth, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, depth, target):
        image = self.color_jitter(image)
        return image, depth, target


# class RandomRotate(object):
#     def __init__(self, deg):
#         self.deg = deg
#         self.random_rotation = torchvision.transforms.RandomRotation(self.deg)
#
#     def __call__(self, image, target=None):
#         image = self.random_rotation(image)
#         target = target.rotate(self.deg)
#         return image, target

class ToTensor(object):
    def __call__(self, image, depth, target):
        return F.to_tensor(image), depth, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, depth, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, depth, target


# class RandomCrop(object):
#     def __init__(self, min_size, max_size):
#         self.min_size = min_size
#         self.max_size = max_size
#
#     @staticmethod
#     def get_params(img, min_size, max_size):
#         """Get parameters for ``crop`` for a random crop.
#         Args:
#             img (PIL Image): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         w, h = img.size
#         th = random.randint(min_size, max_size)
#         tw = th
#         if w == tw and h == th:
#             return 0, 0, h, w
#
#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#         return i, j, th, tw
#
#     def __call__(self, image, target):
#         i, j, h, w = self.get_params(image, self.min_size, self.max_size)
#         image = F.crop(image, i, j ,h, w)
#         target = target.crop(i, j, h, w)
#         return image, target
