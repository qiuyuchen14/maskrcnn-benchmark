import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import sys
sys.path.append("/home/zoey/nas/zoey/github/maskrcnn-benchmark")
import os
from maskrcnn_benchmark.config import cfg
from kitchenPrediction_rgb import KitchenDemo
# from predictor import COCODemo
# import Image
import cv2

config_file = "/home/zoey/nas/zoey/github/maskrcnn-benchmark/configs/kitchen1.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

kitchen_demo = KitchenDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):

    plt.imshow(img[:, :, [2, 1, 0]])

    plt.axis("off")


# from http://cocodataset.org/#explore?id=345434
# for kitid in range(5):
    # path, dirs, files = next(os.walk("/home/zoey/nas/zoey/data/gt_coco/kitchen{0}/val".format(kitid+1)))
# path, dirs, files = next(os.walk("/home/zoey/ssds/data/Kitchen/robot_simulator/distance_test2/distance{0}".format(3)))
#     file_count = len(files)
for id in range(1, 500):
    # image = cv2.imread("/home/zoey/ssds/data/Kitchen/robot_simulator/distance_test2/distance{0}/frame{1}.png".format(3, id+21))
    image = cv2.imread("/home/zoey/nas/zoey/data/gt_coco/kitchen{0}/val/frame{1}.png".format(1, id+1))
    # image = cv2.imread("/home/zoey/ssds/data/Kitchen/renderpy/train2/frame{0}.png".format(id + 1))
    # image = np.array(image)[:, :, [2, 1, 0]]
    # compute predictions
    predictions = kitchen_demo.run_on_opencv_image(image, id+1)
    # plt.savefig("/home/zoey/nas/zoey/data/pred_gt_coco1/kitchen{0}/frame{1}.png".format(kitid+1, id+1), predictions[:, :, [2, 1, 0]])
    cv2.imwrite("/home/zoey/nas/zoey/github/maskrcnn-benchmark/checkpoints/combined_renderpy/prediction/hurb1/frame{0}.png".format(id+1), predictions)
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.merge_from_list(["MODEL.MASK_ON", False])

# image = cv2.imread(
#     "/home/zoey/nas/zoey/data/gt_coco/test5.png")
# # image = np.array(image)[:, :, [2, 1, 0]]
# # compute predictions
# predictions = kitchen_demo.run_on_opencv_image(image, 1)
# # plt.savefig("/home/zoey/nas/zoey/data/pred_gt_coco1/kitchen{0}/frame{1}.png".format(kitid+1, id+1), predictions[:, :, [2, 1, 0]])
# cv2.imwrite(
#     "/home/zoey/nas/zoey/data/gt_coco/test_prediction.png", predictions)
# cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])

        # imshow(predictions)
        # plt.show()
# # set up demo for keypoints
# cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])
# imshow(predictions)
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab

# import requests
# from io import BytesIO
# from PIL import Image
# import numpy as np

# from maskrcnn_benchmark.config import cfg
# from predictor import COCODemo
# import cv2


# config_file = "/home/zoey/nas/zoey/github/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
# # config_file = "/home/zoey/nas/zoey/github/maskrcnn-benchmark/coco/model_final.pth"
# # update the config options with the config file
# cfg.merge_from_file(config_file)
# # manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

# coco_demo = COCODemo(
#     cfg,
#     min_image_size=800,
#     confidence_threshold=0.7,
# )

# def load(url):
#     """
#     Given an url of an image, downloads the image and
#     returns a PIL image
#     """
#     response = requests.get(url)
#     pil_image = Image.open(BytesIO(response.content)).convert("RGB")
#     # convert to BGR format
#     image = np.array(pil_image)[:, :, [2, 1, 0]]
#     return image

# def imshow(img):
#     plt.imshow(img[:, :, [2, 1, 0]])
#     plt.axis("off")


# # from http://cocodataset.org/#explore?id=345434
# # image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# image = cv2.imread("/home/zoey/nas/zoey/github/maskrcnn-benchmark/datasets/coco/val2014/COCO_train2014_000000048759.jpg")

# imshow(image)

# # compute predictions
# predictions = coco_demo.run_on_opencv_image(image)
# print(predictions)
# imshow(predictions)
# plt.show()
# # set up demo for keypoints
# cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])
# imshow(predictions)
# plt.show()
