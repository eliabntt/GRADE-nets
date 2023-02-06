import os
import sys
import random
import math
import re
import time
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
import utils
import visualize
import model as modellib
from model import log
from dataset import GRADEDataset
from dataset import GRADEConfig
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from coco import evaluate_coco
from coco import CocoDataset

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Evaluation
#class InferenceConfig(GRADEConfig):
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False
    NUM_CLASSES = 1 + 1 # 80

inference_config = InferenceConfig()
inference_config.display()


model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#model_path = COCO_MODEL_PATH


res = {}
res["coco_scratch"] = {}
res["coco_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/coco_scratch/mask_rcnn_coco_min_scratch_nobg_500_0101.h5" #101 - 1.9145, 273 - 1.010656

res["grade_blur_scratch"] = {}
res["grade_blur_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_blur_scratch/mask_rcnn_grade_blur_scratch_0295.h5" # 295 - 0.691497, 249 - 0.274253

res["grade_gt_scratch"] = {}
res["grade_gt_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_gt_scratch/mask_rcnn_grade_blur_load_resnet_3_stages_0270.h5" # 270 - 0.499945, 241 - 0.158938

res["grade_blur_fine"] = {}
res["grade_blur_fine"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_blur_scratch_fine/mask_rcnn_blur_scratch_fine_coco_500_0073.h5" # 73 - 1.496842, 55 - 0.8086484

res["grade_gt_fine"] = {}
res["grade_gt_fine"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_gt_scratch_fine/mask_rcnn_grade_gt_3stages_coco_500_0064.h5" # 64 - 1.4293022, 64 - 0.8214

lim = 0

for key in res.keys():
    k = res[key]
    model_path = k["checkpoint"]
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco("/home/ebonetto/coco/coco", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
    dataset_val.prepare()
    k["coco"] =  {}
    k["tum"] = {}
    k["coco"]["segm"], k["coco"]["bbox"] = evaluate_coco(model, dataset_val, coco, "segm", limit=lim)
    
    dataset_val = CocoDataset()
    coco = dataset_val.load_coco("/home/ebonetto/tum_valid", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
    dataset_val.prepare()
    k["tum"]["segm"], k["tum"]["bbox"] = evaluate_coco(model, dataset_val, coco, "segm", limit=lim)

np.save("res_all_metrics.npy", res)

res = {}
res["coco_scratch"] = {}
res["coco_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/coco_scratch/mask_rcnn_coco_min_scratch_nobg_500_0273.h5" #101 - 1.9145, 273 - 1.010656

res["grade_blur_scratch"] = {}
res["grade_blur_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_blur_scratch/mask_rcnn_grade_blur_scratch_0249.h5" # 295 - 0.691497, 249 - 0.274253

res["grade_gt_scratch"] = {}
res["grade_gt_scratch"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_gt_scratch/mask_rcnn_grade_blur_load_resnet_3_stages_0241.h5" # 270 - 0.499945, 241 - 0.158938

res["grade_blur_fine"] = {}
res["grade_blur_fine"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_blur_scratch_fine/mask_rcnn_blur_scratch_fine_coco_500_0055.h5" # 73 - 1.496842, 55 - 0.8086484

res["grade_gt_fine"] = {}
res["grade_gt_fine"]["checkpoint"] = "/ps/project/irotate/GRADE_nets/train/maskrcnn/grade_gt_scratch_fine/mask_rcnn_grade_gt_3stages_coco_500_0064.h5" # 64 - 1.4293022, 64 - 0.8214

for key in res.keys():
    k = res[key]
    model_path = k["checkpoint"]
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco("/home/ebonetto/coco/coco", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
    dataset_val.prepare()

    k["coco"] =  {}
    k["tum"] = {}
    k["coco"]["segm"], k["coco"]["bbox"] = evaluate_coco(model, dataset_val, coco, "segm", limit=lim)
    

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco("/home/ebonetto/tum_valid", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
    dataset_val.prepare()
    k["tum"]["segm"], k["tum"]["bbox"] = evaluate_coco(model, dataset_val, coco, "segm", limit=lim)

np.save("res_bbox_mask_metrics.npy", res)