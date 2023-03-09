# GRADE-nets

## This repository is part of the [GRADE](https://eliabntt.github.io/GRADE-RR/home) project

This repository contains useful scripts and the permanent snapshot of the networks used to train GRADE data.

This will automatically download the `YOLOv5` and `detectron2` repositories at the commits used in our trainings.

Please follow the respective YOLOv5 and detectron2 installation instructions and requirements.

In the `yolo_configs` folder you can find the configuration we used for the various datasets. These configs should normally go in the `data` folder of the `yolo` installation alongside all the other configs.

In the `detectron2_train` folder you can instead finde the `LossEvalHook` script and the train script that we used. The `train_coco.py` file will also run evaluations with TUM and COCO datasets and save these as npy files. Please place these files in `detectron2/tools` folder and simply run the training code.

In the script folders you can find:
1. `coco_to_npy.py` using MaskRCNN data loader youo can convert masks to npy files. The mask  can be saved either as `uint8` or as `bool`. You may want to use `pycocotools` directly to get the imgIds.
2. `convert_coco_json_to_yolo_bbox.py` takes two arguments, the output folder and the input json file, and returns bboxes in yolo format
3. `convert_npy_mask_to_coco_json.py` takes as input a folder and write a json coco-style in the same folder. You may want to add categories following your needs, or change the file iterator.
4. `get_tensorboard_loss.py` takes in input an experiment folder and returns the epoch and the loss value. The loss is computed using the tensorflow logs and is either the sum of all the losses or the sum of just bbox and mask loss. You can easily edit that to get your custom losses or a different combination. 
5. `visualize_coco_style_annos.ipynb` is a notebook to be used with maskrcnn that loads a coco-style folder and helps you visualize the mask. It is useful to check if the data has been converted correctly and to show how you can go from ImgIds, annos, and mask and directly show them.
6. `filter_coco_json.py` is a script to select a specific subset of a given COCO json

For LICENSE, refer to the main [GRADE](https://github.com/eliabntt/GRADE-RR) repository. The same terms applies here. By using any related software you are anyway implicitly accepting and must comply with the corresponding terms. This includes, but is not limited to, YOLOv5 terms, GRADE, GRADE-RR, or more in general the GRADE-repositories terms, Matterport Mask R-CNN and any other software you may use for this.

The results and the network models can be found [here](https://github.com/eliabntt/GRADE_data)

__________
### CITATION
If you find this work useful please cite our work as

```
@misc{https://doi.org/10.48550/arxiv.2303.04466,
  doi = {10.48550/ARXIV.2303.04466},
  url = {https://arxiv.org/abs/2303.04466},
  author = {Bonetto, Elia and Xu, Chenghao and Ahmad, Aamir},
  keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GRADE: Generating Realistic Animated Dynamic Environments for Robotics Research},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
