# GRADE-nets

This repository contains useful scripts and the permanent snapshot of the networks used to train GRADE data.

Please follow the respective YOLOv5 and MASK R-CNN installation instructions and requirements.

In the `yolo_configs` folder you can find the configuration we used for the various datasets. These configs should normally go in the `data` folder of the `yolo` installation alongside all the other configs.

In the script folders you can find:
1. `coco_to_npy.py` using MaskRCNN data loader youo can convert masks to npy files. The mask  can be saved either as `uint8` or as `bool`. You may want to use `pycocotools` directly to get the imgIds.
2. `convert_coco_json_to_yolo_txt` takes two arguments, the output folder and the input json file, and returns bboxes in yolo format
3. `convert_npy_mask_to_coco_json` takes as input a folder and write a json coco-style in the same folder. You may want to add categories following your needs, or change the file iterator.
4. `mrcnn_get_loss.py` takes in input an experiment folder and returns the epoch and the loss value. The loss is computed using the tensorflow logs and is either the sum of all the losses or the sum of just bbox and mask loss. 
5. `mrcnn_test.py` is what we used for getting the results. It loads a coco-style (even the folder needs to be coco style with `annotations` and `val2017/images`) dataset [tum and coco in our case] and save the results in npy. It needs our modified version of Mask RCNN.
6. `test.py` is a simple script that shows how you can test the maskrcnn with your custom dataset, weights, or classes.
7. `visualize_coco_style_annos` is a notebook to be used with maskrcnn that loads a coco-style folder and helps you visualize the mask. It is useful to check if the data has been converted correctly and to show how you can go from ImgIds, annos, and mask and directly show them.

All scripts regarding Mask R-CNN should be copied into the repository `mrcnn` folder. They are reported here since they can be generalized with minimal effort using pycocotools.

For LICENSE, refer to the main [GRADE](https://github.com/eliabntt/GRADE-RR) repository. The same terms applies here.
