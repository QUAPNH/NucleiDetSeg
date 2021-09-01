# cas-dc-template
A Region-based Convolutional Network for Nuclei Detection and Segmentation in Microscopy Images
## Introduction
The installation of MMDetection can be found from the official github(https://github.com/open-mmlab/mmdetection/blob/v2.5.0/docs/install.md).<br>

In this paper, we propose a region-based convolutional network for a more accurate nuclei detection. <br>
![IoUPred](https://user-images.githubusercontent.com/54254748/131081566-2644b250-ddaa-4b47-a506-7b2a71315122.png)

## Configuration
Our method is improved on the basis of Mask-RCNN, including GA-RPN,FBS and SoftNMS modules. The corresponding configuration can be found in(Ours\GARPN_FBS_SoftNMS-r50_fpn_1x_coco.py).

## Preparing Data
The mmdetection supports the coco dataset, but the DSB and MonuSeg datasets we use are not in the coco format. So we need to convert them to the coco format.<br>

The dataset that we processed can be downloaded from [here](https://drive.google.com/drive/folders/19SRU1PyKoz-kdOzh-WBktCz3P6QjWvbo) or [Dropbox](https://www.dropbox.com/sh/vcm8s3vtglhjbv5/AACmzwwTOIIYn2nVg2bLNZ_9a?dl=0).

## Get Started
```
./tools/train 
```

## Visualization of Results
For visual assessment of the experiments results, we detailed display the visual results of six images of DSB and seven images of monuseg in the paper.

To further prove the effectiveness of our method, we publish all the visual images in [here](https://drive.google.com/drive/folders/1fG1nQVqxlANfUNfIZMM1T6wf71uxgImn) or [Dropbox](https://www.dropbox.com/sh/0hsga3f3kamgn65/AACZcDlC5jKlAqA79An5eCOGa?dl=0).
