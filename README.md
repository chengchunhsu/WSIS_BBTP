# Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

[[PDF\]](https://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf) [[BibTeX\]](https://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior/bibtex) [[Supplemental\]](https://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior-supplemental.zip)

![](/fig/intro.png)



This project hosts the code for the implementation of [Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior](http://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf) (NeurIPS 2019).

The main code is based on maskrcnn-benchmark ([\#5c44ca7](https://github.com/facebookresearch/maskrcnn-benchmark/tree/5c44ca7414b5c744aeda6d8bfb60d1de6d99c049)) and the post-processing code is based on [meanfield-matlab](https://github.com/johannesu/meanfield-matlab).



## Introduction

This paper presents a weakly supervised instance segmentation method that consumes training data with tight bounding box annotations. The major difficulty lies in the uncertain figure-ground separation within each bounding box since there is no supervisory signal about it. We address the difficulty by formulating the problem as a multiple instance learning (MIL) task, and generate positive and negative bags based on the sweeping lines of each bounding box. The proposed deep model integrates MIL into a fully supervised instance segmentation network, and can be derived by the objective consisting of two terms, i.e., the unary term and the pairwise term. The former estimates the foreground and background areas of each bounding box while the latter maintains the unity of the estimated object masks. The experimental results show that our method performs favorably against existing weakly supervised methods and even surpasses some fully supervised methods for instance segmentation on the PASCAL VOC dataset.



## Installation 

Check [INSTALL.md](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/INSTALL.md) for installation instructions. 



## Dataset

All details of dataset construction can be found in Sec 4.1 of [our paper](http://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf).

**Training**

We construct the training set by two following settings:

- PASCAL VOC (Augmented)
  - Extent the training set of VOC 2012 with [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) training set.
  - Result in an augmented set of 10582 training images. ([COCO format download link](https://drive.google.com/file/d/1lGCVvrst_PVsdG6C57Xz00PF3ge2kJgL/view?usp=sharing))

- PASCAL VOC (Augmented) + COCO
  - Extent the training set of VOC (Augmented) with COCO dataset.
  - Consider only the images that contain any of the 20 Pascal classes and only objects with a bounding box area larger than 200 pixels from COCO dataset.
  - After the ﬁltering, 99310 images remain from both the training and validation sets of COCO dataset.

**Testing**

We evaluate our method on PASCAL VOC 2012 validation set. ([COCO format download link](https://drive.google.com/file/d/1xBDKJmP-M8WWb0qTIbhdtunEIu4cAGFm/view?usp=sharing))

Note that the conversion of annotated format from VOC to COCO will result in inaccurate segment boundaries. See [Evaluation](https://github.com/chengchunhsu/WSIS_BBTP#Evaluation) for more details.

**Format and Path**

In our experiment, we convert the generated dataset into COCO format.

Before the training, please modified [paths_catalog.py]( https://github.com/chengchunhsu/WSIS_BBTP/blob/master/maskrcnn_benchmark/config/paths_catalog.py) and enter the correct data path for `voc_2012_aug_train_cocostyle`, `voc_2012_val_cocostyle`, and `voc_2012_coco_aug_train_cocostyle`.



## Training

Run the bash files directly:

**Training on PASCAL VOC (Augmented) with 4 GPUs**

```
bash train_voc_aug.sh
```

**Training on PASCAL VOC (Augmented) + COCO with 4 GPUs**

```
train_voc_coco_aug.sh
```



or type the bash commands:

**Training on PASCAL VOC (Augmented) with 4 GPUs**

```
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file ./configs/BBTP/e2e_mask_rcnn_R_101_FPN_4x_voc_aug_cocostyle.yaml
```

**Training on PASCAL VOC (Augmented) + COCO with 4 GPUs**

```
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file ./configs/BBTP/e2e_mask_rcnn_R_101_FPN_4x_voc_coco_aug_cocostyle.yaml
```



All bash commands are derived from maskrcnn-benchmark ([\#5c44ca7](https://github.com/facebookresearch/maskrcnn-benchmark/tree/5c44ca7414b5c744aeda6d8bfb60d1de6d99c049)).

You may also want to see the original [README.md](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/MASKRCNN_README.md) of maskrcnn-benchmark.



## Evaluation

Although COCO dataset has its own python API for evaluation, the conversion of annotated format from VOC to COCO will result in inaccurate segment boundaries.

To avoid such issue, we recommend to evaluate the predicted results via standard VOC API directly by the following steps:

**1. Save the predictions**

- Modify the key `TEST.SAVE_PRED_AS_MAT` as `True` in config files ([example](https://github.com/chengchunhsu/WSIS_BBTP/blob/c53d109100def34cc702086e0d94aa3959237e68/configs/BBTP/e2e_mask_rcnn_R_101_FPN_4x_voc_aug_cocostyle.yaml#L51)).
- Run `test_voc_aug.sh` or `test_voc_coco_aug.sh`, then the predictions will be saved as *mask.mat* in the directory. (the mat file is usually around 4~5 GB)

**2. Evaluate**

- Download [VOCcode](https://github.com/weiliu89/VOCdevkit/tree/master/VOCcode).
- Set up the paths in [EvalBaseline.m](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/matlab/EvalBaseline.m) (L3~L16).
- Run [EvalBaseline.m](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/matlab/EvalBaseline.m) in Matlab.

**(Optional) 3. Post-processing (DenseCRF)**

- Set up the paths in [Run_VOCInst.m](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/matlab/Run_VOCInst.m) (L4~L22).
- Run [Run_VOCInst.m](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/matlab/Run_VOCInst.m) in Matlab.



## Result

We provide the model weights and mask files of all experiments in this section.

**Reported in the main paper**

| Dataset                       | mAP@0.25 | mAP@0.50 | mAP@0.70 | mAP@0.75 | Post-processing | Model                                                        |
| ----------------------------- | -------- | -------- | -------- | -------- | --------------- | ------------------------------------------------------------ |
| PASCAL VOC (Augmented)        | 74.7     | 53.7     | 23.6     | 16.9     | w/o DenseCRF    | [link](https://drive.google.com/drive/folders/1p_I6eL7Ge4itWqzL1GR393W4AsLkDba9?usp=sharing) |
| PASCAL VOC (Augmented)        | 75.0     | 58.9     | 30.4     | 21.6     | w/ DenseCRF     | -                                                            |
| PASCAL VOC (Augmented) + COCO | 76.8     | 54.4     | 23.7     | 17.4     | w/o DenseCRF    | [link](https://drive.google.com/drive/folders/1p_I6eL7Ge4itWqzL1GR393W4AsLkDba9?usp=sharing) |
| PASCAL VOC (Augmented) + COCO | 77.2     | 60.1     | 29.4     | 21.2     | w/ DenseCRF     | -                                                            |



**Reproduced with the released code**

| Dataset                | mAP@0.25 | mAP@0.50 | mAP@0.70 | mAP@0.75 | Post-processing | Model                                                        |
| ---------------------- | -------- | -------- | -------- | -------- | --------------- | ------------------------------------------------------------ |
| PASCAL VOC (Augmented) | 74.0     | 54.1     | 24.5     | 17.1     | w/o DenseCRF    | [link](https://drive.google.com/drive/folders/1MI51WI9rBcSr9I6jW38rYO3jy-Mngnkw?usp=sharing) |
| PASCAL VOC (Augmented) | 74.4     | 59.1     | 30.2     | 21.9     | w/ DenseCRF     | -                                                            |

\* training log can be found [here](https://drive.google.com/drive/folders/1MI51WI9rBcSr9I6jW38rYO3jy-Mngnkw?usp=sharing).



All experiments is running on following environments:

- **Hardware**
  - 4 NVIDIA 1080 Ti GPUs

- **Software**
  - PyTorch version: 1.0.1
  - CUDA 10.2



## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{hsu2019bbtp,
  title     =  {Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior},
  author    =  {Cheng-Chun Hsu, Kuang-Jui Hsu, Chung-Chi Tsai, Yen-Yu Lin, Yung-Yu Chuang},
  booktitle =  {Neural Information Processing Systems},
  year      =  {2019}
}
```