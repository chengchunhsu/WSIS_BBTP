# Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

This project hosts the code for the implementation of [Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior](http://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf) (NeurIPS 2019).

The code is based on maskrcnn-benchmark ([\#5c44ca7](https://github.com/facebookresearch/maskrcnn-benchmark/tree/5c44ca7414b5c744aeda6d8bfb60d1de6d99c049)).



## Installation 

Check [INSTALL.md](https://github.com/chengchunhsu/WSIS_BBTP/blob/master/INSTALL.md) for installation instructions. 



## Dataset

All details of dataset construction can be found in Sec 4.1 of [our paper](http://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf).

**Training**

We construct the training set by two following settings:

- PASCAL VOC (Augmented)
  - Extent the training set of VOC 2012 with [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) training set.
  - Result in an augmented set of 10582 training images.

- PASCAL VOC (Augmented) + COCO
  - Extent the training set of VOC (Augmented) with COCO dataset.
  - Consider only the images that contain any of the 20 Pascal classes and only objects with a bounding box area larger than 200 pixels from COCO dataset.
  - After the ﬁltering, 99310 images remain from both the training and validation sets of COCO dataset.

**Testing**

We evaluate our method on PASCAL VOC 2012 validation set.

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