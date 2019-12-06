CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file ./configs/BBTP/e2e_mask_rcnn_R_101_FPN_4x_voc_aug_cocostyle.yaml