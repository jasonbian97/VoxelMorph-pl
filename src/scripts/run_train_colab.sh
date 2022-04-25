#!/bin/bash

# training settings
epoch=300
tr_bs=6
val_bs=6
gpu="0,"
lim_train=1.0
lim_val=1.0
val_num_workers=$val_bs
train_num_workers=$tr_bs
cache_rate=1.0
VMHCP_root_dir=/content/VoxelMorph-pl/data/cache/data.json

export PYTHONPATH="${PYTHONPATH}:/content/VoxelMorph-pl"
python /content/VoxelMorph-pl/src/scripts/train.py voxelmorph --exp_name LR_only_MI --variant LR  --image-loss mi --w_sm 0.1 --inshape 96 112 112 --pixdim 2 --train_dataset VMHCP --val_dataset VMHCP --VMHCP_root_dir $VMHCP_root_dir --cache_rate $cache_rate --gpu $gpu --train_batch_size $tr_bs --val_batch_size $val_bs --max_epochs $epoch --lim_val $lim_val --lim_train $lim_train --val_num_workers $val_num_workers --train_num_workers $train_num_workers
