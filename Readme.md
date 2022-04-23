# VoxelMorph (Pytroch + PytrochLightning + Monai)

## Introduction
This is an unofficial implementation for VoxelMorph. Pytroch Lightning + Monai are used to construct the training & evaluation framework to facilitate people make their own modifications and run experiment. 

Monai's CacheDataset is used to accelerate training speed by 20x.

## What's the difference with official VoxelMorph-repo?
This repo built upon the official VoxelMorph-repo, but has several differences: 

- Official repo does not provide a directly-runable *pytroch* training script. (as the time of my writing)
- Official repo does not provide a pretrained model in pytorch format.
- This repo includes the mutual information (MI) as the loss function
- This repo uses Monai framework for image preprocessing and augmentation
- This repo uses Pytorch-Lightning framework to manage more complicated experiment, which is easier for developers and researchers who want to make modifications to network architecture, training strategy or data augmentation.

## How to use
Before training, I used to organize summarize training dataset information in a json file, which will be used for construct `Dataset` class. An exmaple can be found [here](./data/cache/data.json).
Note that only the nii.gz path is given, and monai will automatically read it.

To do the training, run
```shell script
bash src/script/run_train.sh
```

To do the inference using pretrained model:
```shell script
# modifiy the image path
python inference.py 
```

# TODO
- Write more informative doc.
- Provide pytorch pretrained model on OASIS dataset for T1w/MPRAGE registration
- Provide pytorch pretrained model on HCP dataset
