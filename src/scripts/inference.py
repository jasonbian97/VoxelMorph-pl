""" Check if load the right weights
# way 1
what = torch.load(ckpt)
args = EasyDict(what['hyper_parameters'])
model1 = VoxelMorph(args)
model1.load_state_dict(what["state_dict"])
key = list(what["state_dict"].keys())[0]

assert (model1.state_dict()[key].cpu() == what["state_dict"][key].cpu()).all()
assert (model2.state_dict()[key].cpu() == what["state_dict"][key].cpu()).all()

"""

from src.lib.VoxelMorph import VoxelMorph
import random
import torch
from easydict import EasyDict
from monai.transforms import (
ScaleIntensityRangePercentilesd,
CenterSpatialCropd,
SpatialPadd,
    AddChanneld,
    Compose,
    LoadImaged,
    Spacingd,
    EnsureTyped,
)
import ants

from dipy.tracking import utils
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import _mapping_to_voxel, _to_voxel_coordinates

from nilearn.image import resample_img

import nibabel as nib
import numpy as np
import os
from os.path import join
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
import random
import subprocess
from glob import glob
from easydict import EasyDict
import pandas as pd
from path import Path
import scipy.ndimage as ndi
import shutil
from einops import rearrange

# way 2
dout = "/mnt/ssd2/Projects/EPICorrection/data/cache/680957/VoxelMorphBaseline"
if not os.path.exists(dout):
    os.makedirs(dout)

inputs = {"LR": "/mnt/ssd2/Projects/EPICorrection/data/cache/680957/LR2T1w_affine.nii.gz",
          "T1w": "/mnt/ssd2/Projects/EPICorrection/data/cache/680957/T1w_acpc_dc_restore_brain.nii.gz"}

ckpt = "/mnt/ssd2/Projects/EPICorrection/src/scripts/epicorrection/fbbd3ed172c6486db87220fa6d5709f6/checkpoints/voxelmorph_last_epoch=249.ckpt"
model = VoxelMorph.load_from_checkpoint(ckpt)
model.freeze()

pixdim = model.args.pixdim
img_names = ["LR", "T1w"]
transform = Compose(
    [
        LoadImaged(keys=img_names),
        AddChanneld(keys=img_names),
        Spacingd(keys=img_names, pixdim=(pixdim, pixdim, pixdim), mode="bilinear",
                 padding_mode="border"),

        ScaleIntensityRangePercentilesd(keys=img_names, lower=0.01, upper=99.99, b_min=0,
                                        b_max=1, clip=True),

        CenterSpatialCropd(keys=img_names, roi_size=model.args.inshape),
        SpatialPadd(keys=img_names, spatial_size=model.args.inshape, mode="constant"),
        EnsureTyped(keys=img_names),
    ])
adder = AddChanneld(keys=img_names) # add pseudo batch dimension


# img = ants.image_read(inputs["LR"])
img_data, _, img = load_nifti(inputs["LR"], return_img=True)

inputs = adder(transform(inputs))
outputs = model(inputs,registration=True)

warped = outputs["warped"].cpu().numpy().squeeze() * np.percentile(img_data[:], 99.99)
affine = inputs["LR_meta_dict"]["affine"]

transformation = outputs["flow"].cpu().squeeze().numpy()
transformation = rearrange(transformation," C H W D -> H W D C")

out_name = join(dout, "LR2T1w_vm.nii.gz")
nib.Nifti1Image(warped.astype(np.uint32), affine, img.header).to_filename(out_name)

out_name = join(dout, "LR2T1w_vm_transform.nii.gz")
nib.Nifti1Image(transformation.astype(np.float32), affine, img.header).to_filename(out_name)

print()



