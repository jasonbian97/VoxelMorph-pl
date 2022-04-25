from src.lib.BaseModel import BaseModel
import src.lib.networks as networks
from src.lib.losses import VMHCPLoss
from src.lib.Metrics import VMHCPMetric
from argparse import ArgumentParser, Namespace
import random
import torch
"""
1. Architect
2. Loss
3. Metric
4. Forward function
"""
# TODO: change upsample path to transpose conv
# TODO:

class VoxelMorph(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        # Model similar to previous section:
        self.loss_fn = VMHCPLoss(self.args)
        self.val_metrics = VMHCPMetric(self.args, prefix = "val/")
        self.train_metric = VMHCPMetric(self.args, prefix = "train/")

        # unet architecture
        enc_nf = self.args.enc if self.args.enc else [16, 32, 32, 32]
        dec_nf = self.args.dec if self.args.dec else [32, 32, 32, 32, 32, 16, 16]

        self.model = networks.VxmDense(
            inshape=self.args.inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=self.args.bidir,
            int_steps=self.args.int_steps,
            int_downsize=self.args.int_downsize
        )

    def forward(self, input, registration=False):
        if isinstance(input,list):
            input = input[0]

        moving = None; fixed = None

        if self.args.variant == "LR":
            fixed = input["T1w"]
            moving = input["LR"]
        elif self.args.variant == "RL":
            fixed = input["T1w"]
            moving = input["RL"]
        elif self.args.variant == "LR+RL": # radomly decide this is batch is LR or RL
            fixed = input["T1w"]
            moving = input["RL"] if random.random() < 0.5 else input["LR"]
        elif self.args.variant == "concat-LR+RL": # both LR and RL present when do the registration
            fixed = torch.cat([input["T1w"],input["T1w"]], dim=1) # assume BCHWD
            moving = torch.cat([input["LR"],input["RL"]], dim=1)
        else:
            raise ValueError(f"Unknown args.variant {self.args.variant}")

        warped, flow = self.model(moving,fixed, registration = registration)

        output = {}
        output["warped"] = warped
        output["flow"] = flow
        return output

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--VMHCP_root_dir', type=str,  default="../../data/cache/data.json")

        parser.add_argument('--bidir', type=int,  default=0)
        parser.add_argument('--pixdim', type=float, default=2.0)
        parser.add_argument('--int-steps', type=int, default=7,
                            help='number of integration steps (default: 7)')
        parser.add_argument('--int-downsize', type=int, default=2,
                            help='flow downsample factor for integration (default: 2)')
        parser.add_argument('--image-loss', default='ncc',
                            help='image reconstruction loss - can be mse or ncc (default: mse)')
        parser.add_argument('--enc', type=int, nargs='+',
                            help='list of unet encoder filters (default: 16 32 32 32)')
        parser.add_argument('--dec', type=int, nargs='+',
                            help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
        parser.add_argument('--inshape', type=int, nargs='+', default=[192,192,192],
                            help='for pixdim = 1.25 -> 144 176 176; for pixdim = 2 -> 96 112 112')

        parser.add_argument('--variant', type=str, default="LR",
                            help='model variants')

        parser.add_argument('--NCC_win', type=int, default=9,
                            help='local window for computing ncc metric, when inshape=192 -> win=9')
        parser.add_argument('--w_sm', type=float, default=1.0)
        # inshape

        return parser
