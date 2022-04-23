
import torch
import torch.nn as nn
from torchmetrics import Metric
import matplotlib as mpl


mpl.use('Agg') # if you are on a headless machine


import ants
from src.lib.losses import NCC

class VMHCPMetric(Metric):
    def __init__(self, args, prefix = "", dist_sync_on_step=False, logger = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.args = args
        self.logger = logger
        self.prefix = prefix
        self.step = 0
        self.ncc_func = NCC(9).loss

        # self.add_state("mse_b0", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("MI", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("NCC", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")


    def set_logger(self,logger):
        self.logger = logger

    def update(self, preds, target):
        if isinstance(target,list):
            target = target[0]
        warped = preds["warped"]
        B = warped.shape[0]

        # self.mse_b0 += nn.functional.mse_loss(warped.detach(), target["b0_star"])
        for i in range(B):
            fixed = ants.from_numpy(target["T1w"][i,0].cpu().numpy())
            moving = ants.from_numpy(preds["warped"][i,0].detach().cpu().numpy())
            self.MI += ants.image_mutual_information(fixed, moving)

        self.NCC += self.ncc_func(preds["warped"].detach(), target["T1w"]) * B

        self.total += B


    def compute(self):
        metrics = {}
        # metrics[f"{self.prefix}mse_b0"] = self.mse_b0 / self.total
        metrics[f"{self.prefix}MI"] = self.MI / self.total
        metrics[f"{self.prefix}NCC"] = self.NCC / self.total

        return metrics
