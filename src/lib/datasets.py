
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
import json
import os
import pathlib
class VMHCPDataset(Dataset):
    def __init__(
            self,
            transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
            data_json: str = "",
            split: str = "", # "train" or "val",
            lim: float = 1.0
    ) -> None:
        self.dataset_name = type(self).__name__
        self.transform = transform
        self.data_json = data_json
        self.split = split
        self.lim = lim

        self.data_seq = []
        with open(self.data_json, "r") as f:
            jdict = json.load(f)

        # TODO: Customize HERE 01
        
        proj_path = pathlib.Path(self.data_json).parent.parent.parent.resolve()
        
        for subid, subdict in jdict.items():
            if subdict["split"] == split:
                self.data_seq.append({
                    "RL": os.path.join(proj_path,subdict["RL2T1w_affine"]),
                    "LR": os.path.join(proj_path,subdict["RL2T1w_affine"]),
                    "T1w": os.path.join(proj_path,subdict["T1w"]),
                    "b0_star": os.path.join(proj_path,subdict["b0_star"]),
                    "meta": {
                        'dataset_name': self.dataset_name,
                         'split': self.split
                    }
                })
        if self.lim < 1.0:
            lim_num = int(len(self.data_seq) * self.lim )
            self.data_seq = self.data_seq[:lim_num]


    def __len__(self) -> int:
        return len(self.data_seq)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, torch.Tensor]:
        inputs = self.transform(self.data_seq[index])

        return inputs








