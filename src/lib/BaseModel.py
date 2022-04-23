import logging
import warnings
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # Workaround until pl stop raising the metrics deprecation warning
    import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from monai.transforms import (
ScaleIntensityRangePercentilesd,
CenterSpatialCropd,
)

from monai.transforms import (
SpatialPadd,
    AddChanneld,
    Compose,
    LoadImaged,
    Spacingd,
    EnsureTyped,
)
from monai.data import CacheDataset
# from RegressBvecBval.src.lib.Losses import XX
# from RegressBvecBval.src.lib.Metrics import XX
from src.lib.datasets import VMHCPDataset
from easydict import EasyDict

class BaseModel(pl.LightningModule):
    """A base abstractmodel."""

    def __init__(
        self,
        args: Union[EasyDict,Namespace,dict],
    ) -> None:

        super(BaseModel, self).__init__()

        self.save_hyperparameters(args)
        self.args = EasyDict(vars(args)) if isinstance(args,Namespace) else EasyDict(args)

        self.train_metric = None
        self.val_metric = None

        self.loss_fn = None


    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[ArgumentParser] = None,
    ) -> ArgumentParser:
        """Generate a parser for the arguments required by this model.

        Parameters
        ----------
        parent_parser : ArgumentParser
            An existing parser, to be extended with the arguments from this model.

        Returns
        -------
        ArgumentParser
            The parser after extending its arguments.

        Notes
        -----
        If the concrete model needs to add more arguments than these defined in this BaseModel, then it should create its
        own method defined as follows:

        >>>
        @staticmethod
        def add_model_specific_args(parent_parser):
            parent_parser = BaseModel.add_model_specific_args(parent_parser)
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument(...add new arguments...)
            return parser
        """

        parents = [parent_parser] if parent_parser is not None else []
        add_help = False if parent_parser is not None else True
        parser = ArgumentParser(parents=parents, add_help=add_help)
        parser.add_argument('--exp_name', type=str, default=None, )
        parser.add_argument('--train_batch_size', type=int, default=2, help='')
        parser.add_argument('--train_num_workers', type=int, default=2, help='')
        parser.add_argument('--val_batch_size', type=int, default=1, help='')
        parser.add_argument('--val_num_workers', type=int, default=1, help='')
        parser.add_argument('--cache_rate', type=float, default=0.0, help='cache part of dataset to accelerate')
        parser.add_argument('--lim_val', type=float, default=1.0, help='val on part of data')
        parser.add_argument('--lim_train', type=float, default=1.0, help='train on part of data')

        parser.add_argument('--train_transform_cuda', action='store_true', default=False, help='')
        parser.add_argument('--train_transform_fp16', action='store_true', default=False, help='')
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--wdecay', type=float, default=1e-5)
        parser.add_argument(
            '--train_dataset', type=str, default=None,)
        # parser.add_argument('--train_crop_size', type=int, nargs=2, default=None, help='')
        parser.add_argument(
            '--val_dataset', type=str, default=None,)
        parser.add_argument(
            '--pretrained_ckpt', type=str, default=None)
        parser.add_argument('--profile', action='store_true', default=False, help='')
        parser.add_argument('--logall', action='store_true', default=False, help='')

        return parser

    @abstractmethod
    def forward(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward the inputs through the network and produce the predictions.

        The method inputs can be anything, up to the implementation of the concrete model. However, the recommended input is
        to receive only dict[str, Any] as argument. This dict should contain everything required for one pass of the network
        (images, etc.). Arguments which do not change for each forward should be defined as arguments in the parser
        (see add_model_specific_args()).

        Parameters
        ----------
        args : Any
            Any arguments.
        kwargs : Any
            Any named arguments.

        Returns
        -------
        Dict[str, torch.Tensor]
            For compatibility with the framework, the output should be a dict containing at least the following keys.
            By default, the output of forward() will be passed to the loss function. So the output may
            include keys which are going to be used for computing the training loss.
        """
        pass

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        """Perform one step of the training.

        Returns
        -------
        Dict[str, Any]
            - 'loss': torch.Tensor, containing the loss value. Required by Pytorch-Lightning for the optimization step.
        """
        preds = self(batch)
        lossd = self.loss_fn(preds, batch)

        loss = lossd['loss']

        metrics_dict = self.train_metric(preds, batch)

        self.log_dict( metrics_dict)
        self.log_dict(lossd)

        outputs = {'loss': loss}

        return outputs

    def training_epoch_end(self, outs):
        self.train_metric.reset()

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Perform one step of the validation.
        """

        preds = self(batch)
        lossd = self.loss_fn(preds, batch)

        self.log_dict({f"val/{key}": val for key,val in lossd.items()}) # log counterpart to the outputs during training

        self.val_metrics.update(preds, batch)

        outputs = {
                   'dataset_name': batch['meta']['dataset_name']}

        return outputs

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> None:
        """Perform operations at the end of one validation epoch.
        This function is called internally by Pytorch-Lightning during validation.
        """
        metric_dict = self.val_metrics.compute()
        self.log_dict(metric_dict)
        self.val_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizers and LR schedulers.

        This function is called internally by Pytorch-Lightning at the beginning of the training.

        Returns
        -------
        Dict[str, Any]
            A dict with two keys:
            - 'optimizer': an optimizer from PyTorch.
            - 'lr_scheduler': Dict['str', Any], a dict with the selected scheduler and its required arguments.
        """

        self.train_dataloader(dummy=True)  # Just to initialize dataloader variables, not use cache dataset

        gpu_divider = self.args.gpus
        if isinstance(gpu_divider, list) or isinstance(gpu_divider, tuple):
            gpu_divider = len(gpu_divider)
        elif isinstance(gpu_divider, str):
            gpu_divider = len(gpu_divider.split(','))
        elif not isinstance(gpu_divider, int):
            gpu_divider = 1

        if self.args.max_steps is None:
            if self.args.max_epochs is None:
                self.args.max_epochs = 10
                logging.warning('--max_epochs is not set. It will be set to %d.', self.args.max_epochs)

            self.args.max_steps = self.args.max_epochs * (self.train_dataloader_length // gpu_divider)

        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)

        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=self.args.lr/4, max_lr=self.args.lr, step_size_up= self.args.max_epochs * self.train_dataloader_length // gpu_divider // 5,
        cycle_momentum=False)

        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=999999)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def train_dataloader(self, dummy=False) -> DataLoader:
        """Initialize and return the training dataloader.

        self.args.train_dataset will be parsed into the selected datasets and their parameters. parse_dataset_selection() is
        used to parse, and the parsed outputs are used as follows: (dataset_multiplier, dataset_name, dataset_params...).

        A method called _get_[dataset_name]_dataset(dataset_params) will be called to get the instance of each dataset.

        Returns
        -------
        DataLoader
            A single dataloader with all the selected training datasets.

        """

        parsed_datasets = self.parse_dataset_selection(self.args.train_dataset)
        train_dataset = None
        for parsed_vals in parsed_datasets:
            dataset_name = parsed_vals[1]
            train_dataset = getattr(self, f'_get_{dataset_name}_dataset')(is_train = True,
                                                                          dummy = dummy,
                                                                          *parsed_vals[2:])

        train_dataloader = DataLoader(
            train_dataset, self.args.train_batch_size, num_workers=self.args.train_num_workers
            , drop_last=False, shuffle=True)
        self.train_dataloader_length = len(train_dataloader)
        return train_dataloader


    def val_dataloader(self) -> Optional[List[DataLoader]]:
        """Initialize and return the list of validation dataloaders.

        Returns
        -------
        Optional[List[DataLoader]]
            A list of dataloaders each for one dataset.

        """
        if self.args.val_dataset.lower() == 'none':
            return None

        parsed_datasets = self.parse_dataset_selection(self.args.val_dataset)
        dataloaders = []
        self.val_dataloader_names = []
        self.val_dataloader_lengths = []
        for parsed_vals in parsed_datasets:
            dataset_name = parsed_vals[1]
            dataset = getattr(self, f'_get_{dataset_name}_dataset')(is_train = False,
                                                                    dummy = False,
                                                                    *parsed_vals[2:])
            dataloaders.append(DataLoader(dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                          num_workers=self.args.val_num_workers, pin_memory=False, drop_last=False))

            self.val_dataloader_names.append('-'.join(parsed_vals[1:]))
            self.val_dataloader_lengths.append(len(dataset))

        return dataloaders


    def parse_dataset_selection(
        self,
        dataset_selection: str,
    ) -> List[Tuple[str, int]]:
        """Parse the input string into the selected dataset and their multipliers and parameters.
        """
        if dataset_selection is None:
            return []

        dataset_selection = dataset_selection.replace(' ', '')
        datasets = dataset_selection.split('+')
        for i in range(len(datasets)):
            tokens = datasets[i].split('*')
            if len(tokens) == 1:
                datasets[i] = (1,) + tuple(tokens[0].split('-'))
            elif len(tokens) == 2:
                try:
                    mult, params = int(tokens[0]), tokens[1]
                except ValueError:
                    params, mult = tokens[0], int(tokens[1])  # if the multiplier comes last.
                datasets[i] = (mult,) + tuple(params.split('-'))
            else:
                raise ValueError(
                    'The specified dataset string {:} is invalid. Check the BaseModel.parse_dataset_selection() documentation '
                    'to see how to write a valid selection string.')
        return datasets

    # TODO: Customize HERE 02
    def _get_VMHCP_dataset(
        self,
        dummy: bool = False,
        is_train: bool = True,
        *args: str
    ) -> Dataset:

        pixdim = self.args.pixdim
        if is_train:
            transform =Compose(
            [
                LoadImaged(keys=["RL","LR","T1w","b0_star"]),
                AddChanneld(keys=["RL","LR","T1w","b0_star"]),
                Spacingd(keys=["RL","LR","T1w","b0_star"],pixdim = (pixdim, pixdim, pixdim), mode = "bilinear",
                         padding_mode = "border"),
                ScaleIntensityRangePercentilesd(keys=["RL", "LR", "T1w", "b0_star"], lower=0.01, upper=99.99, b_min=0,
                                                b_max=1, clip=True),

                CenterSpatialCropd(keys=["RL","LR","T1w","b0_star"],roi_size=self.args.inshape),
                SpatialPadd(keys=["RL","LR","T1w","b0_star"],spatial_size=self.args.inshape, mode="constant"),
                EnsureTyped(keys=["RL","LR","T1w","b0_star"]),
            ]
        )
        else: # val or test
                transform =Compose(
                    [
                        LoadImaged(keys=["RL", "LR", "T1w", "b0_star"]),
                        AddChanneld(keys=["RL", "LR", "T1w", "b0_star"]),
                        Spacingd(keys=["RL", "LR", "T1w", "b0_star"], pixdim=(pixdim, pixdim, pixdim), mode="bilinear",
                                 padding_mode="border"),
                        ScaleIntensityRangePercentilesd(keys=["RL", "LR", "T1w", "b0_star"], lower=0.01, upper=99.99, b_min=0,
                                                        b_max=1, clip=True),

                        CenterSpatialCropd(keys=["RL", "LR", "T1w", "b0_star"], roi_size=self.args.inshape),
                        SpatialPadd(keys=["RL", "LR", "T1w", "b0_star"], spatial_size=self.args.inshape,
                                    mode="constant"),
                        EnsureTyped(keys=["RL", "LR", "T1w", "b0_star"]),
                     ])


        split = "train" if is_train else "val"
        lim = self.args.lim_train if is_train else self.args.lim_val

        dataset = VMHCPDataset(data_json= self.args.VMHCP_root_dir,
                                  transform=transform,
                                  split=split,
                                lim = lim
                                )
        if self.args.cache_rate > 0 and not dummy:
            dataset = CacheDataset(data=dataset.data_seq, transform=transform, cache_rate=self.args.cache_rate)

        return dataset

