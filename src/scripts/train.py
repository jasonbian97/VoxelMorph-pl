
import torch
torch.manual_seed(1)
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler, SimpleProfiler
from datetime import datetime
import sys
import os

import pathlib
FILE_PATH = pathlib.Path(__file__).parent.resolve()
SRC_Folder = FILE_PATH.parent.resolve()
Proj_folder = SRC_Folder.parent.resolve()
sys.path.append(Proj_folder)
sys.path.append(SRC_Folder)
print("SRC_Folder = ", SRC_Folder)
print("Proj_folder = ", Proj_folder)

from monai.config import print_config

print_config()

from argparse import ArgumentParser, Namespace
from pathlib import Path

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # Workaround until pl stop raising the metrics deprecation warning
    import pytorch_lightning as pl
import torch

from src.lib.utility import get_model, get_model_reference, add_datasets_to_parser

def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'model', type=str,choices = ["voxelmorph"],
        help='Name of the model to use.')
    parser.add_argument(
        '--random_seed', type=int, default=1234,
        help='A number to seed the pseudo-random generators.')
    parser.add_argument(
        '--clear_train_state', action='store_true',
        help=('Only used if --resume_from_checkpoint is not None. If set, only the weights are loaded from the checkpoint '
              'and the training state is ignored. Set it when you want to finetune the model from a previous checkpoint.'))
    parser.add_argument(
        '--log_dir', type=str, default='../results/logs',
        help='The path to the directory where the logs will be saved.')
    # parser.add_argument(
    #     '--proj_path', type=str, default='',
    #     help='The path to the project directory')

    return parser

def _gen_dataset_id(
    dataset_string: str
) -> str:
    sep_datasets = dataset_string.split('+')
    names_list = []
    for dataset in sep_datasets:
        if '*' in dataset:
            tokens = dataset.split('*')
            try:
                _, dataset_params = int(tokens[0]), tokens[1]
            except ValueError:  # the multiplier is at the end
                dataset_params = tokens[0]
        else:
            dataset_params = dataset

        dataset_name = dataset_params.split('-')[0]
        names_list.append(dataset_name)

    dataset_id = '_'.join(names_list)
    return dataset_id

# TODO: Customize
def get_monitor_str(model):
    if model.args.model in ["voxelmorph"]:
        return "val/MI"

# TODO: Customize
def get_mode_str(model):
    if model.args.model in ["voxelmorph"]:
        return "min"

def train(args: Namespace) -> None:
    """Run the training.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the training.
    """
    pl.utilities.seed.seed_everything(args.random_seed)

    if args.train_transform_cuda:
        from torch.multiprocessing import set_start_method
        set_start_method('spawn')

    log_model_name = f'{args.model}-{_gen_dataset_id(args.train_dataset)}'

    model = get_model(args.model, args.pretrained_ckpt, args)

    if args.resume_from_checkpoint is not None and args.clear_train_state:
        # Restore model weights, but not the train state
        pl_ckpt = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(pl_ckpt['state_dict'])
        args.resume_from_checkpoint = None

    # Setup loggers and callbacks

    callbacks = []

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_logger)

    log_model_dir = str(Path(args.log_dir) / log_model_name)
    tb_logger = pl.loggers.CSVLogger("tb_logs",name="my_model")

    model_ckpt_last = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model+'_last_{epoch}', save_weights_only=True)
    callbacks.append(model_ckpt_last)
    model_ckpt_train = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model+'_train_{epoch}')
    callbacks.append(model_ckpt_train)

    # if len(model.val_dataloader_names) > 0:
    model_ckpt_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model+ "-".join(args.val_dataset.split("+")) + '_best_{' + get_monitor_str(model)
                 +':.2f}_{epoch}_{step}', save_weights_only=True,
        save_top_k=1, monitor=get_monitor_str(model),mode=get_mode_str(model))
    callbacks.append(model_ckpt_best)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=callbacks,
        check_val_every_n_epoch = 2,
        num_sanity_val_steps = 0
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Workaround until pl stop the LightningModule.datamodule` property warning
        trainer.fit(model)

if __name__ == '__main__':
    from pytorch_lightning.loggers import CometLogger

    torch.multiprocessing.set_sharing_strategy('file_system')


    parser = _init_parser()

    SRModel = None
    if len(sys.argv) > 1 and sys.argv[1] != '-h' and sys.argv[1] != '--help':
        SRModel = get_model_reference(sys.argv[1])
        parser = SRModel.add_model_specific_args(parser)


    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    train(args)