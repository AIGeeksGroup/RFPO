import os
import sys
import hydra
import pathlib
import random
import pickle
import json
import torch
import copy
import argparse
import yaml
import wandb
import tqdm
import time
import threading
import numpy as np
import _init_paths
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from dataset.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from imitate_train import TrainWorkspace
from algos.imitate.env_runner.base_runner import BaseRunner
from algos.imitate.common.checkpoint_util import TopKCheckpointManager
from algos.imitate.common.pytorch_util import dict_apply, optimizer_to
from algos.imitate.model.diffusion.ema_model import EMAModel
from algos.imitate.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath('algos', 'imitate', 'config')))
def main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
