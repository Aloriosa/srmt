import os
import shutil
import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from srmt.training_config import EnvironmentBtlnck
from srmt.inference import AttnCoreMemInferenceConfig, AttnCoreMemInference
from srmt.preprocessing import follower_preprocessor

from pathlib import Path

if __name__ == '__main__':
    ckpt = 'experiments/train_dir/exp_16'
    path_to_weights = Path(f'experiments/train_dir') / ckpt
    cfg = AttnCoreMemInferenceConfig(
        path_to_weights=str(path_to_weights),
        custom_path_to_weights=None,
        checkpoint_type='best'
    )
    algo = AttnCoreMemInference(cfg)
    print(algo)
