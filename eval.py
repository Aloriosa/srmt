import yaml
import os
import json
import wandb
from pathlib import Path

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.evaluator import evaluation, run_views
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results

from env.create_env import create_env_base
from srmt.training_config import EnvironmentBtlnck
from srmt.inference import AttnCoreMemInference, AttnCoreMemInferenceConfig
from srmt.preprocessing import follower_preprocessor

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('experiments')


def main(disable_wandb=True):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, EnvironmentBtlnck)
    ToolboxRegistry.register_algorithm('AttnCoreMem', AttnCoreMemInference, 
                                       AttnCoreMemInferenceConfig,
                                       follower_preprocessor)
    with open('env/test-bottlenecks-9-31000.yaml', 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)
    
    folder_names = [
        'dense_rewards',
        'dir_neg_rewards',
        'sparse_rewards',
        'directional_rewards',
        'mov_neg_rewards'
    ]
    
    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True
        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
