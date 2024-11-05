import os
import shutil
import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from srmt.training_config import EnvironmentBtlnck
from srmt.inference import AttnCoreMemInferenceConfig, AttnCoreMemInference
from srmt.preprocessing import follower_preprocessor


def create_custom_env(cfg, render_dir='renders', toolbox_env=False):
    env_cfg = EnvironmentBtlnck(with_animation=cfg.animation)
    env_cfg.grid_config.map_name = cfg.map_name
    return create_env_base(env_cfg, render_dir=render_dir)


def run(env, path_to_weights, checkpoint_type, log_path=None, custom_path_to_weights=None):
    cfg = AttnCoreMemInferenceConfig(path_to_weights=path_to_weights,
                                     custom_path_to_weights=custom_path_to_weights,
                                     checkpoint_type=checkpoint_type
                                     )
    algo = AttnCoreMemInference(cfg)
    env = follower_preprocessor(env, cfg)
    return run_episode(env, algo, log_path=log_path)


def main():
    parser = argparse.ArgumentParser(description='Example Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--ckpt', type=str, default='', help='experiment folder name')
    parser.add_argument('--ckpt_file', type=str, default='', help='ckpt file name')
    parser.add_argument('--ckpt_type', type=str, default='latest', help='loading latest or best-score ckpt')
    parser.add_argument('--seed', type=int, default=5, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='bottlenecks9-v-20', 
                        help='Map name (default: %(default)s)')
    args = parser.parse_args()

    if args.ckpt == '':
        print('Set ckpt folder name in <experiments/train_dir> and retry')
        return
    path_to_weights = f'experiments/train_dir/{args.ckpt}'
    if args.ckpt_file == 'latest':
        custom_path_to_weights = None
        ckpt_file_short_name = 'latest'
    elif args.ckpt_file != '':
        custom_path_to_weights = path_to_weights + f"/checkpoint_p0/{args.ckpt_file}.pth"
        ckpt_file_short_name = args.ckpt_file.replace('checkpoint', 'ckpt')
    else:
        custom_path_to_weights = None
        ckpt_file_short_name = ''
    
    print(run(create_custom_env(args, 
                                render_dir=path_to_weights + '/' + ckpt_file_short_name,
                                toolbox_env=True
                               ), 
              path_to_weights, 
              args.ckpt_type,
              log_path=f'{path_to_weights}/episode_log_{args.map_name}_{ckpt_file_short_name}_seed_{args.seed}.npy',
              custom_path_to_weights=custom_path_to_weights
             ))


if __name__ == '__main__':
    main()
