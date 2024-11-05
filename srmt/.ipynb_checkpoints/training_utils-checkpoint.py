import json
import yaml
import wandb
from argparse import Namespace

from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.train import make_runner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.utils import log

from srmt.training_config import Experiment
from srmt.register_env import register_custom_components
from srmt.register_training_utils import register_custom_model, register_msg_handlers, register_custom_core


def create_sf_config(exp: Experiment):
    custom_argv = [f'--env={exp.env}']
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    parser.set_defaults(**exp.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    return final_cfg


def run(config=None):
    
    register_custom_model()
    
    if config is None:
        import argparse

        parser = argparse.ArgumentParser(description='Process training config.')

        parser.add_argument('--config_path', type=str, action="store", default='train-debug.yaml',
                            help='path to yaml file with single run configuration', required=False)

        parser.add_argument('--raw_config', type=str, action='store',
                            help='raw json config', required=False)

        parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                            help='Run wandb in thread mode. Usefull for some setups.', required=False)

        params = parser.parse_args()
        if params.raw_config:
            params.raw_config = params.raw_config.replace("\'", "\"")
            config = json.loads(params.raw_config)
        else:
            if params.config_path is None:
                raise ValueError("You should specify --config_path or --raw_config argument!")
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)
    else:
        params = Namespace(**config)
        params.wandb_thread_mode = False

    if config.get("attn_core", None).lower() == "true":
        register_custom_core()
    exp = Experiment(**config)
    flat_config = Namespace(**exp.dict())
    env_name = exp.environment.env
    log.debug(f'env_name = {env_name}')
    register_custom_components(env_name)

    log.info(flat_config)

    if exp.train_for_env_steps == 1_000_000:
        exp.use_wandb = False

    if exp.use_wandb:
        wandb_run_name = exp.experiment
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project='srmt', config=exp.dict(), save_code=False, sync_tensorboard=True,
                   anonymous="allow", job_type=exp.environment.env, group='train', name=wandb_run_name)
    sfconfig = create_sf_config(exp)
    
    flat_config, runner = make_runner(sfconfig)
    register_msg_handlers(flat_config, runner)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    
    return status
