import numpy as np
from sample_factory.utils.typing import Env
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.attr_dict import AttrDict

from env.create_env import create_env_base, create_pogematmaze_env
from env.tmaze_env import create_tmaze_env
from follower.training_config import Experiment

import gymnasium
from follower.training_config import Environment, EnvironmentTMaze, EnvironmentPogemaTMaze
from follower.preprocessing import PreprocessorConfig, wrap_preprocessors, wrap_pogematmaze_preprocessors


def create_env(environment_cfg: Environment, preprocessing_cfg: PreprocessorConfig):
    env = create_env_base(environment_cfg)
    env = wrap_preprocessors(env, config=preprocessing_cfg, auto_reset=True)
    return env

def create_register_env_tmaze(environment_cfg: EnvironmentTMaze):
    env = create_tmaze_env(environment_cfg)
    return env

def create_register_env_pogematmaze(environment_cfg: EnvironmentPogemaTMaze, preprocessing_cfg: PreprocessorConfig):
    env = create_pogematmaze_env(environment_cfg)
    env = wrap_pogematmaze_preprocessors(env, config=preprocessing_cfg, auto_reset=True)
    return env


class MultiEnv(gymnasium.Wrapper):
    def __init__(self, env_cfg: Environment, preprocessing_cfg: PreprocessorConfig):
        if env_cfg.target_num_agents is None:
            self.envs = [create_env(env_cfg, preprocessing_cfg)]
        else:
            assert env_cfg.target_num_agents % env_cfg.grid_config.num_agents == 0, \
                f"Target num follower must be divisible by num agents: env_cfg.target_num_agents = {env_cfg.target_num_agents}, env_cfg.grid_config.num_agents = {env_cfg.grid_config.num_agents}"
            num_envs = env_cfg.target_num_agents // env_cfg.grid_config.num_agents
            self.envs = [create_env(env_cfg, preprocessing_cfg) for _ in range(num_envs)]
        super().__init__(self.envs[0])

    def step(self, actions):
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        last_agents = 0
        for env in self.envs:
            # here we build a single env with huge num of agents and then internally slice the huge action list into sets of actions
            # for smaller groups of agents like they were independant envs with small num of agents
            # this approach does not take into account interaction and mutual influence of agents of huge env that belong to different small envs
            # to truly separate huge env into smaller ones we must localize and separate areas of possible movements for each small env 
            # for them to not interfere
            env_num_agents = env.get_num_agents()
            action = actions[last_agents: last_agents + env_num_agents]
            last_agents = last_agents + env_num_agents
            o, r, d, t, i = env.step(action)
            obs += o
            rewards += r
            dones += d
            truncated += t
            infos += i
        return obs, rewards, dones, truncated, infos

    def reset(self, seed, **kwargs):
        obs = []
        for idx, env in enumerate(self.envs):
            inner_seed = seed + idx
            o, _ = env.reset(seed=inner_seed, **kwargs)
            obs += o
        return obs, {}

    def sample_actions(self):
        actions = []
        for env in self.envs:
            actions += list(env.sample_actions())
        return np.array(actions)

    @property
    def num_agents(self):
        return sum([env.get_num_agents() for env in self.envs])

    def render(self):
        for q in self.envs:
            q.render()

def make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    if isinstance(cfg, AttrDict):
        zz = dict(cfg)
    else:
        zz = vars(cfg)
    p_config = Experiment(**zz)
    assert len(zz) != 0, f"follower/register_env: p_config {zz}"
    
    print(f"env config after {p_config.environment}")
    environment_config = p_config.environment
    preprocessing_config = p_config.preprocessing
    
    if full_env_name == "PogemaTMaze-v0":
        return create_register_env_tmaze(environment_config)
    if full_env_name == "PogemaPogemaTMaze-v0":
        return create_register_env_pogematmaze(environment_config, preprocessing_config)

    if environment_config.agent_bins is not None and environment_config.target_num_agents is not None:
        env_id = env_config['env_id'] if env_config is not None else None
        
        if env_id is None:
            num_agents = environment_config.agent_bins[0]
        else:
            num_agents = environment_config.agent_bins[env_id % len(environment_config.agent_bins)]
            
        environment_config.grid_config.num_agents = num_agents
        return MultiEnv(environment_config, preprocessing_config)
    return create_env(environment_config, preprocessing_config)


class CustomEnv:
    def make_env(self, env_name, cfg, env_config, render_mode) -> Env:
        return make_env(env_name, cfg, env_config, render_mode)


def register_pogema_envs(env_name):
    env_factory = CustomEnv()
    register_env(env_name, env_factory.make_env)


def register_custom_components(env_name):
    register_pogema_envs(env_name)
