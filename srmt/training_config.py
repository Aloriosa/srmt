from typing import Optional, Union
from pogema import GridConfig
from pydantic import BaseModel

from srmt.model import EncoderConfig, CoreConfig
from srmt.preprocessing import PreprocessorConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DecMAPFConfig(GridConfig):
    integration: Literal['SampleFactory'] = 'SampleFactory'
    on_target: Literal['finish', 'restart', 'nothing'] = 'finish'
    collision_system: Literal['priority', 'block_both', 'soft'] = 'block_both'
    observation_type: Literal['POMAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 2
    obs_radius: int = 2
    max_episode_steps: int = 512
    map_name: str = '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
    non_random_possible_targets: bool = False
    seed: Optional[int] = None


class Environment(BaseModel, ):
    grid_config: DecMAPFConfig = DecMAPFConfig()
    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0"] = "PogemaMazes-v0"
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = None
    use_maps: bool = True
    every_step_metrics: bool = False


class EnvironmentMazes(Environment):
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    use_maps: bool = True
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = None
    grid_config: DecMAPFConfig = DecMAPFConfig(on_target='restart', 
                                               collision_system='soft',
                                               max_episode_steps=512,
                                               map_name=r'mazes-.+',
                                               num_agents=64,
                                               obs_radius=5,
                                              )


class Experiment(BaseModel):
    environment: Environment = EnvironmentMazes()
    encoder: EncoderConfig = EncoderConfig(extra_fc_layers=1,
                                           hidden_size=512,
                                           num_filters=64,
                                           num_res_blocks=8,
                                          )
    
    core: CoreConfig = CoreConfig(core_hidden_size=512, 
                                  num_attention_heads=8,
                                  max_position_embeddings=16384,
                                 )
    preprocessing: PreprocessorConfig = PreprocessorConfig(use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           intrinsic_target_reward=0.01,
                                                           network_input_radius=5,
                                                           anontargets=True,
                                                           target_reward=True,
                                                           reversed_reward=False,
                                                           const_reward=False,
                                                           positive_reward=False,
                                                           any_move_reward=False
                                                          )
    attn_core: bool = False
    core_memory: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    rollout: int = 8
    num_workers: int = 8
    
    recurrence: int = 8
    use_rnn: bool = False
    rnn_size: int = 256

    attn_core: bool = False
    core_memory: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    actor_critic_share_weights: bool = True

    ppo_clip_ratio: float = 0.2
    batch_size: int = 4096

    exploration_loss_coeff: float = 0.023
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    max_policy_lag: int = 1

    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "overwrite"
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 16

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "avg_throughput"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 60

    keep_checkpoints: int = 1_000_000
    stats_avg: int = 10
    learning_rate: float = 0.00022
    train_for_env_steps: int = 4_000_000_000

    gamma: float = 0.9756

    lr_schedule: str = 'constant'

    experiment: str = 'exp'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = True

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0"] = "PogemaMazes-v0"

    serial_mode: bool = False 


class EnvironmentBtlnck(Environment):
    env: Literal['PogemaBtlnck-v0'] = "PogemaBtlnck-v0"
    use_maps: bool = True
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = None 
    grid_config: DecMAPFConfig = DecMAPFConfig(on_target='finish',
                                               max_episode_steps=512,
                                               map_name=r'bottlenecks9-v-train-330-.+',
                                               num_agents=2,
                                               obs_radius=2,
                                               collision_system='block_both',
                                              )


class ExperimentBtlnck(Experiment):
    environment: EnvironmentBtlnck = EnvironmentBtlnck()
    encoder: EncoderConfig = EncoderConfig(extra_fc_layers=1,
                                           hidden_size=16,
                                           num_filters=8,
                                           num_res_blocks=1,
                                          )
    batch_size: int = 16384
    
    core: CoreConfig = CoreConfig(core_hidden_size=16, 
                                  num_attention_heads=4,
                                  max_position_embeddings=16384,
                                 )
    
    preprocessing: PreprocessorConfig = PreprocessorConfig(
                                                           # turn off static and dynamic costs that control Heuristic path planner from the Follower
                                                           use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           # a small per-step reward
                                                           intrinsic_target_reward=0.01,
                                                           # agent observation window radius
                                                           network_input_radius=2,
                                                           # whether to show target locations of other agents in the observation 
                                                           anontargets=True,
                                                           # a unit reward for achieving the goal location
                                                           target_reward=True,
                                                           # use the negative intrinsic_target_reward value to penalize agent steps in the env 
                                                           reversed_reward=False,
                                                           # use the fixed negative reward value for any action 
                                                           const_reward=False,
                                                           # use the positive value for per-step reward 
                                                           positive_reward=False,
                                                           # use negative revard value for movement on the map regardless its direction
                                                           any_move_reward=False
                                                          )
    attn_core: bool = False
    core_memory: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    actor_critic_share_weights: bool = True
    
    rollout: int = 8
    num_workers: int = 4

    recurrence: int = 1
    rnn_size: int = 256
    use_rnn: bool = False
    
    ppo_clip_ratio: float = 0.2

    exploration_loss_coeff: float = 0.0156
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    max_policy_lag: int = 1

    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "overwrite"
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 16

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "CSR"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 60

    keep_checkpoints: int = 1_000_000
    stats_avg: int = 10
    learning_rate: float = 0.00013
    train_for_env_steps: int = 20_000_000

    gamma: float = 0.9716

    lr_schedule: str = 'kl_adaptive_minibatch'

    experiment: str = 'exp_btlnck'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = True

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0'] = 'PogemaBtlnck-v0'

    serial_mode: bool = False
