from typing import Optional, Union

from srmt.model import EncoderConfig, CoreConfig
from srmt.preprocessing import PreprocessorConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig
from pydantic import BaseModel



class DecMAPFConfig(GridConfig):
    integration: Literal['SampleFactory', 'TMazeSampleFactory'] = 'SampleFactory'
    on_target: Literal['finish', 'restart', 'nothing', 't-maze'] = 'finish'
    collision_system: Literal['priority', 'block_both', 'soft'] = 'block_both'
    observation_type: Literal['POMAPF', 'MAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 2
    obs_radius: int = 2
    max_episode_steps: int = 512
    map_name: str = '(bottlenecks9-v-train-330-|wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
    non_random_possible_targets: bool = False
    #size: int = 8
    seed: Optional[int] = None


class Environment(BaseModel, ):
    grid_config: DecMAPFConfig = DecMAPFConfig()
    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaTMaze-v0', "PogemaNone-v0", 'PogemaPogemaTMaze-v0'] = "PogemaNone-v0"
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = [64, 128, 256, 256]
    use_maps: bool = True

    every_step_metrics: bool = False

    maze_length: int = None
    intra_num_agents: int = None


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
    encoder_mlp_layers = [16]
    decoder_mlp_layers = []
    


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
                                                           any_move_reward=False,
                                                           meta_agent=False,
                                                          )

    
    wandb_project: str = 'srmt'
    num_transformer_layers: int = 1
    attn_core: bool = True
    core_memory: bool = True
    rate_memory: bool = False
    relational_rnn: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False
    rollout: int = 8
    num_workers: int = 4
    
    recurrence: int = 1 #8

    mem_recurrence: int = 2
    
    use_rnn: bool = False
    rnn_size: int = 512

    actor_critic_share_weights: bool = True

    ppo_clip_ratio: float = 0.2
    batch_size: int = 16384

    exploration_loss_coeff: float = 0.023
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    
    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "resume" 
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 4

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "avg_throughput"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 1800

    keep_checkpoints: int = 5
    stats_avg: int = 10
    learning_rate: float = 0.00022
    train_for_env_steps: int = 1_000_000_000

    gamma: float = 0.9756

    lr_schedule: str = 'constant' #'kl_adaptive_minibatch'

    experiment: str = 'exp'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = False

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaTMaze-v0', 'PogemaPogemaTMaze-v0'] = "PogemaMazes-v0"

    serial_mode: bool = False
    decorrelate_envs_on_one_worker: bool = True
    stats_avg: int = 100
    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = False
    pbt_period_env_steps: int = 5_000_000
    pbt_start_mutation: int = 10_000_000
    num_policies: int = 1
    pbt_replace_fraction: float = 0.5


class EnvironmentBtlnck(Environment):
    env: Literal['PogemaBtlnck-v0'] = "PogemaBtlnck-v0"
    use_maps: bool = True 
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = None 
    grid_config: DecMAPFConfig = DecMAPFConfig(on_target='finish',
                                               max_episode_steps=64,
                                               map_name=r'bottlenecks9-v-train-330-8', num_agents=2,
                                               obs_radius=2,
                                               collision_system='block_both', 
                                              )


class ExperimentBtlnck(Experiment):
    environment: EnvironmentBtlnck = EnvironmentBtlnck() #
    encoder: EncoderConfig = EncoderConfig(extra_fc_layers=1,
                                           hidden_size=16,
                                           num_filters=8,
                                           num_res_blocks=1,
                                          )
    batch_size: int = 16384

    encoder_mlp_layers = [16]
    decoder_mlp_layers = []
    
    core: CoreConfig = CoreConfig(core_hidden_size=16, 
                                  num_attention_heads=1, #4,
                                  max_position_embeddings=16384,
                                 )
    #turn off all the planning heuristics, use regular unit cost for each cell
    preprocessing: PreprocessorConfig = PreprocessorConfig(use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           intrinsic_target_reward=0.01,
                                                           network_input_radius=2,
                                                           anontargets=True,
                                                           target_reward=True,
                                                           reversed_reward=False,
                                                           const_reward=False,
                                                           positive_reward=False,
                                                           any_move_reward=False
                                                          )
    wandb_project: str = 'srmt' 
    num_transformer_layers: int = 1
    attn_core: bool = True
    core_memory: bool = True
    rate_memory: bool = False
    relational_rnn: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    
    actor_critic_share_weights: bool = True
    
    rollout: int = 32
    num_workers: int = 4

    mem_recurrence: int = 2

    recurrence: int = 2
    rnn_size: int = 16
    use_rnn: bool = False
    
    ppo_clip_ratio: float = 0.2
    

    exploration_loss_coeff: float = 0.0156
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    
    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "resume" 
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 16

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "reward"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 600

    keep_checkpoints: int = 5
    stats_avg: int = 10
    learning_rate: float = 0.00013
    train_for_env_steps: int = 30_000_000 

    gamma: float = 0.99#716

    lr_schedule: str = 'kl_adaptive_minibatch'

    experiment: str = 'exp_btlnck'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = False

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0'] = 'PogemaBtlnck-v0' 

    serial_mode: bool = False
    decorrelate_envs_on_one_worker: bool = False
    stats_avg: int = 10

    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = False
    pbt_period_env_steps: int = 5_000_000
    pbt_start_mutation: int = 10_000_000
    
    num_policies: int = 1
    pbt_replace_fraction: float = 0.5



class EnvironmentTMaze(Environment):
    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaTMaze-v0'] = "PogemaTMaze-v0"
    maze_length: int = 10
    intra_num_agents: int = 2
    
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = [64, 128, 256, 256]
    use_maps: bool = False

    every_step_metrics: bool = False

    grid_config: DecMAPFConfig = DecMAPFConfig(num_agents=2) #bool = False
    


class ExperimentTMaze(Experiment):
    
    environment: EnvironmentTMaze = EnvironmentTMaze(maze_length=16, intra_num_agents=1 #2
                                                     ) 
    encoder: EncoderConfig = EncoderConfig(extra_fc_layers=1,
                                           hidden_size=32,
                                           num_filters=8,
                                           num_res_blocks=1,
                                          )
    encoder_mlp_layers = [32]
    decoder_mlp_layers = [32]
    batch_size: int = 128
    
    
    core: CoreConfig = CoreConfig(core_hidden_size=32,
                                  num_attention_heads=4,
                                  max_position_embeddings=16384,
                                 )
    preprocessing: PreprocessorConfig = PreprocessorConfig(use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           intrinsic_target_reward=0.,
                                                           network_input_radius=0.,
                                                           anontargets=False,
                                                           target_reward=False,
                                                           reversed_reward=False,
                                                           const_reward=False,
                                                           positive_reward=False,
                                                           any_move_reward=False,
                                                          )

    wandb_project: str = ''
    num_transformer_layers: int = 1
    attn_core: bool = True
    core_memory: bool = True
    rate_memory: bool = False
    relational_rnn: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    
    actor_critic_share_weights: bool = True
    
    rollout: int = 8
    num_workers: int = 4

    mem_recurrence: int = 8

    recurrence: int = 8
    rnn_size: int = 32
    use_rnn: bool = False
    
    ppo_clip_ratio: float = 0.2
    

    exploration_loss_coeff: float = 1e-2
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    
    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "resume"  
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 10

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "reward"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 3600

    keep_checkpoints: int = 1_000_000
    stats_avg: int = 10
    learning_rate: float = 1e-3
    train_for_env_steps: int = 2_000_000 

    gamma: float = 0.99

    lr_schedule: str = 'kl_adaptive_minibatch'

    experiment: str = 'exp_tmaze'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = False

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaTMaze-v0'] = 'PogemaTMaze-v0' #

    serial_mode: bool = False
    decorrelate_envs_on_one_worker: bool = False
    stats_avg: int = 10

    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = False
    pbt_period_env_steps: int = 5_000_000
    pbt_start_mutation: int = 10_000_000
    
    num_policies: int = 1
    pbt_replace_fraction: float = 0.5


class EnvironmentPogemaTMaze(Environment):
    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaTMaze-v0', 'PogemaPogemaTMaze-v0'] = "PogemaPogemaTMaze-v0"
    maze_length: int = 16
    use_maps: bool = True # train on multiple maps from yaml file
    target_num_agents: Optional[int] = None #2
    agent_bins: Optional[list] = None 
    grid_config: DecMAPFConfig = DecMAPFConfig(on_target='t-maze',
                                               max_episode_steps=16,
                                               map_name=r't-maze-.+', num_agents=2,
                                               obs_radius=1,
                                               collision_system='block_both', 
                                              )


class ExperimentPogemaTMaze(Experiment):
    environment: EnvironmentPogemaTMaze = EnvironmentPogemaTMaze() #
    encoder: EncoderConfig = EncoderConfig(extra_fc_layers=1,
                                           hidden_size=32,
                                           num_filters=8,
                                           num_res_blocks=1,
                                          )
    encoder_mlp_layers = [32]
    decoder_mlp_layers = [32]
    batch_size: int = 128
    
    core: CoreConfig = CoreConfig(core_hidden_size=32, 
                                  num_attention_heads=1,
                                  max_position_embeddings=16384,
                                 )
    #turn off all the planning heuristics, use regular unit cost for each cell
    preprocessing: PreprocessorConfig = PreprocessorConfig(use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           intrinsic_target_reward=0.,
                                                           network_input_radius=1,
                                                           anontargets=True,
                                                           target_reward=False,
                                                           reversed_reward=False,
                                                           const_reward=False,
                                                           positive_reward=False,
                                                           any_move_reward=False
                                                          )
    wandb_project: str = ''
    num_transformer_layers: int = 1
    attn_core: bool = True
    core_memory: bool = True
    rate_memory: bool = False
    relational_rnn: bool = False
    use_global_memory: bool = True
    action_hist: bool = False
    clear_memory: bool = False

    
    actor_critic_share_weights: bool = True
    
    rollout: int = 8
    num_workers: int = 4

    mem_recurrence: int = 2

    recurrence: int = 2
    rnn_size: int = 32
    use_rnn: bool = False
    
    ppo_clip_ratio: float = 0.2
    

    exploration_loss_coeff: float = 1e-2
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    
    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "resume"
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 10

    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers = []
    save_best_metric: str = "reward"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1
    save_every_sec: int = 3600

    keep_checkpoints: int = 5
    stats_avg: int = 10
    learning_rate: float = 1e-3
    train_for_env_steps: int = 100_000_000

    gamma: float = 0.99

    lr_schedule: str = 'kl_adaptive_minibatch'

    experiment: str = 'exp_pogematmaze'
    train_dir: str = 'experiments/train_dir'
    seed: Optional[int] = 42
    use_wandb: bool = False

    env: Literal['PogemaMazes-v0', 'PogemaBtlnck-v0', "PogemaRandom-v0", 'PogemaPogemaTMaze-v0'] = 'PogemaPogemaTMaze-v0' #


    serial_mode: bool = False
    decorrelate_envs_on_one_worker: bool = False
    stats_avg: int = 10

    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = False
    pbt_period_env_steps: int = 5_000_000
    pbt_start_mutation: int = 10_000_000
    
    num_policies: int = 1
    pbt_replace_fraction: float = 0.5


