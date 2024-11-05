import os
import json
import numpy as np
import torch

from copy import deepcopy
from argparse import Namespace
from collections import OrderedDict
from pogema_toolbox.algorithm_config import AlgoBase
from sample_factory.utils.utils import log
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from utils import fix_num_threads_issue
from srmt.preprocessing import PreprocessorConfig
from srmt.register_env import register_custom_components
from srmt.training_config import Experiment, ExperimentBtlnck
from srmt.register_training_utils import register_custom_model, register_custom_core

from pydantic import Extra, validator
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AttnCoreMemInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['AttnCoreMem'] = 'AttnCoreMem'
    path_to_weights: str = ""
    preprocessing: PreprocessorConfig = PreprocessorConfig(use_static_cost=False,
                                                           use_dynamic_cost=False,
                                                           reset_dynamic_cost=False,
                                                           intrinsic_target_reward=0.01,
                                                           network_input_radius=2,
                                                           anontargets=True,
                                                          )
    override_config: Optional[dict] = None
    training_config: Optional[Experiment] = None
    custom_path_to_weights: Optional[str] = None
    checkpoint_type: Literal['best', 'latest'] = 'latest'

    @classmethod
    def recursive_dict_update(cls, original_dict, update_dict):
        for key, value in update_dict.items():
            if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
                cls.recursive_dict_update(original_dict[key], value)
            else:
                if key not in original_dict:
                    raise ValueError(f"Key '{key}' does not exist in the original training config.")
                original_dict[key] = value

    @validator('training_config', always=True, pre=True)
    def load_training_config(cls, _, values, ):
        with open(os.path.join(values['path_to_weights'], 'config.json'), "r") as f:
            field_value = json.load(f)
        if values.get('override_config') is not None:
            cls.recursive_dict_update(field_value, deepcopy(values['override_config']))
        return field_value


class AttnCoreMemInference:
    def __init__(self, config):
        self.algo_cfg: AttnCoreMemInferenceConfig = config
        device = config.device

        register_custom_model()
        if getattr(config.training_config, 'attn_core', False):
            register_custom_core()

        self.path = config.path_to_weights

        with open(os.path.join(self.path, 'config.json'), "r") as f:
            flat_config = json.load(f)
            self.exp = ExperimentBtlnck(**flat_config)
            flat_config = Namespace(**flat_config)
        env_name = self.exp.environment.env
        register_custom_components(env_name)
        config = flat_config

        config.num_envs = 1
        env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, 
                                                                vector_index=0, 
                                                                env_id=0))
        self.env_num_agents = env.num_agents
        
        actor_critic = create_actor_critic(config, 
                                           env.observation_space, 
                                           env.action_space)
        actor_critic.eval()
        env.close()

        if device != 'cpu' and not torch.cuda.is_available():
            os.environ['OMP_NUM_THREADS'] = str(1)
            os.environ['MKL_NUM_THREADS'] = str(1)
            device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            log.warning('CUDA is not available, using CPU. This might be slow.')

        actor_critic.model_to_device(device)

        if self.algo_cfg.custom_path_to_weights:
            checkpoints = [self.algo_cfg.custom_path_to_weights]

        else:
            name_prefix = dict(latest="checkpoint", best="best")[self.algo_cfg.checkpoint_type]
            policy_index = 0 if 'policy_index' not in flat_config else flat_config.policy_index
            checkpoints = Learner.get_checkpoints(
                os.path.join(self.path, f"checkpoint_p{policy_index}"), f"{name_prefix}_*")
        
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        
        self.net = actor_critic
        self.device = device
        self.cfg = config
        self.rnn_states = None
        self.agent_memory = None
        self.global_memory = None
        self.history_seq = None

    def act(self, observations, custom_num_agents=None, infos=None):
        if custom_num_agents is not None:
            self.env_num_agents = custom_num_agents
        self.rnn_states = torch.zeros(
            [len(observations), get_rnn_size(self.cfg)], dtype=torch.float32,
            device=self.device) if self.rnn_states is None else self.rnn_states
        
        if self.cfg.core_memory:
            self.agent_memory = torch.zeros(
                [len(observations),self.cfg.core['core_hidden_size']],
                dtype=torch.float32, device=self.device) if self.agent_memory is None else self.agent_memory
            self.global_memory = torch.zeros(
                [len(observations), self.env_num_agents * self.cfg.core['core_hidden_size']],
                dtype=torch.float32, device=self.device) if self.global_memory is None else self.global_memory

        if self.cfg.attn_core:
            self.history_seq = torch.zeros(
                [len(observations), self.cfg.rollout * self.cfg.core['core_hidden_size']],
                dtype=torch.float32, device=self.device) if self.history_seq is None else self.history_seq

        obs = AttrDict(self.transform_dict_observations(observations))
        with torch.no_grad():
            policy_outputs = self.net(prepare_and_normalize_obs(self.net, obs), 
                                      self.rnn_states,
                                      agent_memory=self.agent_memory, 
                                      global_memory=self.global_memory, 
                                      history_seq=self.history_seq,
                                     )
        self.rnn_states = policy_outputs['new_rnn_states']
        # update memory and history_seq
        if self.cfg.core_memory:
            if infos is not None:
                for ag_idx, ag_infos in enumerate(infos):
                    if ag_infos['is_active']:
                        self.agent_memory[ag_idx] = policy_outputs["agent_new_memory"][ag_idx]
                self.global_memory = self.agent_memory.contiguous().flatten().repeat((self.env_num_agents,1))

            else:
                if self.cfg.core_memory:
                    self.agent_memory = policy_outputs["agent_new_memory"]
                    self.global_memory = policy_outputs["agent_new_memory"].contiguous().flatten().repeat((self.env_num_agents,1))
                
        if self.cfg.attn_core:
            self.history_seq = policy_outputs["new_history_seq"]
        
        policy_outputs_cpu = dict()
        for kk, vv in policy_outputs.items():
            policy_outputs_cpu[kk] = policy_outputs[kk].cpu().numpy()
        return policy_outputs['actions'].cpu().numpy(), policy_outputs_cpu

    def reset_states(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None
        self.agent_memory = None
        self.global_memory = None
        self.history_seq = None

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_parameters(self):
        return self.count_parameters(self.net)

    @staticmethod
    def transform_dict_observations(observations):
        """Transform list of dict observations into a dict of lists."""
        obs_dict = dict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict

    def to_onnx(self, filename='srmt.onnx'):
        self.net.eval()
        r = self.algo_cfg.training_config.preprocessing.network_input_radius
        log.info(f"Saving model with network_input_radius = {r}")
        d = 2 * r + 1
        obs_example = torch.rand(1, 2, d, d, device=self.device)
        rnn_example = torch.rand(1, 1, device=self.device)
        with torch.no_grad():
            q = self.net({'obs': obs_example}, rnn_example)
            print(q)
        input_names = ['obs', 'rnn_state']
        output_names = ['values', 'action_logits', 'log_prob_actions', 'actions', 'new_rnn_states']

        torch.onnx.export(self.net, ({'obs': obs_example}, rnn_example), filename,
                          input_names=input_names, output_names=output_names,
                          export_params=True)
