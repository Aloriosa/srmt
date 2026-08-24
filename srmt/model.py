from argparse import Namespace
from typing import Literal

import torch
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements

from sample_factory.utils.utils import log

from transformers import GPT2Config#, BertLayer, BertConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from transformers.activations import NewGELUActivation
from transformers.pytorch_utils import Conv1D
from torch import nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderConfig(BaseModel):
    """
    Configuration for an encoder.

    Args:
        extra_fc_layers (int): Number of extra fully connected (fc) layers. Default is 0.
        num_filters (int): Number of filters. Default is 64.
        num_res_blocks (int): Number of residual blocks. Default is 1.
        activation_func (Literal['ReLU', 'ELU']): Activation function to use. Default is 'ReLU'.
        hidden_size (int): Hidden size for extra fc layers. Default is 128.
    """
    extra_fc_layers: int = 0
    num_filters: int = 64
    num_res_blocks: int = 1
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'
    hidden_size: int = 128
    #mem: bool = False


def activation_func(cfg: EncoderConfig) -> nn.Module:
    """
    Returns an instance of nn.Module representing the activation function specified in the configuration.

    Args:
        cfg (EncoderConfig): Encoder configuration.

    Returns:
        nn.Module: Instance of nn.Module representing the activation function.

    Raises:
        Exception: If the activation function specified in the configuration is unknown.
    """
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    else:
        raise Exception("Unknown activation_func")


class ResBlock(nn.Module):
    """
    Residual block in the encoder.

    Args:
        cfg (EncoderConfig): Encoder configuration.
        input_ch (int): Input channel size.
        output_ch (int): Output channel size.
    """

    def __init__(self, cfg: EncoderConfig, input_ch, output_ch):
        super().__init__()

        layers = [
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    """
    ResNet-based encoder.

    Args:
        cfg (Config): Configuration.
        obs_space (ObsSpace): Observation space.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)

        input_ch = obs_space['obs'].shape[0]
        #print(f"curr_input_channels = {input_ch}, {obs_space['obs'].shape}")
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]
        curr_input_channels = input_ch
        layers = []

        for out_channels, res_blocks in resnet_conf:
            layers.extend([nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)])
            layers.extend([ResBlock(self.encoder_cfg, out_channels, out_channels) for _ in range(res_blocks)])
            curr_input_channels = out_channels
            

        layers.append(activation_func(self.encoder_cfg))
        self.conv_head = nn.Sequential(*layers)
        print(f"obs_space['obs'].shape {obs_space['obs'].shape}")
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        self.encoder_out_size = self.conv_head_out_size

        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size
        '''
        self.mem_head = None
        self.mem_out_size = None
        if self.encoder_cfg.mem:
            # takes obstacles cahnnel from observation
            self.mem_head = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.mem_out_size = self.encoder_cfg.hidden_size
        '''    
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)
        #log.debug('Mem layer output size: %r', self.mem_out_size)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        #print(f"resnet obs {x['obs'].shape}")
        x = x['obs']
        x = self.conv_head(x)
        #print(f"conv {x.shape}")
        x = x.contiguous().view(-1, self.conv_head_out_size)
        #print(f"flatten {x.shape}")
        '''
        mem = None
        if self.encoder_cfg.mem:
            mem = self.mem_linear(x)
        '''
        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)
        '''
        if self.encoder_cfg.mem:
            return (x, mem)
        else:
        '''
        #print(f"encoder out {x.shape}")
        return x


# this class largely follows the official sonnet implementation
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py



class CoreConfig(BaseModel):
    """
    Configuration for an encoder.

    """
    num_attention_heads: int = 8
    core_hidden_size: int = 512
    max_position_embeddings: int = 16384
    add_cross_attention: bool = True


idx = 0
global_train_init = False
class TransformerCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        self.num_agents = cfg.environment['grid_config']['num_agents']
        core_cfg_copy = cfg.core.copy()
        self.core_cfg: CoreConfig = CoreConfig(**cfg.core)
        self.use_memory = cfg.core_memory
        self.relational_rnn = cfg.relational_rnn
        self.rate_memory = cfg.rate_memory
        self.use_global_memory = cfg.use_global_memory
        self.mem_recurrence = bool(cfg.mem_recurrence != -1)

        self.layer_size = self.core_cfg.core_hidden_size
        if self.mem_recurrence:
            initial_hidden_values = torch.zeros((self.layer_size)) # torch.normal(0, 1, size=(self.layer_size,)) # 
            self.initial_agent_memory = torch.nn.Parameter(initial_hidden_values, requires_grad=True)
            self.register_parameter(param=self.initial_agent_memory, name='initial_agent_memory')
        else:
            self.initial_agent_memory = None
        
        # for T-Mazes
        '''
        # Self-attention: inp and state attend to each other
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.layer_size, 
            num_heads=4,  # cfg.core['num_attention_heads'],
            batch_first=True
        )
        if self.use_global_memory:
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim=self.layer_size, 
                num_heads=4,  # cfg.core['num_attention_heads'],
                batch_first=True
            )
        self.mask = torch.nn.Transformer().generate_square_subsequent_mask(cfg.rollout + 3)# 
        '''

        # for bottlenecks
        core_cfg_copy['hidden_size'] = core_cfg_copy.pop('core_hidden_size')
        self.core_transformer = nn.ModuleList([

            GPT2Block(GPT2Config(**core_cfg_copy), layer_idx=i) for i in range(cfg.num_transformer_layers)])

        self.wpe = nn.Embedding(core_cfg_copy['max_position_embeddings'], self.layer_size)

        
        self.out_proj = torch.nn.Linear(self.layer_size, self.layer_size)
        self.state_proj = torch.nn.Linear(self.layer_size, self.layer_size)
        self.ln_f = nn.LayerNorm(self.layer_size, eps=1e-5)

        


    def get_out_size(self) -> int:
        return self.core_cfg.core_hidden_size

    def forward(self, head_output, rnn_states=None, 
                agent_memory=None, global_memory=None, 
                history_seq=None,
                reward_seq=None,
                action_seq=None,
                env_agent_buffer_rollout_info=None,
                values_only=False,
                global_state_indices=None,
                custom_num_agents=None
                ):
        if custom_num_agents is not None:
            self.num_agents = custom_num_agents
            
        global idx
        global global_train_init
        inp = head_output
        
        if history_seq is not None: # (bs, seq_len * h_dim * num_agents) # lays everything flattened for the first agent, then for the second etc.
            # i want the unflattened version to be (num_agents, bs, seq_len, h_dim)
            history_seq = history_seq.unflatten(dim=1, sizes=(-1, self.core_cfg.core_hidden_size))
        initial_mem = False
        if self.use_memory:
            if self.mem_recurrence and (not torch.sum(agent_memory.abs()).is_nonzero()):
                batch_size = head_output.shape[0]

                state = self.initial_agent_memory[:]
                state = state.unsqueeze(0).expand(batch_size, -1)


                initial_mem = True
            else:
                state = agent_memory
        else:
            state = None
        
        history_list = history_seq.split(1, dim=1)
        history_list = [i.squeeze(1) for i in history_list]
        
        if self.use_memory:
            seq = torch.stack([state] + history_list + [inp, state], dim=1) # 
        else:
            seq = inp.unsqueeze(1)
            seq = torch.cat([history_seq, seq], dim=1)
        
        def create_global_memory_batch(state, env_agent_buffer_rollout_info, global_state_indices=None):
            if global_state_indices is None:
                env_idx = env_agent_buffer_rollout_info[:, 0]
                agent_idx = env_agent_buffer_rollout_info[:, 1]
                rollout_step = env_agent_buffer_rollout_info[:, 2]

                ear = env_agent_buffer_rollout_info[:, :3]

                is_active_list = env_agent_buffer_rollout_info[:, 3:]
                
                batch_indices_global_mem = [None] * len(env_agent_buffer_rollout_info)
                batch_indices_padding = torch.all(env_agent_buffer_rollout_info <= 0, dim=1).nonzero(as_tuple=True)[0]
                global_state_indices = [None] * len(env_agent_buffer_rollout_info)
                for i in batch_indices_padding:
                    batch_indices_global_mem[i] = torch.stack([torch.zeros_like(state[0]).to(state.device)] * self.num_agents)
                actual_indices = [i for i in range(len(env_agent_buffer_rollout_info)) if i not in batch_indices_padding]
                while len(actual_indices) > 0:
                    idx = actual_indices[0]
                    env = env_idx[idx]
                    step = rollout_step[idx]
                    active_agents_ids = is_active_list[idx].nonzero(as_tuple=True)[0]
                    
                    active_agents_batch_indices = []
                    team_batch_indices = []
                    team_global_mem = []
                    for agent in range(self.num_agents):
                        if agent in active_agents_ids:
                            zz = torch.all(ear[idx:] == torch.tensor([env,
                                                                      agent,
                                                                      step
                                                                     ]).to(ear.device), 
                                                            dim=1
                                          ).nonzero(as_tuple=True)[0]
                            if len(zz) < 1:
                                print(f"looking {[env,agent,step]} in {ear[:]}, {ear.shape}")
                                assert False
                                
                            else:
                                curr_agent_batch_index = idx + zz.min()
                                active_agents_batch_indices.append(curr_agent_batch_index) #[0]
                                team_batch_indices.append(curr_agent_batch_index)
                                team_global_mem.append(state[curr_agent_batch_index]) #[0]
                        else:
                            team_global_mem.append(torch.zeros_like(state[0]).to(state.device))
                            team_batch_indices.append(-1)
                    team_global_mem = torch.stack(team_global_mem)     
                    assert len(active_agents_batch_indices) == len(set(active_agents_batch_indices)), f'indices are repeating: idx {idx}, indices {active_agents_batch_indices}'
                    assert -1 not in team_batch_indices, f"team_batch_indices not full {team_batch_indices}"
                    for i in active_agents_batch_indices:
                        assert i in actual_indices, f"i = {i} from {active_agents_batch_indices} not in actual_indices {actual_indices}, curr idx {idx}, {env_agent_buffer_rollout_info[:idx+3]}, batch_indices_padding {batch_indices_padding}"

                        batch_indices_global_mem[i] = team_global_mem
                        global_state_indices[i] = torch.stack(team_batch_indices)
                        actual_indices.remove(i)

                assert None not in batch_indices_global_mem, f"batch_indices_global_mem has nones {batch_indices_global_mem}"
                assert None not in global_state_indices, f"global_state_indices has nones {global_state_indices}"
                global_memory_batch = torch.stack(batch_indices_global_mem, dim=0).to(state.device)
                
            else:
                global_memory_batch = torch.stack([state[i] for i in global_state_indices], dim=0).to(state.device)
                
            assert global_memory_batch.shape[0] == env_agent_buffer_rollout_info.shape[0], 'global memory batch is not fully done'
            return global_memory_batch, global_state_indices

        
        global_memory_batch = None
        if self.use_global_memory:
            if values_only:
                global_memory_batch = global_memory.unflatten(dim=1, sizes=(-1, self.core_cfg.core_hidden_size)).contiguous()
            else:
                if initial_mem:
                    assert state is not None, 'state is None but calling global_memory_batch_creation'
                    assert global_state_indices is None, 'initial mem but global_state_indices is not None'
                    global_memory_batch, global_state_indices = create_global_memory_batch(state, env_agent_buffer_rollout_info, 
                    global_state_indices=global_state_indices)
                    
                else:
                    global_memory_batch = global_memory.unflatten(dim=1, sizes=(-1, self.core_cfg.core_hidden_size))
                    
            position_ids = torch.arange(0, global_memory_batch.size(1), dtype=torch.long).to(global_memory_batch.device)
            position_ids = position_ids.unsqueeze(0)
            position_embeds = self.wpe(position_ids)
            global_memory_batch = global_memory_batch + position_embeds
    
        # for T-mazes
        '''
        residual = seq 
        attn_output, _ = self.attention(seq, seq, seq, attn_mask=self.mask.to(seq.device))
        attn_output = attn_output + residual
        if self.use_global_memory:
            residual = attn_output
            attn_output, _ = self.cross_attention(query=attn_output,
                                                  key=global_memory_batch, 
                                                  value=global_memory_batch
                                                  )
            attn_output = attn_output + residual
        '''
        # for bottlenecks
        for block in self.core_transformer:
            outputs = block(seq, encoder_hidden_states=global_memory_batch)
            seq = outputs[0]
        attn_output = seq

        attn_output = self.ln_f(attn_output)
        if self.use_memory:
            attended_inp = attn_output[:, -2, :]
            attended_state = attn_output[:, -1, :]
            new_state = self.state_proj(attended_state)
        else:
            attended_inp = attn_output[:, -1, :]
            new_state = head_output
            
        out = self.out_proj(attended_inp)
        
        if history_seq is not None:
            new_history_seq = torch.cat([history_seq[:, 1:], inp.unsqueeze(1)], dim=1)
            new_history_seq = new_history_seq.flatten(start_dim=1)           

        if self.use_global_memory:
            if values_only:
                new_global_memory = global_memory
            else:
                new_global_memory, global_state_indices = create_global_memory_batch(new_state, env_agent_buffer_rollout_info, 
                    global_state_indices=global_state_indices)
                new_global_memory = new_global_memory.flatten(start_dim=1)          
        else:
            new_global_memory = global_memory


        def get_pr(idx_val):
            def pr(*args):
                print("doing backward for new state {}".format(idx_val))
            return pr

        def get_pr_out(idx_val):
            def pr(*args):
                print("doing backward for out {}".format(idx_val))
            return pr
        
        def get_pr_state(idx_val):
            def pr(*args):
                print("doing backward for inp state {}".format(idx_val))
            return pr
        
        def get_pr_initmem(idx_val):
            def pr(*args):
                print("doing backward for init ag mem {}".format(idx_val))
            return pr
        
        def get_pr_gmb(idx_val):
            def pr(*args):
                print("doing backward for global mem batch {}".format(idx_val))
            return pr
        
        def get_pr_gm(idx_val):
            def pr(*args):
                print("doing backward for global mem {}".format(idx_val))
            return pr
        
        def get_pr_ngm(idx_val):
            def pr(*args):
                print("doing backward for new global mem {}".format(idx_val))
            return pr
        
        return out, rnn_states, {'agent_new_memory': new_state,
                                    'new_history_seq': new_history_seq,
                                    'new_global_memory': new_global_memory,
                                    'global_state_indices': global_state_indices
                                    }
