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
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from torch import nn as nn


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


class CoreConfig(BaseModel):
    """
    Configuration for an encoder.

    """
    num_attention_heads: int = 8
    core_hidden_size: int = 512
    mem: bool = True
    max_position_embeddings: int = 16384
    add_cross_attention: bool = True


class TransformerCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        # GPT2Block, GPT2Config
        self.core_cfg: CoreConfig = CoreConfig(**cfg.core)
        
        self.use_memory = cfg.core_memory
        self.use_global_memory = cfg.use_global_memory
        self.num_agents = cfg.environment['grid_config']['num_agents']
        #print(f"TransformerCore cfg.core = {type(cfg.core)}")
        core_cfg_copy = cfg.core.copy()
        core_cfg_copy['hidden_size'] = core_cfg_copy.pop('core_hidden_size')
        self.core_transformer = GPT2Block(GPT2Config(**core_cfg_copy))
        self.rnn_placeholder = nn.Linear(self.core_cfg.core_hidden_size, 1, bias=False)

        self.wpe = nn.Embedding(core_cfg_copy['max_position_embeddings'], 
                                self.core_cfg.core_hidden_size)
        
        #self.re = nn.Embedding(1, self.core_cfg.core_hidden_size)
        # type(env.action_space.n) = 5 but env is inaccesible from model, so hardcoding
        self.ae = nn.Embedding(5, self.core_cfg.core_hidden_size)
        
        if self.use_memory:
            self.mem_head = nn.Linear(self.core_cfg.core_hidden_size, 
                                      self.core_cfg.core_hidden_size, 
                                      bias=False)
        self.ln_f = nn.LayerNorm(self.core_cfg.core_hidden_size, eps=1e-5)


        
    def forward(self, head_output, rnn_states, 
                agent_memory=None, global_memory=None, 
                history_seq=None,
                #reward_seq=None,
                action_seq=None
               ):
        is_seq = not torch.is_tensor(head_output)
        
        # current obs seq len + num_agents steps for mem if mem is enabled
        
        if not is_seq:
            
            history_position_embeds = None
            if history_seq is not None:
                history_seq = history_seq.unflatten(dim=1, sizes=(-1, self.core_cfg.core_hidden_size))
                history_position_ids = torch.arange(self.num_agents * int(agent_memory is not None),
                                                self.num_agents * int(agent_memory is not None) + history_seq.size(1), dtype=torch.long).unsqueeze(0).to('cuda')
                history_position_embeds = self.wpe(history_position_ids)
                history_seq_embeds = history_seq + history_position_embeds
            '''
            if reward_seq is not None:
                reward_seq = reward_seq.unflatten(dim=1, sizes=(-1, 1))
                reward_seq = self.re(reward_seq)
            '''

            head_output = head_output.unsqueeze(1)
            #first take care of history seq, then memory to prepare attn inputs
            head_output_position_id = torch.tensor([[
                self.num_agents * int(agent_memory is not None) + (history_seq.size(1) if history_seq is not None else 0)]], dtype=torch.long).to('cuda')
            head_output_position_embeds = self.wpe(head_output_position_id)
            head_output_embeds = head_output + head_output_position_embeds
        

            first_time_mem = False
            if self.use_memory:
                if not agent_memory.abs().sum().is_nonzero():
                    agent_memory = None
                    first_time_mem = True
                    
            
            # a-la decision transformer input
            ha_seq_embeds = None
            if action_seq is not None and not first_time_mem: 
                #print(f"action_seq = {action_seq.shape}, {action_seq}")
                action_seq_embeds = self.ae(action_seq.long().contiguous())
                if history_position_embeds is not None:
                    action_seq_embeds = action_seq_embeds + history_position_embeds
                else:
                    action_seq_embeds = None
                    print('model: no history_position_embeds provided!!!!')
                #print(f"history_seq_embeds {history_seq_embeds.shape}, action_seq_embeds {action_seq_embeds.shape}, {action_seq_embeds}")
                ha_seq_embeds = torch.stack([history_seq_embeds, action_seq_embeds], dim=2)
                
                ha_seq_embeds = torch.reshape(ha_seq_embeds.contiguous(), 
                                       (ha_seq_embeds.size(0), -1, self.core_cfg.core_hidden_size))
                #print(f"ha_seq new {ha_seq_embeds}")
            else:
                ha_seq_embeds = history_seq_embeds

            restored_global_memory_embeds = None
            agent_memory_batch_embeds = None
            if self.use_memory:
                
                if first_time_mem: #torch.equal(agent_memory[0], agent_memory[1]):
                    print('first_time_memfirst_time_memfirst_time_mem')
                else:
                    
                    restored_global_memory = global_memory.unflatten(dim=1, sizes=(self.num_agents,-1))
                    global_mem_position_ids = torch.arange(0, self.num_agents, 
                                                    dtype=torch.long).unsqueeze(0).to('cuda')
                    global_mem_position_embeds = self.wpe(global_mem_position_ids)
                    restored_global_memory_embeds = restored_global_memory + global_mem_position_embeds

                    
                    agent_memory_batch = agent_memory.unsqueeze(1)
                    # recognize which agent is acting in each sample if batch
                    agent_indices_batch = [None] * agent_memory_batch.size(0)
                    #print(f"init agent_indices_batch = {agent_indices_batch}, {agent_memory_batch.shape}")
                    for i in range(len(agent_indices_batch)):
                        for j in range(restored_global_memory.size(1)):                        
                            if torch.equal(agent_memory[i], restored_global_memory[i, j]):
                                agent_indices_batch[i] = j
                                break
                    assert (None not in agent_indices_batch), f"not all batch samples mem detected {agent_indices_batch}"
                    assert all([i < self.num_agents for i in agent_indices_batch])
                    agent_indices_batch_tensor = torch.stack([torch.tensor([i],dtype=torch.long).to('cuda') for i in agent_indices_batch])
                    
                    agent_indices_batch_tensor_embeds = self.wpe(agent_indices_batch_tensor)
                    
                    agent_memory_batch_embeds = agent_memory_batch + agent_indices_batch_tensor_embeds

                    
                     
        
        if ha_seq_embeds is not None:
            inputs = torch.cat([ha_seq_embeds, head_output_embeds], dim=1)
        else:
            inputs = head_output_embeds
        
        
        if agent_memory_batch_embeds is not None:
            inputs = torch.cat([agent_memory_batch_embeds, inputs], dim=1)
        
        hidden_states = inputs.contiguous()
        
        encoder_hidden_states = None
        if restored_global_memory_embeds is not None:
            if self.use_global_memory:
                encoder_hidden_states = restored_global_memory_embeds.contiguous()
            else:
                print('\n\ncore memory is enabled but global memory is turned off\n\n')


        x = self.core_transformer(hidden_states=hidden_states,
                                  encoder_hidden_states=encoder_hidden_states,
                                 )[0]
        #print('attention done')
        x = self.ln_f(x)
        core_out = x[:,-1:]
        #print(f"core_out {x.shape}")

        if self.use_memory:
            if first_time_mem:
                my_new_mem = core_out.contiguous()
            else:
                my_new_mem, _ = torch.split(x, [1, x.size()[1] - 1], dim=1)
            #print(f"my_new_mem {my_new_mem.shape}")
            my_new_mem = self.mem_head(my_new_mem)
        

        rnn_out_placeholder = self.rnn_placeholder(core_out).squeeze(1)
        


        #print(f"history_seq = {history_seq.shape}")
        # update history with current head_output
        if history_seq is not None:
            new_history_seq = torch.cat([history_seq[:, 1:], head_output], dim=1)
            new_history_seq = new_history_seq.flatten(start_dim=1)
            
        #head_output - bs x hidden_size
        
        if not is_seq:
            core_out = core_out.squeeze(1)
            if self.use_memory:
                my_new_mem = my_new_mem.squeeze(1)

        
        if self.use_memory and history_seq is not None:
            return core_out, rnn_out_placeholder, {'agent_new_memory': my_new_mem, 'global_memory': global_memory, 'new_history_seq': new_history_seq}
        elif self.use_memory:
            return core_out, rnn_out_placeholder, {'agent_new_memory': my_new_mem, 'global_memory': global_memory}
        elif history_seq is not None:
            return core_out, rnn_out_placeholder, {'new_history_seq': new_history_seq}
        else:
            return core_out, rnn_out_placeholder, {}

    def get_out_size(self) -> int:
        return self.core_cfg.core_hidden_size


def main():
    exp_cfg = {'encoder': EncoderConfig().dict()}
    # check mem out
    # exp_cfg['encoder'].update({'mem': True})
    r = 5
    obs = torch.rand(1, 3, r * 2 + 1, r * 2 + 1) # 3 means obstacles, agents, target placements
    # how to combine inputs of different scale in CNN?
    #mem = torch.rand(1, 1, r * 2 + 1, r * 2 + 1) # spatial mem variant as global map representation
    #mem_dim = 128
    #mem = torch.rand(1, mem_dim) # non-spatial mem representation to mix with flatten obs
    
    q_obs = {'obs': obs, 
             #'mem': mem
            }
    # noinspection PyTypeChecker
    re = ResnetEncoder(Namespace(**exp_cfg), dict(obs=obs[0]#, mem=mem[0]
                                                 ))
    re(q_obs)


if __name__ == '__main__':
    main()
