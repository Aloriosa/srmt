import torch

from typing import Literal
from argparse import Namespace
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.utils import log
from transformers import GPT2Config
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

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        x = x['obs']
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)
        return x


class CoreConfig(BaseModel):
    num_attention_heads: int = 8
    core_hidden_size: int = 512
    mem: bool = True
    max_position_embeddings: int = 16384
    add_cross_attention: bool = True


class TransformerCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        self.core_cfg: CoreConfig = CoreConfig(**cfg.core)
        self.use_memory = cfg.core_memory
        self.use_global_memory = cfg.use_global_memory
        self.num_agents = cfg.environment['grid_config']['num_agents']
        core_cfg_copy = cfg.core.copy()
        core_cfg_copy['hidden_size'] = core_cfg_copy.pop('core_hidden_size')
        self.core_transformer = GPT2Block(GPT2Config(**core_cfg_copy))
        self.rnn_placeholder = nn.Linear(self.core_cfg.core_hidden_size, 1, bias=False)
        self.wpe = nn.Embedding(core_cfg_copy['max_position_embeddings'], 
                                self.core_cfg.core_hidden_size)
        if self.use_memory:
            self.mem_head = nn.Linear(self.core_cfg.core_hidden_size, 
                                      self.core_cfg.core_hidden_size, 
                                      bias=False)
        self.ln_f = nn.LayerNorm(self.core_cfg.core_hidden_size, eps=1e-5)
        
    def forward(self, head_output, rnn_states, 
                agent_memory=None, global_memory=None, 
                history_seq=None, **kwargs
               ):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(1)
            if history_seq is not None:
                history_seq = history_seq.unflatten(
                    dim=1, sizes=(-1, self.core_cfg.core_hidden_size)
                )
            first_time_mem = False
            if self.use_memory:
                # first pass with empty memory
                if not agent_memory.abs().sum().is_nonzero():
                    agent_memory = None
                    first_time_mem = True
                else:
                    agent_memory_batch = agent_memory.unsqueeze(1)
                    restored_global_memory = global_memory.unflatten(
                        dim=1, sizes=(-1, self.core_cfg.core_hidden_size)
                        )
        if history_seq is not None:
            inputs = torch.cat([history_seq, head_output], dim=1)
        else:
            inputs = head_output.contiguous()
                
        if agent_memory is not None:
            inputs = torch.cat([agent_memory_batch, inputs], dim=1)
                
        position_ids = torch.arange(0, inputs.size(1), dtype=torch.long).to('cuda')
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs + position_embeds

        encoder_hidden_states = None
        if agent_memory is not None:
            if self.use_global_memory:
                encoder_hidden_states = restored_global_memory.contiguous()
        x = self.core_transformer(hidden_states=hidden_states.contiguous(),
                                  encoder_hidden_states=encoder_hidden_states,
                                 )[0]
        x = self.ln_f(x)
        core_out = x[:,-1:]
        
        if self.use_memory:
            if first_time_mem:
                my_new_mem = core_out.contiguous()
            else:
                my_new_mem, _ = torch.split(x, [1, x.size()[1] - 1], dim=1)
            my_new_mem = self.mem_head(my_new_mem)
        rnn_out_placeholder = self.rnn_placeholder(core_out).squeeze(1)
        
        # update history with current head_output
        if history_seq is not None:
            new_history_seq = torch.cat([history_seq[:, 1:], head_output], dim=1)
            new_history_seq = new_history_seq.flatten(start_dim=1)
        
        if not is_seq:
            core_out = core_out.squeeze(1)
            if self.use_memory:
                my_new_mem = my_new_mem.squeeze(1)
        
        if self.use_memory and history_seq is not None:
            return core_out, rnn_out_placeholder, {'agent_new_memory': my_new_mem, 
                                                   'global_memory': global_memory, 
                                                   'new_history_seq': new_history_seq}
        elif self.use_memory:
            return core_out, rnn_out_placeholder, {'agent_new_memory': my_new_mem, 
                                                   'global_memory': global_memory}
        elif history_seq is not None:
            return core_out, rnn_out_placeholder, {'new_history_seq': new_history_seq}
        else:
            return core_out, rnn_out_placeholder, {}

    def get_out_size(self) -> int:
        return self.core_cfg.core_hidden_size
