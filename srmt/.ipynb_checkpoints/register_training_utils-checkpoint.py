import numpy as np

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import ObsSpace
from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.utils import log
from tensorboardX import SummaryWriter

from srmt.model import ResnetEncoder, TransformerCore


def pogema_extra_episodic_stats_processing(*args, **kwargs):
    pass


def pogema_extra_summaries(runner: Runner, policy_id: PolicyID, summary_writer: SummaryWriter, env_steps: int):
    policy_avg_stats = runner.policy_avg_stats
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


class CustomExtraSummariesObserver(AlgoObserver):
    def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int) -> None:
        pogema_extra_summaries(runner, policy_id, writer, env_steps)


def register_msg_handlers(cfg: Config, runner: Runner):
    runner.register_episodic_stats_handler(pogema_extra_episodic_stats_processing)
    runner.register_observer(CustomExtraSummariesObserver())


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return ResnetEncoder(cfg, obs_space)


def make_custom_core(cfg: Config, input_size: int) -> ModelCore:
    """Factory function as required by the API."""
    return TransformerCore(cfg, input_size)


def register_custom_model():
    global_model_factory().register_encoder_factory(make_custom_encoder)

def register_custom_core():
    print("register_custom_core")
    global_model_factory().register_model_core_factory(make_custom_core)
