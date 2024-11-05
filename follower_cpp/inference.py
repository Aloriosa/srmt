from follower.inference import FollowerInferenceConfig
from follower.preprocessing import PreprocessorConfig
# noinspection PyUnresolvedReferences
from utils import fix_num_threads_issue

import multiprocessing
from pathlib import Path

from sample_factory.utils.utils import log

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra


class FollowerConfigCPP(FollowerInferenceConfig, extra=Extra.forbid):
    name: Literal['FollowerLite'] = 'FollowerLite'
    num_process: int = 8
    num_threads: int = 8
    path_to_weights: str = "model/follower-lite/"
    preprocessing: PreprocessorConfig = PreprocessorConfig()


class FollowerInferenceCPP:
    def __init__(self, cfg: FollowerConfigCPP):
        self.cfg: FollowerConfigCPP = cfg

        # noinspection PyUnresolvedReferences
        import cppimport.import_hook
        # noinspection PyUnresolvedReferences
        from follower_cpp.follower import Follower
        # noinspection PyUnresolvedReferences
        from follower_cpp.config import Config

        self.algo = Follower()
        self.cpp_config = Config()
        self.cpp_config.obs_radius = self.cfg.training_config.environment.grid_config.obs_radius
        if self.cfg.num_threads > multiprocessing.cpu_count():
            log.warning(f'Setting num_threads to {multiprocessing.cpu_count()}, based on CPU count.')
            self.cfg.num_threads = multiprocessing.cpu_count()
        self.cpp_config.num_threads = self.cfg.num_threads
        self.cpp_config.use_static_cost = self.cfg.training_config.preprocessing.use_static_cost
        self.cpp_config.use_dynamic_cost = self.cfg.training_config.preprocessing.use_dynamic_cost
        self.cpp_config.reset_dynamic_cost = self.cfg.training_config.preprocessing.reset_dynamic_cost
        self.cpp_config.seed = self.cfg.seed

        w_dir = Path(self.cfg.path_to_weights)
        self.cpp_config.path_to_weights = str(w_dir / str(w_dir.name + '.onnx'))

    def full_act(self, episode_length):
        return self.algo.full_act(episode_length)

    def act(self, observations):
        if 'global_obstacles' in observations[0]:
            agents_xy = [obs['global_agent_xy'] for obs in observations]
            self.algo.init(self.cpp_config, observations[0]['global_obstacles'], agents_xy)

        xy = [o['xy'] for o in observations]
        target_xy = [o['target_xy'] for o in observations]
        actions = self.algo.act(xy, target_xy)
        return actions

    def reset_states(self):
        pass
