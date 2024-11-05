import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from srmt.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    anontargets: bool = True
    target_reward: bool = True
    reversed_reward: bool = False
    const_reward: bool = False
    positive_reward: bool = False
    any_move_reward: bool = False


def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = FollowerWrapper(env=env, config=config)
    if config.anontargets:
        env = AnonymousTargetsWrapper(env)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class FollowerWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        self.re_plan.update(observations)
        paths = self.re_plan.get_path()
        new_goals = []
        intrinsic_rewards = []

        for k, path in enumerate(paths):
            obs = observations[k]

            if path is None:
                new_goals.append(obs['target_xy'])
                path = []
            else:
                # Check if the agent reached their subgoal from its previous step
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                # Assign an intrinsic reward if conditions are met, otherwise set it to 0.
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                # Select a new target point.
                new_goals.append(path[1])

            # Set obstacle values to -1.0 in the observation.
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # Adding path to the observation, setting path values to +1.0.
            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
        # Update the previous goals and intrinsic rewards for the next step.
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward, action=None):
        for agent_idx, r in enumerate(reward):
            if (self._cfg.any_move_reward == True) and (action is not None):
                if reward[agent_idx] != 1:
                    if action[agent_idx] == 0: # if the agent predicted to stay at current location
                        reward[agent_idx] = -self._cfg.intrinsic_target_reward / 2.
                    else:
                        reward[agent_idx] = -self._cfg.intrinsic_target_reward
            elif self._cfg.const_reward == True:
                    upd_reward = -self._cfg.intrinsic_target_reward
            elif self._cfg.reversed_reward == True:
                if self.intrinsic_reward[agent_idx] == 0:
                    upd_reward = -self._cfg.intrinsic_target_reward
                else:
                    upd_reward = -self._cfg.intrinsic_target_reward / 2.
            elif self._cfg.positive_reward == True:
                upd_reward = self.intrinsic_reward[agent_idx]
            else:
                if self.intrinsic_reward[agent_idx] == 0:
                    upd_reward = -self._cfg.intrinsic_target_reward / 2.
                else:
                    upd_reward = -self.intrinsic_reward[agent_idx]
            if self._cfg.target_reward:
                if reward[agent_idx] != 1:
                    reward[agent_idx] = upd_reward
            else:
                reward[agent_idx] = upd_reward
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward, action=action), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), 
                                               self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class AnonymousTargetsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        full_size = self.grid_config.obs_radius * 2 + 1
        self.observation_space['anonymous_targets'] = gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size))

    def observation(self, observations):
        targets_xy = self.env.get_targets_xy()
        # Placing targets on global grid
        for tx, ty in targets_xy:
            self._anonymous_targets[tx, ty] = 1.0

        agents_xy = self.env.get_agents_xy()
        for agent_idx, obs in enumerate(observations):
            x, y = agents_xy[agent_idx]
            r = self.grid_config.obs_radius
            # Removing own target
            self._anonymous_targets[targets_xy[agent_idx]] = 0.0
            obs['anonymous_targets'] = self._anonymous_targets[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)
            self._anonymous_targets[targets_xy[agent_idx]] = 1.0
        # Removing targets on global grid
        for tx, ty in targets_xy:
            self._anonymous_targets[tx, ty] = 0.0

        return observations

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward, action=action), done, tr, info
        
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos

    def reset_state(self):
        self._anonymous_targets = np.zeros_like(self.env.get_obstacles(), dtype=np.float32)


class CutObservationWrapper(ObservationWrapper):
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        return '2_' + x


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
