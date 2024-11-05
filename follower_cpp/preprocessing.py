import gymnasium


class ProvideMapWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        global_obstacles = self.get_global_obstacles()
        global_agents_xy = self.get_global_agents_xy()
        for idx, obs in enumerate(observations):
            obs['global_obstacles'] = global_obstacles
            obs['global_agent_xy'] = global_agents_xy[idx]
        return observations, infos


def follower_cpp_preprocessor(env, algo_config):
    env = ProvideMapWrapper(env)
    return env
