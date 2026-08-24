import json
import numpy as np


def run_episode(env, algo, log_path=None):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()
    obs, reset_infos = env.reset(seed=env.grid_config.seed if getattr(env, 'grid_config', False) else 0)
    episode_log = [{'obs': obs, 'infos': reset_infos, 'step': 0}]
    infos = None
    while True:
        actions, policy_outputs = algo.act(obs, infos=infos)
        obs, rew, dones, tr, infos = env.step(actions)
        #if getattr(env, 'print_elapsed_steps', False):
        #    print(f"evl utls rew {rew}, step # {env.print_elapsed_steps}, dones {sum(dones)}, tr {sum(tr) }")
        #else:
        #    print(f"evl utls rew {rew}, dones {sum(dones)}, tr {sum(tr) }")
        
        step_outputs = policy_outputs.copy()
        step_outputs['step'] = env.print_elapsed_steps if getattr(env, 'print_elapsed_steps', False) else 0
        step_outputs['obs'] = obs
        step_outputs['rew'] = rew
        step_outputs['dones'] = dones
        step_outputs['tr'] = tr
        step_outputs['infos'] = infos
        episode_log.append(step_outputs)
        #print(len(episode_log))
        #print(f"upd obs = {obs}")
        results_holder.after_step(infos)
        if log_path is not None:
            np.save(log_path, episode_log, allow_pickle=True)
            #with open(log_path + '/episode_log.json', 'w') as f:
            #    json.dump(episode_log, f)

        if all(dones) or all(tr):
            break
    
    return results_holder.get_final()


class ResultsHolder:
    """
    Holds and manages the results obtained during an episode.

    """

    def __init__(self):
        """
        Initializes an instance of ResultsHolder.
        """
        self.results = dict()

    def after_step(self, infos):
        """
        Updates the results with the metrics from the given information.

        Args:
            infos (List[dict]): List of dictionaries containing information about the episode.

        """
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        """
        Returns the final results obtained during the episode.

        Returns:
            dict: The final results.

        """
        return self.results

    def __repr__(self):
        """
        Returns a string representation of the ResultsHolder.

        Returns:
            str: The string representation of the ResultsHolder.

        """
        return str(self.get_final())
