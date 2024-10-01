import gymnasium as gym
from typing import Dict
import torch
from mani_skill.envs.sapien_env import BaseEnv


class NForceWrapper(gym.ObservationWrapper):
    """
    Keeps the last n force readings

    Args:
        n (int): number of observations to keep

    """
    def __init__(self, env, n = 10, test_obs=None) -> None:

        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.force_data = torch.zeros((env.unwrapped.num_envs, n, 3))

        if not type(test_obs) == dict:
            raise NotImplementedError("Observation must be a dict")
        if 'force' not in test_obs and 'force' not in test_obs['extra']:
            raise NotImplementedError("Force must be at the top level dict or in the extra sub-dict")

    def observation(self, observation: Dict):
        self.force_data = torch.roll(self.force_data, 1, 1)
        if 'force' in observation:
            self.force_data[:,0,:] = observation['force']
            observation['force'] = self.force_data
        elif 'extra' in observation and 'force' in observation['extra']:
            self.force_data[:,0,:] = observation['extra']['force']
            observation['extra']['force'] = self.force_data
        else:
            raise NotImplementedError("Unable to locate force readings in observation")

        return observation