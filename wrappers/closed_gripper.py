import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class CloseGripperActionSpaceWrapper(gym.ActionWrapper):
    """
    Ensures the gripper is always closed
    """

    def __init__(self, env) -> None:
        super().__init__(env)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def action(self, action):
        action[:,-1] *= 0.0
        return action