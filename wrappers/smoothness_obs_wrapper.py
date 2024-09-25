import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class SmoothnessObservationWrapper(gym.Wrapper):
    """
        Adds to the observation space data required to describe
        the smoothness of a run 


    """
    def __init__(self, env)->None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        
        self.ssv = torch.zeros((self.base_env.num_envs), device=self.device)
        self.ssa = torch.zeros((self.base_env.num_envs), device=self.device)
        self.ssj = torch.zeros((self.base_env.num_envs), device=self.device)
        self.sdf = torch.zeros((self.base_env.num_envs), device=self.device)
        self.old_acc = torch.zeros_like(self.base_env.agent.robot.qacc)
    

    def step(self, action):
        reset_set = (self.unwrapped.elapsed_steps == 0)
        self.ssv[reset_set] *= 0
        self.ssa[reset_set] *= 0
        self.ssj[reset_set] *= 0
        self.old_acc[reset_set] *= 0
        self.sdf[reset_set] *= 0

        observation, r, term, trun, info = self.base_env.step(action)

        obs = {}
        # sum squared velocity
        qvel = self.base_env.agent.robot.get_qvel()
        obs['sqr_qv'] = torch.linalg.norm(qvel * qvel, axis=1)
        self.ssv += obs['sqr_qv']
        obs['sum_sqr_qv'] = self.ssv

        # sum squared accel
        qacc = self.base_env.agent.robot.qacc
        obs['sqr_qa'] = torch.linalg.norm(qacc * qacc, axis=1)
        self.ssa += obs['sqr_qa']
        obs['sum_sqr_qa'] = self.ssa

        # jerk
        jerk = (qacc - self.old_acc) / 0.1
        obs['sqr_qjerk'] =  torch.linalg.norm(jerk * jerk, axis=1)
        self.old_acc = qacc
        self.ssj += obs['sqr_qjerk']
        obs['sum_sqr_qjerk'] = self.ssj

        # force
        self.sdf += info['dmg_force']
        obs['sum_dmg_force'] = self.sdf
        obs['max_dmg_force'] = info['max_dmg_force']

        info['smoothness'] = obs
        
        return observation, r, term, trun, info