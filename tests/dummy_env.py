import sapien.core as sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.envs.utils import randomization
import numpy as np
import torch
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)
from typing import Any, Dict, Union, List, Optional
import gymnasium as gym

class IncForceWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:

        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)

    def observation(self, observation: Dict):
        assert type(observation) == dict, "Observation must be dict"
        if 'force' in observation:
            observation['force'] = torch.ones((self.base_env.unwrapped.num_envs, 3), device=self.base_env.unwrapped.device) * self.base_env.unwrapped.elapsed_steps[0]
        elif 'force' in observation['extra']:
            observation['extra']['force'] = torch.ones((self.base_env.unwrapped.num_envs, 3), device=self.base_env.unwrapped.device) * self.base_env.unwrapped.elapsed_steps[0]
        else:
            raise NotImplementedError("Force  must be in dict or a sub-dict called extra")
        return observation
    
@register_env("DummyEnv-v1", max_episode_steps=50)
class FOREnv(BaseEnv):
    diff_obs = ['inc_force']
    holder_obs = None
    def __init__(self, *args, 
                 obs_mode= 'state_dict',
                 **kwargs):
        if obs_mode in self.diff_obs:
            self.holder_obs = obs_mode
            obs_mode = 'state_dict'
        super().__init__(*args, obs_mode=obs_mode, **kwargs)

    def _load_scene(self, options: dict):
        pass

    def _initialize_episode(self, 
                            env_idx: torch.Tensor, 
                            options: dict):
        pass

    def evaluate(self):
        out_dic = {}
        out_dic['inc'] = self.elapsed_steps
        return out_dic

    def _get_obs_extra(self, info: Dict):
        data = super()._get_obs_extra(info)
        if self.holder_obs == 'inc_force':
            data['force'] = torch.ones(
                (self.num_envs, 3), 
                device=self.device) * info['inc'][0]
        
        return data

    def compute_dense_reward(self, 
            obs: Any, 
            action: torch.Tensor, 
            info: Dict):
        return 0.0
