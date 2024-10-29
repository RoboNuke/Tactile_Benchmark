import unittest
from tasks.simple_fPiH import SimpleFragilePiH
#import h5py
import numpy as np
import mani_skill.envs
#from mani_skill.trajectory import utils as trajectory_utils
#from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import torch
import sapien.physx as physx
import sapien.core as sapien
import time
from wrappers.closed_gripper import *


class TestCloseWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        traj_path = "demos/ForeignObjectRemoval-v1/teleop/trajectory"

        #cls.h5_file = h5py.File(traj_path+".h5", "r")
        #cls.json_data = io_utils.load_json(traj_path + ".json")
        cls.max_reward = 4
        cls.env_id = "SimpleFragilePiH-v1"
        cls.has_set = False

    def tearDown(self):
        if self.has_set:
            self.has_set = False
            self.envs.close()
            del self.envs

    def get_env(self, 
                num_envs=2, 
                reward_mode='sparse', 
                obs_mode='state',
                backend='gpu',
                reset_kwargs = None,
                dmg_force=7.0,
                control_mode="pd_joint_delta_pos"):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(self.env_id, 
                            num_envs=num_envs, 
                            control_mode = control_mode,
                            sim_backend=backend,
                            robot_uids="panda",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode,
                            dmg_force=dmg_force
        )

        if reset_kwargs is None:
            self.envs.reset()
        else:
            if "seed" in reset_kwargs:
                assert reset_kwargs["seed"] == self.ep["episode_seed"]
            else:
                reset_kwargs["seed"] = self.ep["episode_seed"]
            seed = reset_kwargs.pop("seed")
            self.envs.reset(seed=seed, **reset_kwargs)
        
        self.envs = CloseGripperActionSpaceWrapper(self.envs)
    
    def test_gripper_closed(self):
        self.get_env()
        self.envs.reset()
        
        act = self.envs.action_space.sample()*0.0
        for i in range(100):
            act[:,-1] = torch.rand(1)*2-1
            self.envs.step(act)
            assert torch.all(
                self.envs.agent.is_grasping(self.envs.peg)
            ), f"Agent dropped the peg! {i}"