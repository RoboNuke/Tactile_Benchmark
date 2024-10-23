
from typing import Any, Dict, Union
import unittest
from tasks.fragile_insert import *

import gymnasium as gym
import mani_skill.envs
import torch
import time


import h5py

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, io_utils, wrappers

class TestFragileInsert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fail_reward = -1
        traj_path = "demos/PegInsertionSide-v1/teleop/trajectory"
        cls.h5_file = h5py.File(traj_path+".h5", "r")

        cls.json_data = io_utils.load_json(traj_path + ".json")

        cls.env_info = cls.json_data["env_info"]
        cls.env_id = 'FragilePegInsert-v1'
        #cls.env_id = "PegInsertionSide-v1"
        cls.env_kwargs = cls.env_info["env_kwargs"]
        cls.num_envs = 2
        #cls.env = gym.make(cls.env_id, num_envs=cls.num_envs, 
        #                    sim_backend="gpu",
        #                    #parallel_in_single_scene=True,
        #                    **cls.env_kwargs)


        
    def playDemo(self, idx, reward_mode='sparse'):

        ep = self.json_data['episodes'][idx]
        
        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]
        seed = reset_kwargs.pop("seed")
        self.env_kwargs['reward_mode'] = reward_mode
        self.env = gym.make(self.env_id, 
                            num_envs=self.num_envs, 
                            sim_backend="gpu",
                            #parallel_in_single_scene=True,
                            **self.env_kwargs)
        self.env.reset(seed=seed, **reset_kwargs)
        #self.env.reward_mode = reward_mode
        #print(self.env.reward_mode)
        # set first environment state and update recorded env state
        ori_env_states = trajectory_utils.dict_to_list_of_dicts(
            self.h5_file[f"traj_{idx}"]["env_states"]
        )
        self.env.base_env.set_state_dict(ori_env_states[0])
        ori_env_states = ori_env_states[1:]

        # Original actions to replay
        ori_actions = self.h5_file[f"traj_{idx}"]["actions"][:]
        info = {}

        n = len(ori_actions)
    
        info = None
        truncated = None
        reward = None
        for t, a in enumerate(ori_actions):
            
            sa = torch.tensor([a for k in range(self.num_envs)])
            _, reward, terminated, truncated, info = self.env.step(sa)
            print(
                self.env.scene.get_pairwise_contact_forces(self.env.peg, self.env.box)
            )
            #self.env.base_env.set_state_dict(ori_env_states[t])
            self.env.base_env.render_human()
            if terminated[0]: # running the same action so shoudl be same
                break

        self.env.close()
        return info, truncated, terminated, reward

    """def setUp(self):
        # create maniskill env
        self.env = gym.make(
            "FragilePegInsert-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            sim_backend="cpu",
            obs_mode="state", # there is also "state_dict", "rgbd", ...
            control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
            #parallel_gui_render_enabled=True,
            render_mode="sensors",
            robot_uids="panda_wristcam"
        )
        self.obs, _ = self.env.reset()
    """
    def test_peg_arm_collision(self):
        info, truncated, terminated, reward = self.playDemo(1)
        # should have failed
        assert(terminated[0])
        assert('fail' in info)
        assert(info['fail'][0] == True)
        assert(info['fail_cause'][0] == 1)
        assert(reward[0] == self.fail_reward)
    
    
    def test_peg_hole_collision(self):
        info, truncated, terminated, reward = self.playDemo(0)
        # should have failed
        assert(terminated[0])
        assert('fail' in info)
        assert(info['fail'][0] == True)
        assert(info['fail_cause'][0] == 2)
        assert(reward[0] == self.fail_reward)

    def test_dense_reward(self):
        info, truncated, terminated, reward = self.playDemo(1, reward_mode='dense')
        assert(reward[0] == -10)

if __name__ == "__main__":
    unittest.main()
    """
        next_obs, reward, terminations, truncations, infos = self.env.step(self.env.action_space.sample())
        print(terminations, truncations)
        i = 1
        while not torch.any(terminations) and not torch.any(torch.tensor(truncations)):
            i += 1
            next_obs, reward, terminations, truncations, infos = self.env.step(self.env.action_space.sample())
            print(i, terminations, truncations)
        print("all done!")

    """