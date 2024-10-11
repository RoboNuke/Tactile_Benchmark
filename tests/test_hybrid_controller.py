import unittest
from tasks.for_env import *
from tasks.fragile_insert import *
from learning.agent import *
from wrappers.n_force_wrapper import NForceWrapper

from tests.dummy_env import *
from controllers.torque import *
from controllers.hybrid import *

from copy import copy

class TestTorqueController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_id = "ForeignObjectRemoval-v1"
        cls.has_set = False

    def tearDown(self):
        if self.has_set:
            self.has_set = False
            self.envs.close()
            del self.envs

    def get_env(self, 
                num_envs=2, 
                reward_mode='normalized_dense', 
                obs_mode='state_dict',
                backend='gpu',
                reset_kwargs = None,
                dmg_table_force=7.0,
                obj_norm_force_range=[0.5, 25.0]):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(self.env_id, 
                            num_envs=num_envs, 
                            control_mode = "joint_torque",
                            sim_backend=backend,
                            robot_uids="panda_force",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode,
                            dmg_table_force=dmg_table_force,
                            obj_norm_force = obj_norm_force_range
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
