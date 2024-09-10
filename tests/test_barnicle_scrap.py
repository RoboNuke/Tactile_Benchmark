import unittest
from tasks.barnicle_scrap import *

import gymnasium as gym
import mani_skill.envs
import torch
import time


import h5py

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, io_utils, wrappers


class TestBarnicleScrap(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    print("Class setup!")

    #def setUp(self):
    #    print("Problem setup!")

    def test_scene_load(self):
        # check for barnicle location + friction
        for barn in self.env.barnicles:
            pass
        assert(1 == 0 )

    def test_scene_random_init(self):
        # reset scene and check important states
        assert(1==1)

    def test_obs_conditions(self):
        # check state only
        # check camera (world+wrist)
        # check without force-torque
        # check human viewable (rgb-array)
        assert(1==1)

    def test_robot_control(self):
        # both robots
        # position
        # velocity
        # ee_delta
        assert(1==1)

    def test_init_checks(self):
        #ensure __init__ catches bad args
        assert(1==1)

    def test_evaluate(self):
        # ensure eval catches failures
        # ensure catches success conditions
        assert(1==1)

    def test_dense_reward(self):
        # ensure reward comes out correctly
        assert(1==1)

    def test_normalized_dense_reward(self):
        assert(1==1)

if __name__ == "__main__":
    unittest.main()
