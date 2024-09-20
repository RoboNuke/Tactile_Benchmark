import unittest
from tasks.fragile_insert import *
from tasks.wiping import *
from tasks.barnicle_scrap import *
from learning.agent import *

import gymnasium as gym
import mani_skill.envs
import torch
import time


import h5py

import sapien.physx as physx
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, io_utils, wrappers

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper

class TestAgent(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    pass

    #def setUp(self):
    #    print("Problem setup!")

    #def tearDown(self):
    #    pass

    def get_env(self, 
                env_id="WipeFood-v1",
                num_envs=2, 
                reward_mode='sparse', 
                obs_mode='state',
                backend='gpu'):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(env_id, 
                            num_envs=num_envs, 
                            sim_backend=backend,
                            robot_uids="panda",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode)
        self.envs.reset()

    def test_rgb_extractor(self):
        self.get_env(obs_mode='rgb')
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, 
                                                  rgb=True, 
                                                  depth=False, 
                                                  state=False,
                                                  force=False)
        obs, _ = self.envs.reset()
        encoder = NatureCNN(obs).to("cuda:0")

        assert('rgb' in encoder.extractors)
        
        encoded = encoder.forward(obs) # don't need to do much because this already implemented
        assert(encoded.dtype == torch.float)


    def test_state_extractor(self):
        self.get_env(obs_mode='state_dict')
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, 
                                                  rgb=False, 
                                                  depth=False, 
                                                  state=True,
                                                  force=False)
        obs, _ = self.envs.reset()

        assert('state' in obs)
        assert(not 'force' in obs)
        assert(not 'rgb' in obs)

        encoder = NatureCNN(obs).to("cuda:0")

        assert('state' in encoder.extractors)
        assert(not 'force' in encoder.extractors)
        assert(not 'rgb' in encoder.extractors)

        encoded = encoder.forward(obs)

        assert(encoded.dtype == torch.float)
        assert(encoded.size()[0] == 2)
        assert(encoded.size()[1] == 256)        
        
    def test_force_extractor(self):
        self.get_env(obs_mode='rgb')
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, 
                                                  rgb=False, 
                                                  depth=False, 
                                                  state=False,
                                                  force=True)
        obs, _ = self.envs.reset()

        assert(not 'state' in obs)
        assert('force' in obs)
        assert(not 'rgb' in obs)

        encoder = NatureCNN(obs, force_type="FNN").to("cuda:0")

        assert(not 'state' in encoder.extractors)
        assert('force' in encoder.extractors)
        assert(not 'rgb' in encoder.extractors)

        encoded = encoder.forward(obs)

        assert(encoded.dtype == torch.float)
        assert(encoded.size()[0] == 2)
        assert(encoded.size()[1] == 256)   
    
    
    def test_1D_CNN_extractor(self):
        """ TODO: Need to figure out the whole 1DCNN stuff """
        assert(1==1)
        
    def test_all_extractor(self):
        self.get_env(obs_mode='rgb')
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, 
                                                  rgb=True, 
                                                  depth=False, 
                                                  state=True,
                                                  force=True)
        obs, _ = self.envs.reset()

        assert('state' in obs)
        assert('force' in obs)
        assert('rgb' in obs)

        assert(obs['state'].size()[1] == 25)
        assert(obs['force'].size()[1] == 3)

        encoder = NatureCNN(obs, force_type="FNN").to("cuda:0")

        assert('state' in encoder.extractors)
        assert('force' in encoder.extractors)
        assert('rgb' in encoder.extractors)

        encoded = encoder.forward(obs)

        assert(encoded.dtype == torch.float)
        assert(encoded.size()[0] == 2)
        assert(encoded.size()[1] == encoder.out_features)   
    
    def ensure(self, env_id):
        self.get_env(
                env_id=env_id,
                obs_mode='rgb'
        )
        obs, _ = self.envs.reset()
        
        assert('agent' in obs)
        assert('force' in obs['extra'])
        assert('sensor_data' in obs)

        
        self.get_env(
                env_id=env_id,
                obs_mode='rgb_no_ft'
        )
        obs, _ = self.envs.reset()

        assert('agent' in obs)
        assert(not 'force' in obs['extra'])
        assert('sensor_data' in obs)

        
        self.get_env(
                env_id=env_id,
                obs_mode='state_dict')
        obs, _ = self.envs.reset()

        assert('agent' in obs)
        assert('force' in obs['extra'])
        assert(not 'sensor_data' in obs)

        
        self.get_env(
                env_id=env_id,
                obs_mode='state_dict_no_ft')
        obs, _ = self.envs.reset()

        assert('agent' in obs)
        assert(not 'force' in obs['extra'])
        assert(not 'sensor_data' in obs)

    def test_scraping_obs_space(self):
        self.ensure('ScrapBarnicle-v1')

    def test_wiping_obs_space(self):
        self.ensure('WipeFood-v1')

    def test_ppih_obs_space(self):
        self.ensure('FragilePegInsert-v1')

        
    
