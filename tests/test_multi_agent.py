from learning.agent import *
import unittest
import torch

from tasks.fragile_insert import *
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper

mp.set_start_method('spawn')
class TestMultiAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def tearDown(self):
        pass

    def setUp(self):
        self.envs = gym.make(
            'FragilePegInsert-v1', 
            num_envs=20, 
            obs_mode='state_dict',
            robot_uids="panda",
            dmg_force=1000.0
        )
        self.envs = FlattenRGBDFTObservationWrapper(
            self.envs, 
            rgb= False, 
            depth=False, 
            state=True,
            force=False
        )
        self.envs = ManiSkillVectorEnv(
            self.envs, 
            20,
            ignore_terminations=False,
            record_metric=False
        )
        self.next_obs, _ = self.envs.reset()
        self.ma = MultiAgent(10, self.envs, self.next_obs, 'FNN')
        

    def test_multi(self):
        x = torch.tensor([i for i in range(10)], dtype=torch.float32)
        start = time.time()
        self.ma.mp_funct('print_test', x, True)
        end = time.time()

        dt = end - start
        # add .1 to comp for getting everything started
        assert dt < 10.1, f'Parallelize took too long should be less than 11 secs but {dt}'
    
    def test_get_features(self):
        ground_truth = torch.zeros((20, 256), dtype=torch.float32, device=0)
        seq_time = time.time()
        for i in range(10):
            obs = {
                'state': self.next_obs['state'][2*i:2*i+2,:]
            }
            ground_truth[2*i:2*i+2, :] = self.ma.agents[i].get_features(obs)
        torch.cuda.synchronize(0)
        seq_time = time.time() - seq_time

        par_time = time.time()
        test = self.ma.get_features(self.next_obs)
        torch.cuda.synchronize(0)
        par_time = time.time() - par_time
        print(seq_time, par_time)
        assert torch.all( torch.isclose(ground_truth, test) ), f'Parallelize did not get the same result as sequential'
        assert par_time < seq_time, f"Parallelization not faster {par_time} vs sequential time {seq_time}"

    def test_all(self):
        self.ma.get_features(self.next_obs)
        a = self.ma.get_action(self.next_obs)
        self.ma.get_value(self.next_obs)
        self.ma.get_action_and_value(self.next_obs)
        self.ma.get_action_and_value(self.next_obs, a)
        

