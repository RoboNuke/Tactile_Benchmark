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
    @classmethod
    def setUpClass(cls):
        cls.fail_reward = -1
        traj_path = "demos/PegInsertionSide-v1/teleop/trajectory"
        cls.h5_file = h5py.File(traj_path+".h5", "r")

        cls.json_data = io_utils.load_json(traj_path + ".json")

        cls.env_info = cls.json_data["env_info"]
        cls.env_id = 'ScrapBarnicle-v1'
        cls.env_kwargs = cls.env_info["env_kwargs"]
        cls.num_envs = 2
        cls.min_con_force: float = 1.0
        cls.max_reward = 4.0
        cls.has_set = False
    #def setUp(self):
    #    print("Problem setup!")

    def set_envs(self, reward_mode='sparse', obs_mode='state'):
        self.has_set = True
        self.env_kwargs['reward_mode'] = reward_mode
        self.env_kwargs['obs_mode'] = obs_mode
        self.envs = gym.make(self.env_id, 
                            num_envs=self.num_envs, 
                            #sim_backend="gpu",
                            #parallel_in_single_scene=True,
                            **self.env_kwargs)
    
    def in_contact(self, a, b):
        l_f = torch.linalg.norm(self.scene.get_pairwise_contact_forces(a, b), axis=1)
        return torch.all(l_f > self.min_con_force)

    def in_xy_range(self, a, b, thresh=0.001):
        dx = a.p[:,0] - b.p[:,0]
        dy = a.p[:,1] - b.p[:,1]

        return torch.all(dx < thresh) and torch.all(dy < thresh)

    def test_scene_load(self):
        self.set_envs()
        self.envs.reset()
        """
        self.envs.base_env.render_human()
        self.envs.reset()
        for i in range(100):
            self.envs.step(self.envs.action_space.sample())
            self.envs.base_env.render_human()
            time.sleep(0.1)
        """
        # check for barnicle location + friction
        for i in range(len(self.envs.barnicles)-2):
            l_barn = self.env.barnicles[i]
            c_barn = self.env.barnicles[i+1]
            r_barn = self.env.barnicles[i+2]

            # make sure we are holding still
            assert(c_barn.is_static())

            # make sure center is in contact with left and right
            assert( self.in_contact(l_barn.pose, c_barn.pose) )
            assert( self.in_contact(r_barn.pose, c_barn.pose) )

            # make sure all x,y locs are the same
            assert( self.in_xy_range(l_barn.pose, c_barn.pose) )
            assert( self.in_xy_range(r_barn.pose, c_barn.pose) )

            # check friction?
        #input()


    def test_scene_random_init(self):
        self.set_envs()
        centers = torch.zeros((self.num_envs, 3))
        for i in range(3):
            centers[:,i] = self.envs.BARNICLE_SPAWN_CENTER[i] 

        for i in range(100): # sample 100 init configs
            self.envs.reset()
            for barn in self.envs.barnicles:
                assert( self.in_xy_range(barn.pose, centers, self.envs.BARNICLE_SPAWN_RADIUS))      


    def test_obs_state_conditions(self):
        # check state only
        self.set_envs(obs_mode='state')
        obs_wf, _ = self.envs.reset()
        self.envs.close()

        # check without force
        self.set_envs(obs_mode='state_no_ft')
        obs_wof, _ = self.envs.reset()

        # check size is different
        assert(obs_wf.size()[1] - obs_wof.size()[1] == 3)

    def test_obs_sensor_condition(self):
        # check state only
        self.set_envs(obs_mode='rgb')
        obs_wf, _ = self.envs.reset()
        self.envs.close()
        #assert('force' in obs_wf['extra'])

        # check without force
        self.set_envs(obs_mode='rgb_no_ft')
        obs_wof, _ = self.envs.reset()
        assert(not 'force' in obs_wof['extra'])

    def test_evaluate_success(self):
        # ensure eval catches success flag
        self.set_envs()
        self.envs.reset()
        # move all barnicles
        for barn in self.envs.barnicles:
            cur_pos = barn.pose
            cur_pos[:,0] += 10.0 # move it farm away
            barn.pose = cur_pos

        # eval
        info = self.envs.evaluate()

        assert(torch.all(info['success']))
        assert(torch.all(1-info['fail']))
        assert(torch.all(info['is_robot_static']))
        assert(torch.all(info['max_table_force'] < self.envs.FORCE_DMG_TABLE))

    def test_evaluate_fail_table_force(self):
        # check failure by max table force exceeded
        self.set_envs()
        self.envs.reset()
        
        for i in range(149):
            self.envs.step(self.envs.action_space.sample()*0)

        self.envs.max_table_force += self.envs.FORCE_DMG_TABLE * 2

        _, _, _, _, info = self.envs.step(self.envs.action_space.sample()*0)

        assert(torch.all(info['fail']))

    def test_evaluate_fail_barnicle(self):
        # check failure by barnicle still attached
        self.set_envs()
        self.envs.reset()

        for i in range(150):
            self.envs.step(self.envs.action_space.sample()*0)
        # eval
        info = self.envs.evaluate()

        assert(torch.all(1-info['success']))
        assert(torch.all(info['fail']))
        assert(torch.all(info['attached_barnicles'] > 0 ))

    def test_dense_reward_barnicles(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='dense')
        self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

        barn_rewards = [0.06, 0.24, 0.5, 1.0]

        for i in range(len(self.envs.barnicles)-1,-1,-1):
            barn = self.envs.barnicles[i]
            cur_pos = barn.pose
            cur_pos[:,0] += 10.0 # move it farm away
            barn.pose = cur_pos
            _, reward, _,_,_ = self.env.step(self.envs.action_space.sample()*0)
            assert(reward - arm_dist_offset == barn_rewards[i])
        

    def test_normalized_dense_reward_barnicles(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='normalized_dense')
        self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)
        arm_dist_offset /= self.max_reward

        barn_rewards = [0.06, 0.24, 0.5, 1.0]

        for i in range(len(self.envs.barnicles)-1,-1,-1):
            barn = self.envs.barnicles[i]
            cur_pos = barn.pose
            cur_pos[:,0] += 10.0 # move it farm away
            barn.pose = cur_pos
            _, reward, _,_,_ = self.env.step(self.envs.action_space.sample()*0)
            assert(reward - arm_dist_offset == barn_rewards[i])
        

    def test_dense_reward_force_on_table(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='normalized_dense')
        obs = self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

        self.table_force(obs, arm_dist_offset, 0.0, 0.0)
        self.table_force(obs, arm_dist_offset, 0.5*self.envs.FORCE_DMG_TABLE, -0.5)
        self.table_force(obs, arm_dist_offset, self.envs.FORCE_DMG_TABLE, -1.0, fail=1)
        self.table_force(obs, arm_dist_offset, 2*self.envs.FORCE_DMG_TABLE, -1.0, fail=1)
        

    def table_force(self, obs, arm_offset, table_force, result, fail=0):
        info = {
            "fail": torch.ones((self.num_envs)) * fail,
            "success": torch.ones((self.num_envs)) * (1-fail),
            'attached_barnicles': 0,
            "is_robot_static": torch.ones((self.num_envs)),
            "max_table_force": torch.ones((self.num_envs)) * table_force,
            "table_force": torch.ones((self.num_envs)) * table_force
        }
        r = self.envs.compute_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset
        nr = self.envs.compute_normalized_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset/self.max_reward
        assert( torch.all(r - result < 0.001))
        assert( torch.all(r - result > -0.001))

        # check normalized
        assert( torch.all(nr - result/self.max_reward < 0.001))
        assert( torch.all(nr - result/self.max_reward > -0.001))

    def tearDown(self) -> None:
        if self.has_set:
            self.has_set = False
            self.envs.close()
        
if __name__ == "__main__":
    unittest.main()
