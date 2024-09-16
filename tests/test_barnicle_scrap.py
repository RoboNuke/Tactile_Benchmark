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
        cls.max_reward = 3.0
        cls.has_set = False
    #def setUp(self):
    #    print("Problem setup!")
    
    def test_play(self):
        self.set_envs()
        self.envs.reset()
        
        self.envs.base_env.render_human()
        self.envs.reset()
        i = 0
        while True:
            self.envs.step(self.envs.action_space.sample()*0)
            #self.envs.reset()
            self.envs.base_env.render_human()
            i += 1
            if i % 10 == 0:
                self.envs.reset()
    

    def set_envs(self, reward_mode='sparse', obs_mode='state'):
        self.has_set = True
        self.env_kwargs['reward_mode'] = reward_mode
        self.env_kwargs['obs_mode'] = obs_mode
        self.envs = gym.make(self.env_id, 
                            num_envs=self.num_envs, 
                            sim_backend="gpu",
                            #parallel_in_single_scene=True,
                            **self.env_kwargs)
    
    def in_contact(self, a, b):
        l_f = torch.linalg.norm(self.envs.scene.get_pairwise_contact_forces(a, b), axis=1)
        return torch.all(l_f > self.min_con_force)

    def in_xy_range(self, a, b, thresh=0.001):
        dx = a.p[:,0] - b.p[:,0]
        dy = a.p[:,1] - b.p[:,1]

        return torch.all(dx < thresh) and torch.all(dy < thresh)

    def in_xy(self, a, b, thresh=0.001):
        dx = a.p[:,0] - b[:,0]
        dy = a.p[:,1] - b[:,1]

        return torch.all(dx < thresh) and torch.all(dy < thresh)

    def test_scene_load(self):
        self.set_envs()
        self.envs.reset()
        self.envs.step(self.envs.action_space.sample()*0.0)
        
        # check for barnicle location + friction
        for i in range(len(self.envs.barnicles)-2):
            l_barn = self.envs.barnicles[i]
            c_barn = self.envs.barnicles[i+1]
            r_barn = self.envs.barnicles[i+2]

            # make sure we are holding still
            #assert(torch.all(c_barn.is_static()))

            # make sure center is in contact with left and right
            assert( torch.all(self.in_contact(l_barn, c_barn) ))
            assert( torch.all(self.in_contact(r_barn, c_barn) ))

            # make sure all x,y locs are the same
            assert( torch.all(self.in_xy_range(l_barn.pose, c_barn.pose) ))
            assert( torch.all(self.in_xy_range(r_barn.pose, c_barn.pose) ))


    def test_scene_random_init(self):
        self.set_envs()
        centers = torch.zeros((self.num_envs, 3), device="cuda:0")
        for i in range(2):
            centers[:,i] = self.envs.BARNICLE_SPAWN_CENTER[i] 

        for i in range(100): # sample 100 init configs
            self.envs.reset()
            for barn in self.envs.barnicles:
                assert( self.in_xy(barn.pose, centers, self.envs.BARNICLE_SPAWN_DELTA))      


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
        for barn in self.envs.unwrapped.barnicles:
            barn.set_pose(sapien.Pose([10,10,0]))

        # eval
        info = self.envs.unwrapped.evaluate()

        assert(torch.all(info['success']))
        assert(torch.all(~info['fail']))
        assert(torch.all(info['is_robot_static']))
        assert(torch.all(info['max_table_force'] < self.envs.unwrapped.FORCE_DMG_TABLE))

    def test_evaluate_fail_table_force(self):
        # check failure by max table force exceeded
        self.set_envs()
        self.envs.reset()
        
        starting_poses = self.envs.unwrapped.barn_starts.clone()
        for i in range(50):
            if i == 25: # halfway  "fail" in env 1
                self.envs.unwrapped.max_table_force[0] += self.envs.unwrapped.FORCE_DMG_TABLE * 2
            if i == 40: # last step "fail" in env 2
                self.envs.unwrapped.max_table_force[1] += self.envs.unwrapped.FORCE_DMG_TABLE * 2
            
            obs, r, term, trunc, info = self.envs.step(self.envs.action_space.sample()*0)
            #print(term, trunc,  info['fail'])
            if i == 25:
                assert(term[0] and not term[1]) # only env 1 failed
                assert(info['fail'][0] and not info['fail'][1])
                self.envs.reset(options={'env_idx':[0]})
                # check env reset correctly
                assert(self.envs.unwrapped.elapsed_steps[0] == 0)
                assert(self.envs.unwrapped.elapsed_steps[1] == 26)
                # ensure barnicles updated correctly
                assert(torch.linalg.norm(starting_poses[0,:] - self.envs.unwrapped.barn_starts[0,:]) > 0.001)
                assert(torch.linalg.norm(starting_poses[1,:] - self.envs.unwrapped.barn_starts[1,:]) < 0.0001)

            if i == 26:
                assert(self.envs.unwrapped.elapsed_steps[0] == 1) # only first env reset
                assert(self.envs.unwrapped.elapsed_steps[1] == 27)

            if i >= 40: # check fail stays until reset
                assert(info['fail'][1] and not info['fail'][0])
            elif not i == 25:
                assert(not info['fail'][1] and not info['fail'][0])
            

        _, _, _, _, info = self.envs.step(self.envs.action_space.sample()*0)

        assert(info['fail'][1])
        assert(torch.all(~info['success'][1]))
        assert(torch.all(info['max_table_force'][1] == self.envs.unwrapped.FORCE_DMG_TABLE * 2))

    def test_dense_reward_barnicles(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='dense')
        self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

        nb = len(self.envs.unwrapped.barnicles)
        for i in range(nb):
            barn = self.envs.unwrapped.barnicles[i]
            barn.set_pose(sapien.Pose([10,10,0]))
            _, reward, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)
            for k in range(self.num_envs):
                assert(reward[k] - arm_dist_offset[k] - (i+1)/nb < 0.001)
        

    def test_normalized_dense_reward_barnicles(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='normalized_dense')
        self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)
        #arm_dist_offset /= self.max_reward

        nb = len(self.envs.unwrapped.barnicles)
        for i in range(nb):
            barn = self.envs.barnicles[i]
            barn.set_pose(sapien.Pose([10,10,0]))
            _, reward, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)
            for k in range(self.num_envs):
                assert(reward[k] - arm_dist_offset[k] - (i+1)/nb/self.max_reward < 0.001)
        

    def test_dense_reward_force_on_table(self):
        # ensure reward increases as barnicles are removed
        self.set_envs(reward_mode='dense')
        obs = self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

        arm_dist_offset[1] -= 0.25

        #print("No force")
        self.table_force(obs, arm_dist_offset, 0.0, 0.0)
        #print("half")
        self.table_force(obs, arm_dist_offset, 0.5*self.envs.unwrapped.FORCE_DMG_TABLE, -0.01799)
        #print("full")
        self.table_force(obs, arm_dist_offset, self.envs.unwrapped.FORCE_DMG_TABLE, -0.99753, fail=1)
        #print("double")
        self.table_force(obs, arm_dist_offset, 2*self.envs.unwrapped.FORCE_DMG_TABLE, -1.0, fail=1)
        

    def table_force(self, obs, arm_offset, table_force, result, fail=0):
        info = {
            "fail": torch.zeros((self.num_envs), device="cuda:0", dtype=torch.bool),
            "success": torch.zeros((self.num_envs), device="cuda:0", dtype=torch.bool),
            'attached_barnicles': torch.ones((self.num_envs), device="cuda:0")*4,
            "is_robot_static": torch.ones((self.num_envs), device="cuda:0"),
            "max_table_force": torch.ones((self.num_envs), device="cuda:0") * table_force,
            "table_force": torch.ones((self.num_envs), device="cuda:0") * table_force
        }
        #print(self.envs.unwrapped.compute_dense_reward(obs, self.envs.action_space.sample()*0, info), arm_offset)
        r = self.envs.unwrapped.compute_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset
        nr = self.envs.unwrapped.compute_normalized_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset/self.max_reward
        #print(r, result)
        
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
