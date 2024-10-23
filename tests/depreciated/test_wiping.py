import unittest
from tasks.wiping import *
from tasks.barnicle_scrap import *

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


class TestWiping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fail_reward = -1
        cls.env_id = 'WipeFood-v1'
        cls.min_con_force: float = 1.0
        cls.max_reward = 2.0
        cls.has_set = False

    #def setUp(self):
    #    print("Problem setup!")

    def tearDown(self):
        if self.has_set:
            self.has_set = False
            self.envs.close()
            del self.envs

    def get_env(self, 
                num_envs=2, 
                reward_mode='sparse', 
                obs_mode='state',
                backend='gpu'):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(self.env_id, 
                            num_envs=num_envs, 
                            sim_backend=backend,
                            robot_uids="panda",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode)
        self.envs.reset()


    def test_episode_init(self):
        """
            Re-initializes episodes and checks:
            - Food pose in range
            - tabletop at location
        """
        self.get_env()
        uw = self.envs.unwrapped

        for i in range(100):
            self.envs.reset()
            fp = torch.linalg.norm(uw.food.pose.raw_pose[:,:2], axis=1)
            # check food in pose range
            assert(torch.all(fp <= uw.FOOD_SPAWN_RADIUS))

            # ensure tabletop is correct size
            tt = torch.zeros((2,1))
            for i, obj in enumerate(uw.table_top._objs):
                tths = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
                tt[i] = float(tths.get_collision_shapes()[0].get_half_size()[2])
            assert(torch.all(tt - uw.TABLE_TOP_THICKNESS/2.0 < 0.0001))
            assert(torch.all(tt - uw.TABLE_TOP_THICKNESS/2.0 > -0.0001))

            # ensure tabletop is at correct location
            ttps = uw.table_top.pose.raw_pose
            assert(torch.all( ttps[:,:2] < 0.001))
            assert(torch.all( ttps[:,:2] > -0.001))
            assert(torch.all( ttps[:,2]- uw.TABLE_TOP_THICKNESS/2.0 < 0.0001))
            assert(torch.all( ttps[:,2]- uw.TABLE_TOP_THICKNESS/2.0 > -0.0001))

    def test_episode_partial_init(self):
        """ 
            Resets single envs 
            Checks
            - single env changed
            - other env do not change
        """
        self.get_env()
        uw = self.envs.unwrapped

        self.envs.reset()
        # get OG food poses
        fp1 = uw.food.pose.raw_pose[:,:3]
        self.envs.step(self.envs.action_space.sample()*0)
        self.envs.reset(options={'env_idx':[0]})

        fp2 = uw.food.pose.raw_pose[:,:3]
        ds = torch.linalg.norm(fp1 - fp2, axis=1)
        # first one changed
        assert(ds[0] > 0.001 or ds[0] < -0.001)
        assert(uw.elapsed_steps[0] == 0)
        # second did not
        assert(ds[1] < 0.001 and ds[1] > -0.001)
        assert(uw.elapsed_steps[1] == 1)


    def not_moved_by(self, force):
        uw = self.envs.unwrapped

        self.envs.reset()
        #self.envs.render_human()
        self.envs.step(self.envs.action_space.sample() * 0.0)
        p0 = uw.food.pose.raw_pose[0,:] # get init pos
        # apply the force to env 0 only
        """uw.food._bodies[0].add_force_at_point(
            force=force,
            point=[0,0,0]
        )"""
        #self.envs.render_human()

        for i in range(100):
            if i >= 50:
                uw.food._bodies[0].add_force_at_point(
                    force=force,
                    point=[0,0,0]
                )

            #time.sleep(0.1)
            self.envs.step(self.envs.action_space.sample() * 0.0)
            #self.envs.render_human()
        pf = uw.food.pose.raw_pose[0,:]

        d = torch.linalg.norm(p0-pf)
        return(d < 0.001 and d > -0.001)

    def test_food_norm_force(self):
        """
            Applies force in the food's normal direction
            checks:
            - food doesn't move (significantly)
        """
        self.get_env(num_envs=1, backend='cpu')
        assert(self.not_moved_by([0,0,-0.1]))
    
    def test_food_small_force(self):
        """
            Applies a small force // to the table
            checks:
            - food doesn't move 
        """
        self.get_env(num_envs=1, backend='cpu')
        uw = self.envs.unwrapped

        min_force_to_move = (uw.MIN_FOOD_DENSITY * 
                             uw.MIN_FOOD_W * uw.MIN_FOOD_W * 
                             uw.MIN_FOOD_H * 
                             uw.MIN_FOOD_FRIC * 9.81) - 0.01

        assert(self.not_moved_by([min_force_to_move,0,0]))
    
    def test_food_large_force(self):
        """
            Applies a large force // to the table
            checks:
            - food moved
        """
        self.get_env(num_envs=1, backend='cpu')
        uw = self.envs.unwrapped

        max_force_to_move = (uw.MAX_FOOD_DENSITY * 
                             uw.MAX_FOOD_W * uw.MAX_FOOD_W * 
                             uw.MAX_FOOD_H * 
                             uw.MAX_FOOD_FRIC * 9.81) * 1.1

        assert(not self.not_moved_by([max_force_to_move,0,0]))

    def test_extra_obs_force(self):
        """ 
            Compares obs_mode with and without force data
            Checks
            - Force present by default
            - Not present with _no_ft in the obs_mode
        """
        self.get_env(obs_mode='state_dict')
        obs, _ = self.envs.reset()
        obs,_,_,_,info = self.envs.step(self.envs.action_space.sample()*0)
        assert( 'force' in obs['extra'])

        self.get_env(obs_mode='state_dict_no_ft')
        obs, _ = self.envs.reset()
        obs,_,_,_,_ = self.envs.step(self.envs.action_space.sample()*0)
        assert( not 'force' in obs['extra'])

    def test_eval_success(self):
        """
            Tests success is correctly found
            checks
            - success after all food moved
            - failure is false
        """
        self.get_env(num_envs=1, backend='cpu')
        uw = self.envs.unwrapped

        max_force = (uw.MAX_FOOD_DENSITY * 
                    uw.MAX_FOOD_W * uw.MAX_FOOD_W * 
                    uw.MAX_FOOD_H * 
                    uw.MAX_FOOD_FRIC * 9.81) * 1.2 # should always move it

        self.envs.reset()
        self.envs.step(self.envs.action_space.sample() * 0.0)
        p0 = uw.food.pose.raw_pose[0,:3] # get init pos

        for i in range(100):
            uw.food._bodies[0].add_force_at_point(
                force=[max_force,0,0],
                point=[0,0,0]
            )
            obs, r, term, trunc, info = self.envs.step(self.envs.action_space.sample() * 0.0)
            if( torch.any(info['success']) or 
                torch.any(trunc) or
                torch.any(term) or 
                torch.any(info['fail'])):
                break
        pf = uw.food.pose.raw_pose[0,:3]

        d = torch.linalg.norm(p0-pf)
        #print(d, info)
        #print(term, trunc)
        assert(d > 0.001 or d < -0.001)
        assert(info['success'])
        assert(torch.all(~info['fail'])) # neither should have failed
        assert(term)
        assert(~trunc)
    """
    def test_eval_failure(self):
        
        #    Tests large table forces cause failure
        #    checks
        #    - Failure on large table force
        
        self.get_env(num_envs=1, backend='cpu')
        uw = self.envs.unwrapped

        self.envs.reset()
        #uw.food.set_pose(sapien.Pose([0.0, -2.25 * uw.FOOD_SPAWN_RADIUS,0]))
        self.envs.step(self.envs.action_space.sample() * 0.0)
        force_applied = 80

        for i in range(50):
            # apply the force to env 0 only
            uw.render_human()
            time.sleep(0.1)
            uw.food._bodies[0].add_force_at_point(
                force=[0,0,-force_applied],
                point=[0,0,0]
            )
            obs, r, term, trunc, info = self.envs.step(self.envs.action_space.sample() * 0.0)
            if( torch.any(info['success']) or 
                torch.any(trunc) or
                torch.any(term) or 
                torch.any(info['fail'])):
                break
            time.sleep(0.1)

        print(info)
        print(term)
        print(trunc)
        
        assert(info['fail'])
        assert(~info['success']) # neither should have succeeded
        assert(~trunc)
        assert(term)
        assert(info['max_table_force'] - force_applied < 0.001 and
               info['max_table_force'] - force_applied > -0.001)
    """
    def test_dense_table_force(self):
        """
            Ensures reward updates with table force
            checks
            - dense reward
            - normalized dense reward
        """
        self.get_env(reward_mode='dense')
        obs = self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

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
            "is_robot_static": torch.ones((self.num_envs), device="cuda:0"),
            "max_table_force": torch.ones((self.num_envs), device="cuda:0") * table_force,
            "table_force": torch.ones((self.num_envs), device="cuda:0") * table_force
        }
        #print(self.envs.unwrapped.compute_dense_reward(obs, self.envs.action_space.sample()*0, info), arm_offset)
        #print("arm offset:", arm_offset)
        r = self.envs.unwrapped.compute_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset
        nr = self.envs.unwrapped.compute_normalized_dense_reward(obs, self.envs.action_space.sample()*0, info) - arm_offset/self.max_reward
        #print("raw results:", r, result, r - result)
        
        assert( torch.all(r - result < 0.001))
        assert( torch.all(r - result > -0.001))

        # check normalized
        assert( torch.all(nr - result/self.max_reward < 0.001))
        assert( torch.all(nr - result/self.max_reward > -0.001))

    


    