import unittest
from tasks.simple_fPiH import SimpleFragilePiH
import h5py
import numpy as np
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import torch
import sapien.physx as physx
import sapien.core as sapien
import time

class TestFPiH(unittest.TestCase):
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
    
    """
    # not required as we stole the config from mani skill
    def test_reconfig(self):
        
            Re-loads the scene and 
            checks 
            - hole block location in range
            - hole size is in range
            - peg size
        
        assert(1==1)
        
        self.get_env()
        uw = self.envs.unwrapped
        for i in range(1):
            self.envs.reset(options={"reconfigure":True})
            while True:
                self.envs.step(self.envs.unwrapped.action_space.sample()*0)
                self.envs.render_human()
                print(self.envs.unwrapped.agent.is_grasping(self.envs.unwrapped.peg))
                #print(uw.agent.tcp.pose)
                time.sleep(0.01)
            #for k in range(2):
            #    self.envs.step(None)
            #    #self.envs.render_human()
            #    time.sleep(0.01)
    """

    def test_scene_init(self):
        """
            Reloads scene to check
            - hole block in range
            - peg in robot gipper and doesn't move
        """
        self.get_env()
        self.envs.reset()
        self.envs.step(self.envs.unwrapped.action_space.sample()*0)
        uw = self.envs.unwrapped
        for i in range(100):
            # check block in range
            p = uw.box.pose.p
            rs = torch.sqrt(p[:,0] * p[:,0] + p[:,1] * p[:,1])
            assert torch.all(rs<=uw.HOLE_RADIUS), f"Box spawned outside radius {p}\n{rs}\n{uw.HOLE_RADIUS}" 
            assert torch.all( torch.logical_and(p[:,2] <=0.125, p[:,2] >= 0.085 )), f'Hole height is off {p[:,2]}'
            # check peg in gripper
            grasped = uw.agent.is_grasping(uw.peg)
            assert torch.all(grasped), f'Agent not grasping peg {grasped}'

    def test_partial_scene_init(self):
        """
            Same as test_scene_init except 
            reseting a subset of scenes
        """
        self.get_env(num_envs=2)
        uw = self.envs.unwrapped
        env_idxs = [[0],[1]]
        for env_idx in env_idxs:
            for i in range(10):
                self.envs.reset()
                self.envs.step(self.envs.action_space.sample()*0.0)
                while torch.any(uw.agent.is_grasping(uw.peg)):
                    for k in range(100):
                        self.envs.step(self.envs.action_space.sample())
                # reset now
                self.envs.reset(options={'env_idx':env_idx})
                self.envs.step(uw.action_space.sample()*0)
                grasp = uw.agent.is_grasping(uw.peg)
                assert torch.all(grasp[env_idx]), f"Env Idxs did not reset {grasp}"
                mask = torch.ones(2, dtype=torch.bool)
                mask[env_idx] = False
                assert torch.all(~grasp[mask]), f"Env reset all elements {grasp}"
        
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
        assert 'force' in obs['extra'], 'force not in state_dict'

        self.get_env(obs_mode='state_dict_no_ft')
        obs, _ = self.envs.reset()
        obs,_,_,_,_ = self.envs.step(self.envs.action_space.sample()*0)
        assert not 'force' in obs['extra'], 'force in state_dict_no_ft'

    def test_obs_sensor_condition(self):
        # check state only
        self.get_env(obs_mode='rgb')
        obs_wf, _ = self.envs.reset()
        self.envs.close()
        assert 'force' in obs_wf['extra'], 'Force not  included in rgb'

        # check without force
        self.get_env(obs_mode='rgb_no_ft')
        obs_wof, _ = self.envs.reset()
        assert not 'force' in obs_wof['extra'], 'Force included in rgb_no_ft'

    def test_evaluate_success(self):
        # ensure eval catches success flag
        self.get_env()
        self.envs.step(self.envs.action_space.sample()*0.0)
        uw = self.envs.unwrapped

        # ensure no succ
        eout = uw.evaluate()
        #assert torch.all(~eout['success']), f'Starting in successful state {eout['success']}'

        # move all the pegs
        uw.peg.set_pose(uw.goal_pose)
        eout2 = uw.evaluate()
        assert torch.all(eout2['success']), f'Not all pegs succeeded {eout2}'

    # was tested in fragile peg unit tests  
    def test_evaluate_fail_peg_force(self):
        # check failure by max peg force exceeded
        pass

