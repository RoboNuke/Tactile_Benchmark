import unittest
from tasks.for_env import *
from tasks.fragile_insert import *
from learning.agent import *
from wrappers.n_force_wrapper import NForceWrapper

from tests.dummy_env import *
from controllers.torque import *

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

    def test_action_space(self):
        # ensures actions space is correct
        self.get_env()
        controller = self.envs.unwrapped.agent.controller
        assert type(controller.action_space) == gym.spaces.box.Box, f'Space is of type {type(controller.action_space)}, should by gymnasium.spaces.Space'
        as_shape = controller.action_space.shape
        assert as_shape[0] == 2, f'Action space dim should is {as_shape}, should be [2,8]'
        assert as_shape[1] == 8, f'Action space dim should is {as_shape}, should be [2,8]'
        dt = controller.action_space.dtype
        assert dt==np.float32, f'Dtype is {dt} but should be np.float32'

    def test_single_action_space(self):
        # ensures single action space correct
        self.get_env()
        controller = self.envs.unwrapped.agent.controller
        sas = controller.single_action_space
        assert type(sas) == gym.spaces.box.Box, f"Single Action Space type is {type(sas)}, but should be gym.spaces.Space"
        sh = sas.shape
        assert sh == (8,), f'Single action space shape is {sh} but should be [1,8]'
        dt = sas.dtype
        assert dt==np.float32, f'Dtype is {dt} but should be np.float32'

    """def test_set_drive_property(self):
        # ensures properties update for each joint
        self.get_env()
        controller = self.envs.unwrapped.agent.controller
        for i, joint in enumerate(controller.joints):
            # for each joint check
            assert joint.drive_mode == 'force', f'Drive mode of joint {i} is {joint.drive_mode}'
            assert joint.sti
        assert(1==1)
    """

    def test_reset(self):
        self.get_env()
        con = self.envs.unwrapped.agent.controller.controllers['arm']
        start_qf = con.qf.clone()
        for i in range(10):
            self.envs.step(np.ones_like(self.envs.action_space.sample()))
        pre_reset_qf = con.qf.clone()
        #step = con._step
        self.envs.reset()
        end_qf = con.qf.clone()
        #assert not step == con._step, f'Step did not reset, {step} to {con._step}'
        assert torch.all(~(pre_reset_qf==start_qf)), f"Robot did not take actions {pre_reset_qf}"
        assert torch.all(~(pre_reset_qf==end_qf)), f"Robot did not reset qf {end_qf} {pre_reset_qf}"
        assert torch.all(start_qf == end_qf), f'{end_qf} Did not reset to original state {start_qf}'
        
    def get_qfs(self, con):
        return con.articulation.get_qf()
    

    def test_partial_reset(self):
        self.get_env()
        con = self.envs.unwrapped.agent.controller.controllers['arm']
        start_qf = con.qf
        for i in range(10):
            self.envs.step(np.ones_like(self.envs.action_space.sample()))
        #step = copy(con._step)
        self.envs.reset(options={'env_idx':[0]})
        #assert not step[0] == con._step[0], f'Env 1 step did not reset, {step[0]} to {con._step[0]}'
        #assert step[1] == con._step[1], f'Env 2 reset, {step[1]} to {con._step[1]}'
        qf = con.qf.clone()
        assert torch.all(start_qf[0,:] == qf[0,:]), f'Env 1 Did not reset to original state'
        assert torch.all(~(start_qf[1,:] == qf[1,:])), f'Env 2 did reset to original state'
        

    def test_set_action(self):
        self.get_env()

        init_action = (torch.ones((2,8)) * 4.0  )/100.0
        final_action = init_action[:,:-1]*100.0#[] # action after preprocessing
        
        self.envs.step(init_action)
        con = self.envs.unwrapped.agent.controller.controllers['arm']
        #for i, joint in enumerate(controller.joints):
        # for each joint check
        qf = con.qf.cpu()
        assert torch.all(qf == final_action), f"qf not set properly {qf} should be {final_action}"