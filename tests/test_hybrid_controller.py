import unittest
from tasks.for_env import *
from tasks.fragile_insert import *
from learning.agent import *
from wrappers.n_force_wrapper import NForceWrapper

from tests.dummy_env import *
from controllers.torque import *
from controllers.hybrid import *

from copy import copy

class TestHybridController(unittest.TestCase):
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
                obj_norm_force_range=[0.5, 25.0],
                control_mode='hybrid_ee_force_pos'):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(self.env_id, 
                            num_envs=num_envs, 
                            control_mode = control_mode,
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
        self.con = self.envs.unwrapped.agent.controller.controllers['arm']

    def test_jacobian(self):
        # ensure jacobian is correctly generated
        self.get_env()
        kin = Kinematics(
            "panda_v2.urdf",
            "panda_hand_tcp",
            self.con.articulation,
            self.con.active_joint_indices,
        )

        qs = self.con.qpos
        
        # get ground truth
        jac_true = kin.pk_chain.jacobian(qs)
        # test
        jac_test = self.con.jacobian
        assert torch.all( torch.isclose(jac_test, jac_true) ), f'Jacobian not equal \nTest:\n{jac_test}\nTrue:\n{jac_true}'


    def test_drive_properties(self):
        # ensures config data properly propigated
        self.get_env()

        config = HybridForcePosControllerConfig()
        config.stiffness = torch.zeros_like(self.con.qpos)
        config.damping = torch.zeros_like(self.con.qpos)
        for i in range(config.stiffness.shape()[1]):
            config.stiffness[:,i] = 2.0 * i + 10 
            config.damping[:,i] = 2.0 * i + 11

        self.con.config = config
        st_before = self.con.pos_stiffness
        dp_before = self.con.pos_damping
        self.con.set_drive_property()

        st = self.con.pos_stiffness
        dp = self.con.pos_damping
        assert not torch.all( torch.isclose(st_before, st)), f'Stiffness is the same {st_before} {st}'
        assert not torch.all( torch.isclose(dp_before, dp)), f'Damping is the same {dp_before} {dp}'

    
    def test_action_space_ee_force_init(self):
        # action space when ee force only controller
        self.get_env(control_mode="ee_force")
        controller = self.envs.unwrapped.agent.controller
        assert type(controller.action_space) == gym.spaces.box.Box, f'Space is of type {type(controller.action_space)}, should by gymnasium.spaces.Space'
        as_shape = controller.action_space.shape
        assert as_shape[0] == 2, f'Action space dim should is {as_shape}, should be [2,6]'
        assert as_shape[1] == 6, f'Action space dim should is {as_shape}, should be [2,6]'
        dt = controller.action_space.dtype
        assert dt==np.float32, f'Dtype is {dt} but should be np.float32'

    def test_action_space_init(self):
        # checks action space is correct size
        self.get_env()
        controller = self.envs.unwrapped.agent.controller
        assert type(controller.action_space) == gym.spaces.box.Box, f'Space is of type {type(controller.action_space)}, should by gymnasium.spaces.Space'
        as_shape = controller.action_space.shape
        assert as_shape[0] == 2, f'Action space dim should is {as_shape}, should be [2,6]'
        assert as_shape[1] == 6, f'Action space dim should is {as_shape}, should be [2,6]'
        dt = controller.action_space.dtype
        assert dt==np.float32, f'Dtype is {dt} but should be np.float32'

    def test_set_action_ee_force(self):
        # set ee force control creates correct target
        assert 1==0

    def test_set_action(self):
        # set hybrid control and ensure correct target
        assert 1==0
