import unittest
from tasks.for_env import *
from tasks.fragile_insert import *
from learning.agent import *
from wrappers.n_force_wrapper import NForceWrapper

from tests.dummy_env import *

class TestNForce(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    pass

    #def setUp(self):
    #    print("Problem setup!")

    #def tearDown(self):
    #    pass

    def get_env(self, 
                env_id="DummyEnv-v1",
                num_envs=2, 
                reward_mode='sparse', 
                obs_mode='state_dict',
                backend='gpu',
                reset_kwargs = None,
                stack=1,
                dmg_table_force=7.0,
                obj_norm_force_range=[0.5, 25.0]):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(env_id, 
                            num_envs=num_envs, 
                            control_mode = "pd_joint_pos",
                            sim_backend=backend,
                            robot_uids="panda",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode,
                            #stack=stack,
                            #dmg_table_force=dmg_table_force,
                            #obj_norm_force = obj_norm_force_range
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

    def test_handle_force_extra(self):
        # Checks that it handles force in obs['extra']
        self.get_env(obs_mode='inc_force')
        obs, _ = self.envs.reset()
        self.envs = NForceWrapper(self.envs, 5, obs)
        for i in range(5):
            obs, _, _, _, _ = self.envs.step(None)

        data = obs['extra']['force']
        assert data.size() == (2,5*3), f"Force tensor size is {data.size()}, but should be (2,5,3)"
        
        for i in range(5):
            assert torch.all( data[:,3*i:3*(1+i)] == 5-i ), f'{data[:,3*i:3*(1+i)]} should all be {i+1}'

    def test_handle_force(self):
        # Checks that it handles force in obs
        self.get_env(obs_mode='inc_force')
        obs, _ = self.envs.reset()
        self.envs = NForceWrapper(self.envs, 5, obs)
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, rgb=False, depth=False, state=True, force=True)
        for i in range(5):
            obs, _, _, _, _ = self.envs.step(None)

        data = obs['force']
        assert data.size() == (2,5,3), f"Force tensor size is {data.size()}, but should be (2,5,3)"
        
        for i in range(5):
            assert torch.all( data[:,4-i,:] == i+1 ), f'{data[:,4-i,:]} should all be {i+1}'


    def test_no_force(self):
        # ensures that error is thrown when force not present
        self.get_env(obs_mode='state')
        obs, _ = self.envs.reset()
        with self.assertRaises(NotImplementedError):
            self.envs = NForceWrapper(self.envs, 5, obs)

        self.get_env(obs_mode='state_dict')
        obs, _ = self.envs.reset()
        with self.assertRaises(NotImplementedError):
            self.envs = NForceWrapper(self.envs, 5, obs)

    def test_FOR_ns(self):
        # Ensures works with FOR env
        self.get_env(env_id='ForeignObjectRemoval-v1', obs_mode='state_dict')
        self.envs = IncForceWrapper(self.envs)
        obs, _ = self.envs.reset()
        self.envs = NForceWrapper(self.envs, 5, obs)
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, rgb=False, depth=False, state=True, force=True)
        for i in range(5):
            obs, _, _, _, _ = self.envs.step(None)

        data = obs['force']
        assert data.size() == (2,5,3), f"Force tensor size is {data.size()}, but should be (2,5,3)"
        
        for i in range(5):
            assert torch.all( data[:,4-i,:] == i+1 ), f'{data[:,4-i,:]} should all be {i+1}'


    def test_ppih_ns(self):
        # Ensures works with FOR env
        self.get_env(env_id='FragilePegInsert-v1', obs_mode='state_dict')
        self.envs = IncForceWrapper(self.envs)
        obs, _ = self.envs.reset()
        self.envs = NForceWrapper(self.envs, 5, obs)
        self.envs = FlattenRGBDFTObservationWrapper(self.envs, rgb=False, depth=False, state=True, force=True)
        for i in range(5):
            obs, _, _, _, _ = self.envs.step(None)

        data = obs['force']
        assert data.size() == (2,5,3), f"Force tensor size is {data.size()}, but should be (2,5,3)"
        
        for i in range(5):
            assert torch.all( data[:,4-i,:] == i+1 ), f'{data[:,4-i,:]} should all be {i+1}'

        