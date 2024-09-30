import unittest
from tasks.for_env import FOREnv
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

class TestFOR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        traj_path = "demos/ForeignObjectRemoval-v1/teleop/trajectory"

        cls.h5_file = h5py.File(traj_path+".h5", "r")
        cls.json_data = io_utils.load_json(traj_path + ".json")
        cls.max_reward = 4
        cls.env_id = "ForeignObjectRemoval-v1"
        cls.has_set = False
        cls.stacks_to_check = [1]

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
                stack=1,
                dmg_table_force=7.0,
                obj_norm_force_range=[0.5, 25.0]):
        self.num_envs = num_envs
        self.has_set = True
        self.envs = gym.make(self.env_id, 
                            num_envs=num_envs, 
                            control_mode = "pd_joint_pos",
                            sim_backend=backend,
                            robot_uids="panda",
                            #parallel_in_single_scene=True,
                            reward_mode=reward_mode,
                            obs_mode=obs_mode,
                            stack=stack,
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


    def getDemoActions(self, idx, reward_mode='sparse'):

        self.ep = self.json_data['episodes'][idx]
        
        reset_kwargs = self.ep["reset_kwargs"].copy()
        
        self.get_env(
            reward_mode=reward_mode,
            reset_kwargs=reset_kwargs,
            obs_mode="state_dict"
        )

        # set first environment state and update recorded env state
        ori_env_states = trajectory_utils.dict_to_list_of_dicts(
            self.h5_file[f"traj_{idx}"]["env_states"]
        )
        #self.envs.base_env.set_state_dict(ori_env_states[0])
        ori_env_states = ori_env_states[1:]

        # Original actions to replay
        ori_actions = self.h5_file[f"traj_{idx}"]["actions"][:]
        return ori_actions
    
    
    def test_reconfig(self):
        """
            Re-loads the scene and 
            checks (asumes single obj)
            - obj sizes
            - table top location
            - table top size
        """
        
        for stacks in self.stacks_to_check:
            self.get_env(stack=stacks)
            uw = self.envs.unwrapped
            
            for i in range(100):
                self.envs.reset(options={"reconfigure":True})
                # get obj properties
                half_sizes = torch.zeros((2, 3), dtype=torch.float32, device=uw.device)
                assert(len(uw.objs) > 0)
                for env_idx, body in enumerate(uw.objs[0]._objs):

                    cb = body.find_component_by_type(physx.PhysxRigidDynamicComponent)
                    cs = cb.get_collision_shapes()[0]
                    half_sizes[env_idx,:] = torch.from_numpy(cs.get_half_size())

                if stacks > 1:
                    for stack_idx in range(1,stacks):
                        for env_idx, body in enumerate(uw.objs[stack_idx]._objs):
                            cb = body.find_component_by_type(physx.PhysxRigidDynamicComponent)
                            cs = cb.get_collision_shapes()[0]
                            assert torch.all( half_sizes[env_idx,:] == torch.from_numpy(cs.get_half_size()).cuda()), "Object Stacks are not the same size"

                assert torch.all(half_sizes[:,:2].cpu() <= uw.OBJ_LW_MAX/2.0), f"Stack [{stacks}] Obj width/length too large {uw.OBJ_LW_MAX},{half_sizes[:,:2]}"
                assert torch.all(half_sizes[:,:2].cpu() >= uw.OBJ_LW_MIN/2.0), f"Stack [{stacks}] Obj width/length too small {uw.OBJ_LW_MIN},{half_sizes[:,:2]}"

                assert torch.all(half_sizes[:,2]*2.0 <= uw.OBJ_H_MAX/stacks), f"Stack [{stacks}] Obj height too large {uw.OBJ_H_MAX/stacks}, {half_sizes[:,2]*2.0}"
                assert torch.all(half_sizes[:,2]*2.0 >= uw.OBJ_H_MIN/stacks), f"Stack [{stacks}] Obj height too small {uw.OBJ_H_MIN/stacks}, {half_sizes[:,2]*2.0}"
            self.envs.close()
            self.has_set = False
    
    def test_scene_init(self):
        """
            Reloads scene to check
            - obj pose in range
            - obj normal force in range
            - if moved set to false
        """
        for stacks in self.stacks_to_check:
            self.get_env(stack=stacks)
            uw = self.envs.unwrapped
            
            for i in range(100):
                self.envs.reset()
                assert(len(uw.objs) > 0)
                rs = torch.linalg.norm(uw.kin_objs[0].pose.p[:,:2], axis=1)
                assert torch.all( rs <= uw.OBJ_SPAWN_RADIUS), f"Stack [{stacks}] Obj spawned outside radius"
                assert torch.all( uw.step_moved == -1), f"Stack {stacks} move flag not reset {uw.step_moved}"
                if stacks > 1:
                    for stack_idx in range(1,stacks):
                        srs = torch.linalg.norm(uw.kin_objs[stack_idx].pose.p[:,:2], axis=1)
                        assert self.float_equ(rs, srs), f"Stack [{stacks}] Obj not stacked {rs} {srs}"

            self.envs.close()
            self.has_set = False

    def float_equ(self, a, b):
        return torch.all(torch.logical_and((a-b) > -0.0001, (a-b) < 0.0001))
    
    def test_partial_scene_init(self):
        """
            Same as test_scene_init except 
            reseting a subset of scenes
        """
        self.get_env()
        uw = self.envs.unwrapped
        
        for i in range(100):
            self.envs.step(None)
            old_rs = torch.linalg.norm(uw.kin_objs[0].pose.p[:,:2], axis=1)
            self.envs.reset(options={'env_idx':[0]})
            new_rs = torch.linalg.norm(uw.kin_objs[0].pose.p[:,:2], axis=1)
            assert not self.float_equ(old_rs[0], new_rs[0]), 'Obj did not move when reset'
            assert self.float_equ(old_rs[1], new_rs[1]), 'Obj in scene 1 moved'
            assert uw.elapsed_steps[1] > uw.elapsed_steps[0], 'Scene 1 elapsed steps not larger than Scene 0'

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
        self.envs.reset()
        self.envs.step(None)

        uw = self.envs.unwrapped
        # move all objs
        for barn in self.envs.unwrapped.objs:
            barn.set_pose(sapien.Pose([10,10,0]))

        # set their move step
        uw.step_moved[:,:] = 2

        for i in range(15):
            self.envs.step(None)
        # eval
        info = self.envs.unwrapped.evaluate()

        assert torch.all(info['success']), f"Success not evaluated correctly {info['success']}"
        assert torch.all(~info['fail']), 'Failure not evaluated correctly'
        assert torch.all(info['is_robot_static']), 'Robot is moving'
        assert torch.all(info['max_dmg_force'] < self.envs.unwrapped.TABLE_DMG_FORCE), 'Table is damaged'

    def test_evaluate_fail_table_force(self):
        # check failure by max table force exceeded
        self.get_env()
        self.envs.reset()
        
        for i in range(50):
            if i == 25: # halfway  "fail" in env 1
                self.envs.unwrapped.max_table_force[0] += self.envs.unwrapped.TABLE_DMG_FORCE * 2
            if i == 40: # last step "fail" in env 2
                self.envs.unwrapped.max_table_force[1] += self.envs.unwrapped.TABLE_DMG_FORCE * 2
            
            obs, r, term, trunc, info = self.envs.step(self.envs.action_space.sample()*0)
            #print(term, trunc,  info['fail'])
            if i == 25:
                assert(term[0] and not term[1]) # only env 1 failed
                assert(info['fail'][0] and not info['fail'][1])
                self.envs.reset(options={'env_idx':[0]})
                # check env reset correctly
                assert(self.envs.unwrapped.elapsed_steps[0] == 0)
                assert(self.envs.unwrapped.elapsed_steps[1] == 26)

            if i == 26:
                assert(self.envs.unwrapped.elapsed_steps[0] == 1) # only first env reset
                assert(self.envs.unwrapped.elapsed_steps[1] == 27)

            if i >= 40: # check fail stays until reset
                assert(info['fail'][1] and not info['fail'][0])
            elif not i == 25:
                assert not info['fail'][1] and not info['fail'][0], f"Fail not reseting {info['fail']}"
            

        _, _, _, _, info = self.envs.step(self.envs.action_space.sample()*0)

        assert(info['fail'][1])
        assert(torch.all(~info['success'][1]))
        assert(torch.all(info['max_dmg_force'][1] == self.envs.unwrapped.TABLE_DMG_FORCE * 2))

    def test_dense_reward_obj_move(self):
        # ensure reward increases as barnicles are removed
        self.get_env(reward_mode='dense', stack=self.stacks_to_check[-1])
        self.envs.reset()
        _, arm_dist_offset, _,_,_ = self.envs.step(None)

        nb = self.stacks_to_check[-1]
        for i in range(nb):
            barn = self.envs.unwrapped.objs[i]
            barn.set_pose(sapien.Pose([10,10,0]))
            obs, reward, _,_,info = self.envs.step(None)
            norm_reward = self.envs.unwrapped.compute_normalized_dense_reward(obs, None, info)
            for k in range(self.num_envs):
                assert reward[k] - arm_dist_offset[k] - (i+1)/nb < 0.001, f'Reward is {reward[k]}, but should be {i+1/nb}'
                assert norm_reward[k] - arm_dist_offset[k]/4 - (i+1)/(4*nb) < 0.001, f'Normalized Reward is {norm_reward[k]}, but should be {i+1/(4*nb)}'

    def test_dense_reward_force_on_table(self):
        # ensure reward increases as barnicles are removed
        self.get_env(reward_mode='dense')
        obs = self.envs.reset()
        #print("init")
        _, arm_dist_offset, _,_,_ = self.envs.step(self.envs.action_space.sample()*0)

        arm_dist_offset -= 1.0 # have to subtract one since zero force = 1 reward

        #print("Arm offset", arm_dist_offset)
        #print("No force")
        self.table_force(obs, arm_dist_offset, 0.0, 1.0)
        #print("half")
        self.table_force(obs, arm_dist_offset, 0.5*self.envs.unwrapped.TABLE_DMG_FORCE, 0.98201)
        #print("full")
        self.table_force(obs, arm_dist_offset, self.envs.unwrapped.TABLE_DMG_FORCE, 0.00247, fail=1)
        #print("double")
        self.table_force(obs, arm_dist_offset, 2*self.envs.unwrapped.TABLE_DMG_FORCE, 0.0, fail=1)
        

    def table_force(self, obs, arm_offset, table_force, result, fail=0):
        info = {
            "fail": torch.zeros((self.num_envs), device="cuda:0", dtype=torch.bool),
            "success": torch.zeros((self.num_envs), device="cuda:0", dtype=torch.bool),
            'moved_objs': torch.zeros((self.num_envs), device="cuda:0"),
            "is_robot_static": torch.ones((self.num_envs), device="cuda:0"),
            "max_dmg_force": torch.ones((self.num_envs), device="cuda:0") * table_force,
            "dmg_force": torch.ones((self.num_envs), device="cuda:0") * table_force
        }
        #print("sep:", self.envs.unwrapped.compute_dense_reward(obs, None, info), arm_offset)
        r = self.envs.unwrapped.compute_dense_reward(obs, None, info) - arm_offset
        nr = self.envs.unwrapped.compute_normalized_dense_reward(obs, None, info) - arm_offset/self.max_reward
        #print("final:", r, result)
        
        assert self.float_equ(r, result), f'r is {r}, but should be result:{result}'

        # check normalized
        assert self.float_equ(nr, result/self.max_reward), f'normalized r is {nr} but should be {result/self.max_reward}'

    def test_arm_table_collision(self):
        """
            Plays demo that runs arm into table
            checks
            - max dmg force is propogated
            - failure on max dmg 
            - tcp force shows similar force
        """
        ori_actions  = self.getDemoActions(0)

        info = {}
        uw = self.envs.unwrapped
        n = len(ori_actions)
        if len(uw.objs) > 0:
            self.envs.unwrapped.objs[0].set_pose(sapien.Pose([10,10,0])) # get this out of here
    
        info = None
        truncated = None
        reward = None
        max_dmg_force = torch.zeros((2),dtype=torch.float32, device=self.envs.unwrapped.device)
        for t, a in enumerate(ori_actions):
            
            sa = torch.tensor(np.array([a for k in range(self.num_envs)]))
            obs, reward, terminated, truncated, info = self.envs.step(sa)
            
            max_dmg_force = torch.maximum(info['dmg_force'], max_dmg_force)
            
            assert torch.all(max_dmg_force == info['max_dmg_force']), "Max dmg force not propigating"

            if torch.any(info['max_dmg_force'] > 25.0):
                dmg_mask = info['max_dmg_force'] > 25.0
                assert torch.all(info['fail'][dmg_mask]), f"Not recognizing max force failure, {info['max_dmg_force']}\n{info['fail']}"
            
            fn = torch.linalg.norm(obs['extra']['force'], axis=1)
            df = info['dmg_force']
            assert self.float_equ(df, fn), f"TCP force match {df}\n{fn}"

            #self.envs.base_env.render_human()
            #time.sleep(0.1)
            #if terminated[0]: # running the same action so should be same
            #    break
        assert torch.all(info['fail']), f"Fail did not flag correctly {info['fail']}"
        assert torch.all(~info['success']), f"Success did not flag {info['success']}"
        assert torch.all(terminated), f"Trucation not flag {truncated}"
        assert torch.all(max_dmg_force > 0.0), f"Not getting force readings correctly {info['max_dmg_force']}"


    def test_free_obj_on_force(self):
        """
            Plays demo that hits into the obj
            - checks that obj moves only after force is large enough
            - object remains moving after hit
            - tcp force is similar during contact
        """
        ori_actions  = self.getDemoActions(1)

        uw = self.envs.unwrapped
        uw.OBJ_LW_MIN = 0.05
        uw.OBJ_LW_MAX = 0.05
        uw.OBJ_H_MIN = 0.05
        uw.OBJ_H_MAX = 0.05
        self.envs.reset(options={"reconfigure":True})
        info = {}

        n = len(ori_actions)
    
        info = None
        truncated = None
        reward = None
        #starting_pos = self.envs.unwrapped.obj_starts[0]
        for t, a in enumerate(ori_actions):
            
            sa = torch.tensor(np.array([a for k in range(self.num_envs)]))
            obs, reward, terminated, truncated, info = self.envs.step(sa)
            
            #print(
            #    self.env.scene.get_pairwise_contact_forces(self.env.peg, self.env.box)
            #)
            #self.envs.base_env.render_human()
            #time.sleep(0.1)
            if terminated[0]: # running the same action so should be same
                break
        #end_pos = self.envs.unwrapped.objs[0].pose.p
        #print(starting_pos, end_pos)
        assert ~info['fail'][0]
        assert info['success'][0]
        self.envs.close()
        return info, truncated, terminated, reward