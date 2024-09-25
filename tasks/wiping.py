# high level
import torch
import sapien # sim physics env
import numpy as np
import sapien.physx as physx

# base env stuff
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

# robot stuff
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.scene import ManiSkillScene

# scene stuff
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.envs.utils import randomization

# camearas
from mani_skill.sensors.camera import CameraConfig

# utilities
from typing import Any, Dict, Union, List, Optional
from mani_skill.utils import sapien_utils, common


@register_env("WipeFood-v1", max_episode_steps=50)
class WipingEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_OBS_MODES = ("none", 
                           "state", 
                           "state_no_ft", # NEW
                           "state_dict", 
                           "sensor_data",
                           "rgb_no_ft",  #NEW 
                           "rgb", "depth", 
                           "segmentation", 
                           "rgbd", "rgb+depth", 
                           "rgb+depth+segmentation", 
                           "rgb+segmentation", 
                           "depth+segmentation", 
                           "pointcloud")
    agent: Union[Panda, Fetch]

    scene: ManiSkillScene

    # food defs
    food: Actor = None
    MIN_FOOD_W: float = 0.02#0.0025
    MAX_FOOD_W: float = 0.02

    MIN_FOOD_H: float = 0.01#0.005
    MAX_FOOD_H: float = 0.01

    FOOD_SPAWN_RADIUS: float = 0.05

    MIN_FOOD_DENSITY: float = 75000.0#50000.0
    MAX_FOOD_DENSITY: float = 75000.0

    MIN_FOOD_FRIC: float = 0.4
    MAX_FOOD_FRIC: float = 0.4#0.8

    #MIN_NUM_FOOD: int = 2
    #MAX_NUM_FOOD: int = 10

    table_top: Actor
    FORCE_DMG_TABLE: float = 2.0
    TABLE_TOP_THICKNESS: float = 0.01
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def __init__(self, *args, obs_mode='state', **kwargs):
        # handle the ft stuff
        self.return_force_data = True
        if 'no_ft' in obs_mode:
            self.return_force_data = False
            obs_mode=obs_mode[:-6]

        super().__init__(*args, obs_mode=obs_mode, **kwargs)

    
    def _load_scene(self, options: dict):
        # build table
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # build flat tabletop
        half_sizes = torch.zeros((3))
        half_sizes[0] = 2 * self.FOOD_SPAWN_RADIUS
        half_sizes[1] = 2 * self.FOOD_SPAWN_RADIUS
        half_sizes[2] = self.TABLE_TOP_THICKNESS/2
        table_tops = []
        for i in range(self.num_envs):
            scene_idxs = [i]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                half_size=half_sizes,
                material=None, #physical_material,
                density=10*self.MAX_FOOD_DENSITY
            )

            builder.add_box_visual(
                half_size=half_sizes,
                material=sapien.render.RenderMaterial(
                    base_color=[1,1,1,1],
                )
            )

            builder.set_scene_idxs(scene_idxs)
            table_tops.append(builder.build_kinematic(name=f"table_top_{i}"))
        self.table_top = Actor.merge(table_tops, name='table_top')
        self.max_table_force = torch.zeros((self.num_envs), 
                                           dtype=torch.float32, 
                                           device=self.device
        )

        # define food sizes
        self.food_hs = randomization.uniform(
                low=self.MIN_FOOD_H, 
                high=self.MAX_FOOD_H,  
                size=(self.num_envs,)
            )
        food_ws = randomization.uniform(
                low=self.MIN_FOOD_W, 
                high=self.MAX_FOOD_W,  
                size=(self.num_envs,)
            )
        food_ls = randomization.uniform(
                low=self.MIN_FOOD_W, 
                high=self.MAX_FOOD_W,  
                size=(self.num_envs,)
            )
        food_ds = randomization.uniform(
                low=self.MIN_FOOD_DENSITY, 
                high=self.MAX_FOOD_DENSITY,  
                size=(self.num_envs,)
            )
        food_us = randomization.uniform(
                low=self.MIN_FOOD_FRIC, 
                high=self.MAX_FOOD_FRIC,  
                size=(self.num_envs,)
            )

        half_sizes = torch.zeros((self.num_envs,3))
        half_sizes[:,0] = food_ws
        half_sizes[:,1] = food_ls
        half_sizes[:,2] = self.food_hs
        
        new_foods = []
        # build food particles
        for i in range(self.num_envs):
            scene_idxs = [i]
            # get random size for the barnicle
            physical_material: sapien.PhysicalMaterial = self.scene.sub_scenes[i].create_physical_material(
                static_friction=food_us[i],
                dynamic_friction=food_us[i],
                restitution=0.5,
            )
            builder = self.scene.create_actor_builder()
            
            builder.add_box_collision(
                half_size=half_sizes[i,:],
                material=physical_material,
                density=food_ds[i]
            )
            builder.add_box_visual(
                half_size=half_sizes[i,:],
                material=sapien.render.RenderMaterial(
                    base_color=[1.0,0,0,1],
                )
            )
            builder.set_scene_idxs(scene_idxs)
            food = builder.build(name=f"food_{i}")
            #print(barn.angular_damping, barn.linear_damping)
            #barn.set_angular_damping(10.0)
            #barn.set_linear_damping(10.0)
            #print(barn.angular_damping, barn.linear_damping)

            new_foods.append(food)
        self.food = Actor.merge(new_foods, name='food')
        self.food_starts = torch.zeros( (self.num_envs,2), 
                                        dtype=torch.float32, 
                                        device=self.device )
        self.food_hs = self.food_hs.to(self.device)
        self.food_weights = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # init table
            self.table_scene.initialize(env_idx)

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            ) 
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # set tabletop
            ttpos = torch.zeros((b,3), dtype=torch.float32, device=self.device)
            ttqs = torch.zeros((b,4), dtype=torch.float32, device=self.device)
            ttpos[:,2] = self.TABLE_TOP_THICKNESS / 2.0
            ttqs[:,3] = 1.0
            self.table_top.set_pose(Pose.create_from_pq(ttpos, ttqs))

            # place food bits
            r = randomization.uniform(
                low=0.0, 
                high=self.FOOD_SPAWN_RADIUS,  
                size=(b, )
            )
            t = randomization.uniform(
                low=0.0, 
                high=6.28318,  
                size=(b, )
            )
            pos = torch.zeros((b,3), dtype=torch.float32, device=self.device)
            pos[:,0] = r * torch.cos(t)
            pos[:,1] = r * torch.sin(t)
            
            self.food_starts[env_idx,:] = pos[:,:2]
            pos[:,2] = self.food_hs[env_idx] + self.TABLE_TOP_THICKNESS 
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            
            self.food.set_pose(Pose.create_from_pq(pos, quat))

            # reinitialize state vars
            self.max_table_force[env_idx] *= 0
           
    def _get_obs_extra(self, info: Dict):
        """ 
            Adds force-torque if no_ft not in obs_mode
            handles state calculations
        """
        data = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            # food location
            data['food_location'] = self.food.pose.raw_pose

        if self.return_force_data:
            data['force'] = self.agent.tcp.get_net_contact_forces()
        return data
    
    def evaluate(self):
        """
            Fail Conditions: table force larger than damage table
            Success Condition: All food removed from scene
        """
        is_robot_static = self.agent.is_static(0.2)

        # get current force on table
        table_force = torch.linalg.norm(
                        self.scene.get_pairwise_contact_forces(self.table_top, self.agent.tcp),
                        axis=1
        )
        # get max force on table this episode
        self.max_table_force = torch.maximum(table_force, self.max_table_force)
        
        # determine failure
        fail = self.max_table_force >= self.FORCE_DMG_TABLE

        succ = torch.linalg.norm(
                self.food.pose.raw_pose[:,:2] - self.food_starts, axis=1
                ) > 0.001
        
        succ *= (~fail)

        return {
            "fail": fail,
            "success": succ,
            "is_robot_static": is_robot_static,
            "max_dmg_force": self.max_table_force,
            "dmg_force": table_force
        }
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
            Reward types:
            - reaching (dist to food) max: 0-1
            - current force on table: max (0-1): 1 => no force; 0 => >max force
            total max reward: 2 
        """

        #print(info['table_force'])
        r = -(1-torch.sigmoid(14-20*info['dmg_force'] / self.FORCE_DMG_TABLE))

        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.food.pose.p,
            axis = 1
        )
        
        r += 1 - torch.tanh(5 * tcp_to_obj_dist)

        r[info['success']] = 2
        r[info['fail']] = -1
        return r
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action,  info) / 2.0
    

    
