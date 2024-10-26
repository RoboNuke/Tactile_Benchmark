
import sapien.core as sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.envs.utils import randomization
import numpy as np
import torch
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)
from typing import Any, Dict, Union, List, Optional


@register_env("ForeignObjectRemoval-v1", max_episode_steps=50)
class FOREnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_force"]

    # scene vars
    scene: ManiSkillScene
    agent: Union[Panda, Fetch]
    table_top: Actor
    TABLE_DMG_FORCE: float = 25.0
    OBJ_STACK: int = 1

    # reconfig vars
    OBJ_LW_MIN = 0.01#0.05 #length/width
    OBJ_LW_MAX = 0.05
    OBJ_H_MIN = 0.01#0.05
    OBJ_H_MAX = 0.05
    TABLE_TOP_THICKNESS = 0.01

    # init vars
    OBJ_SPAWN_RADIUS = 0.05
    OBJ_NORMAL_FORCE_MIN: float = 5.0
    OBJ_NORMAL_FORCE_MAX: float = 25.0

    def __init__(self, *args, 
                 dmg_force = 25.0,
                 obj_norm_force = [0.5, 25.0], 
                 stack = 1,
                 obs_mode= 'state',
                 **kwargs):
        self.OBJ_STACK = stack
        assert(len(obj_norm_force) == 2) # only a min and max
        # handle the ft stuff
        self.return_force_data = True
        if 'no_ft' in obs_mode:
            self.return_force_data = False
            obs_mode=obs_mode[:-6]

        # set table vars
        self.TABLE_DMG_FORCE = dmg_force
        self.OBJ_NORMAL_FORCE_MIN = obj_norm_force[0]
        self.OBJ_NORMAL_FORCE_MAX = obj_norm_force[1]
        self.objs = []
        self.kin_objs = []
        super().__init__(*args, obs_mode=obs_mode, **kwargs)

    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.6], 
            target=[-0.1, 0, 0.1]
        )
        return [
            CameraConfig("base_camera", 
                        pose, 128, 128, 
                        np.pi / 2, 0.01, 100
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            [0.6, 0.7, 0.6], 
            [0.0, 0.0, 0.35]
        )
        return CameraConfig(
            "render_camera", 
            pose, 512, 512, 1, 0.01, 100
        )
    
    def _load_scene(self, options: dict):
        """
            When reconfigure:
            - build table top scene
            - reset tabletop
            - change obj sizes
        """
        self.kin_objs.clear()
        self.objs.clear()
        with torch.device(self.device):
            # build table
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # build flat tabletop
            half_sizes = np.zeros((3))#, dtype=torch.float32, device=self.device)
            half_sizes[0] = 4 * self.OBJ_SPAWN_RADIUS
            half_sizes[1] = 4 * self.OBJ_SPAWN_RADIUS
            half_sizes[2] = self.TABLE_TOP_THICKNESS/2.0
            table_tops = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    half_size=half_sizes
                    #material=None, #physical_material,
                    #density=10*self.MAX_FOOD_DENSITY
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
            
            # build objs
            self.objs = []
            wls = randomization.uniform(
                low=self.OBJ_LW_MIN, 
                high=self.OBJ_LW_MAX,  
                size=(self.num_envs, 2)
            )
            self.hs = randomization.uniform(
                low = self.OBJ_H_MIN,
                high = self.OBJ_H_MAX,
                size = (self.num_envs,)
            )
            
            half_sizes = torch.zeros((self.num_envs, 3), dtype=torch.float32)
            half_sizes[:,:2] = wls/2.0
            half_sizes[:,2] = self.hs/(2*self.OBJ_STACK) # EACH OBJ equal size for now
            
            colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]]
            for stack_idx in range(self.OBJ_STACK):
                obs = []
                obs_kin = []
                color = colors[stack_idx]
                for i in range(self.num_envs):
                    scene_idxs = [i]
                    # get random size for the barnicle
                    physical_material: sapien.PhysicalMaterial = self.scene.sub_scenes[0].create_physical_material(
                        static_friction=0.01,
                        dynamic_friction=0.01,
                        restitution=0.5,
                    )
                    builder = self.scene.create_actor_builder()
                    builder.add_box_collision(
                        half_size=half_sizes[i,:].cpu().numpy(),
                        material=physical_material,
                        #density=ds[i]/self.BARNICLE_HEIGHT_FRAC[i] if  i == 0 else ds[i]
                    )

                    
                    builder.add_box_visual(
                        half_size=half_sizes[i,:].cpu().numpy(),
                        material=sapien.render.RenderMaterial(
                            base_color=color,
                        )
                    )
                    builder.set_scene_idxs(scene_idxs)
                    ob = builder.build(name=f"obj_{stack_idx}_{i}")
                    obs.append(ob)
                    obs_kin.append(builder.build_kinematic(name=f"obj_kin_{stack_idx}_{i}"))
                    
                self.objs.append(Actor.merge(obs, f"obj_{stack_idx}"))
                self.kin_objs.append(Actor.merge(obs_kin, f'obj_kin_{stack_idx}'))
            self.step_moved = torch.zeros((self.num_envs, self.OBJ_STACK), dtype=torch.int32, device=self.device)
            self.objs_normal_forces = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
            self.obj_starts = torch.zeros((self.num_envs,3), dtype=torch.float32, device=self.device)
            
    def _initialize_episode(self, 
                            env_idx: torch.Tensor, 
                            options: dict):
        """
            When reset
            - reinit table top scene
            - reset robot
            - reset table top (may not be needed)
            - set obj pose
            - set obj normal force 
            - reset if moved 
        """
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


            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            r = randomization.uniform(
                low=0.0, 
                high=self.OBJ_SPAWN_RADIUS, 
                size=(b, )
            )
            theta = randomization.uniform(
                low = 0.0, 
                high = 3.14159, 
                size = (b,)
            )
            pos = torch.zeros((b,3), dtype=torch.float32, device=self.device)
            pos[:,0] = r * torch.cos(theta)
            pos[:,1] = r * torch.sin(theta)
            pos[:,2] = self.TABLE_TOP_THICKNESS
            self.obj_starts[env_idx,:2] = pos[:,:2]
            self.obj_starts[env_idx, 2] = self.TABLE_TOP_THICKNESS
            self.step_moved[env_idx] = -1
            print(self.kin_objs)
            for stack_idx in range(self.OBJ_STACK):
                pos[:,2] += self.hs[env_idx]/(self.OBJ_STACK * 2.0)
                print(pos.size(), quat.size())
                self.kin_objs[stack_idx].set_pose(Pose.create_from_pq(pos, quat))
                self.objs[stack_idx].set_pose(sapien.Pose([10,10,0.0]))
                
            self.objs_normal_forces[env_idx] = randomization.uniform(
                low=self.OBJ_NORMAL_FORCE_MIN,
                high=self.OBJ_NORMAL_FORCE_MAX,
                size=(b,)
            )

            self.max_table_force[env_idx] *= 0.0
            print("init env")


    def compute_net_robot_force(self, norm=False):
        l_force = self.agent.finger1_link.get_net_contact_forces()
        r_force = self.agent.finger2_link.get_net_contact_forces()
        if norm:
            return torch.linalg.norm(l_force + r_force, axis=1)
        return l_force + r_force
    
    def compute_robot_force(self, thing, norm=True):
        l_force = self.scene.get_pairwise_contact_forces(
                self.agent.finger1_link, thing
            )
        r_force = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, thing
        )
        if norm:
            return torch.linalg.norm(l_force + r_force, axis=1)
        return l_force + r_force
    

    def _after_simulation_step(self):
        """
            Checks the force on the objs
            marks step in which they started moving
            resets all other obj to init location
        """
        for stack_idx in range(self.OBJ_STACK):
            force = self.compute_robot_force(self.kin_objs[stack_idx], norm=False)
            shear =  torch.linalg.norm(force[:,:2], axis=1)
            norm = force[:,2]
            not_moved = self.step_moved[:, stack_idx] == -1
            
            moved_this_step = torch.logical_and(
                torch.logical_or(
                    shear > self.objs_normal_forces * 0.6,
                    norm > self.objs_normal_forces
                ), 
                not_moved
            )
            
            moved_idxs = torch.where(moved_this_step)
            if torch.any(moved_this_step):
                for ss_idx in range(stack_idx, self.OBJ_STACK):
                    self.step_moved[moved_this_step, ss_idx] = self.elapsed_steps[moved_this_step]

                    self.objs[stack_idx].set_state(
                        self.kin_objs[stack_idx].get_state()[moved_this_step], moved_idxs)
                    
                    self.kin_objs[stack_idx].set_pose(sapien.Pose([10,10,0.0]))
                self.scene._gpu_apply_all()

    def evaluate(self):
        """
            Success: All objs have moved + robot arm is still w/o failure
            Failure: Too many steps w/o success or Table force > max table force
        """
        out_dic = {}

        out_dic['is_robot_static'] = self.agent.is_static(0.2)
        
        out_dic['dmg_force'] = self.compute_robot_force(self.table_top)
        
        self.max_table_force = torch.maximum(
            out_dic['dmg_force'], 
            self.max_table_force
        )

        out_dic['max_dmg_force'] = self.max_table_force
        out_dic['fail'] = self.max_table_force > self.TABLE_DMG_FORCE

        out_dic['moved_objs'] = self.OBJ_STACK - torch.sum(self.step_moved == -1, axis=1)
        out_dic['success'] = torch.logical_and(
            ~torch.any(self.step_moved == -1, axis=1),
            torch.all( (self.step_moved + 10) <= self.elapsed_steps, axis=1)
        )
        out_dic['success'] *= ~out_dic['fail']
        out_dic['success'] *= out_dic['is_robot_static']
        return out_dic

    def _get_obs_extra(self, info: Dict):
        """ 
            Add force data if in obs space
            In state mode:
            - obj locations
            - TCP location
        """
        data = super()._get_obs_extra(info)
        data['tcp_pose']=self.agent.tcp.pose.raw_pose
        if 'state' in self.obs_mode:
            for i in range(self.OBJ_STACK):
                moved = ~(self.step_moved[:, i] == -1 )
                data[f'obj_{i}_pose'] = self.kin_objs[i].pose.raw_pose
                # if the obj moved, must get non-kinematic obj pose instead
                data[f'obj_{i}_pose'][moved,:] = self.objs[i].pose.raw_pose[moved,:]
        if self.return_force_data:
            data['force'] = self.compute_net_robot_force()
        return data

    def compute_dense_reward(self, 
            obs: Any, 
            action: torch.Tensor, 
            info: Dict):
        """
            Reward types:
            - reaching (dist to bottom obj) max: 0-1
            - current force on table: max (0-1): 1 => no force; 0 => >max force
            - percent of object that has moved (0-1) 
            - if all obj moved => is static (0-1)
            total max reward: 4
        """
        
        r = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.obj_starts,
            axis = 1
        )
        r += 1 - torch.tanh(5 * tcp_to_obj_dist)
        #print("\ttcp:", 1 - torch.tanh(5 * tcp_to_obj_dist))
        # how much force we are putting out
        r += 1-torch.sigmoid(-14+20*info['dmg_force'] / self.TABLE_DMG_FORCE)
        #print("\tforce:", 1-torch.sigmoid(-14+20*info['dmg_force'] / self.TABLE_DMG_FORCE))
        # how many objects moved
        r += info['moved_objs'] / self.OBJ_STACK
        #print("\tmove:", info['moved_objs'] / self.OBJ_STACK)
        r[info['fail']] = -1
        r[info['success']] = 4 # note if static is enforced in evaluate function
        return r

    def compute_normalized_dense_reward(self, 
            obs: Any, 
            action: torch.Tensor, 
            info: Dict):
        return self.compute_dense_reward(obs, action, info)/4.0
    