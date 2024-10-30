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

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

@register_env("SimpleFragilePiH-v1", max_episode_steps=50)
class SimpleFragilePiH(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_force"]

    # scene vars
    scene: ManiSkillScene
    agent: Union[Panda, Fetch] 
                
    # env vars
    PEG_MAX_FORCE: float = 500.0
    PEG_LEN_MIN: float = 0.0425 #0.085
    PEG_LEN_MAX: float = 0.0625 #0.125
    PEG_W_MIN: float = 0.015
    PEG_W_MAX: float = 0.025
    HOLE_RADIUS = 0.05
    TIGHT_MIN = 0.75 # as percentage of peg_s
    TIGHT_MAX = 0.85
    
    # constants
    max_reward: float = 2.0


    def __init__(self, *args, 
                 dmg_force = 25.0,
                 obs_mode= 'state_dict',
                 **kwargs):
        self.PEG_MAX_FORCE = dmg_force
        self.return_force_data = True
        if 'no_ft' in obs_mode:
            self.return_force_data = False
            obs_mode=obs_mode[:-6]
        super().__init__(*args, obs_mode=obs_mode, **kwargs)
        #print(self._batched_episode_rng)

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
     
    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.inv().p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets.inv()

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets #* self.peg_head_offsets.inv()

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()
    
    def has_peg_inserted(self):
    
        gpp = self.box_hole_pose.p
        pp = self.peg_head_pose.p # self.peg.pose.p
        inserted = torch.abs(gpp[:,0]-pp[:,0]) <= self._clearance
        inserted = torch.logical_and(inserted, torch.abs(gpp[:,1]-pp[:,1]) < self._clearance)
        inserted = torch.logical_and(inserted, pp[:,2] < self.box.pose.p[:,2]) # inserted half way
        return inserted 
    """
    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )
    """
    def _load_scene(self, options: dict):
        """
            When reconfigure:
            - build table top scene
            - reset tabletop
        """
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = randomization.uniform(
                self.PEG_LEN_MIN, #0.085, 
                self.PEG_LEN_MAX, #0.125,  
                size=(self.num_envs,)
            )
            radii = randomization.uniform(
                self.PEG_W_MIN, #0.015, 
                self.PEG_W_MAX, #0.025,  
                size=(self.num_envs,)
            )
            centers = randomization.uniform(
                -0.5 * (lengths - radii)[:, None], #0.015, 
                0.5 * (lengths - radii)[:, None], #0.025,  
                size=(self.num_envs,2)
            )
            #(
            #    0.5
            #    * (lengths - radii)[:, None]
            #    * randomization.uniform(-1, 1, size=(2,))
            #)

            d = randomization.uniform(
                self.TIGHT_MIN,
                self.TIGHT_MAX,
                size=(self.num_envs,)
            )

            self._clearance = radii * (1.0 - d)/d
            # save some useful values for use later
            self.peg_half_sizes = torch.zeros((self.num_envs, 3), device=self.device)
            self.peg_half_sizes[:,0] = lengths
            self.peg_half_sizes[:,1] = radii
            self.peg_half_sizes[:,2] = radii
            #common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 0] = -lengths/2
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i].cpu().numpy()
                radius = radii[i].cpu().numpy()
                clearance = self._clearance[i].cpu().numpy()
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")

                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + clearance,
                    length*4,
                    length,
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i].cpu().numpy()
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, length/2.0])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")

                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")
        self.obsticles = [
            #self.agent.finger1_link,
            #self.agent.finger2_link,
            self.box
        ]  
        self.max_peg_force = torch.zeros((self.num_envs), device=self.device)
        self.lengths = 4*lengths

    def _initialize_episode(self, 
                            env_idx: torch.Tensor, 
                            options: dict):
        """
            When reset
            - reinit table top scene
            - reset robot
            - move peg to robot gripper
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # set box pose
            r = randomization.uniform(
                low = 0.0,
                high=self.HOLE_RADIUS,
                size=(b,)
            )

            theta = randomization.uniform(
                low=0.0,
                high=3.14,
                size=(b,)
            )
            pos = torch.zeros((b, 3))
            pos[:, 0] = r * torch.cos(theta)
            pos[:,1] = r * torch.sin(theta)
            pos[:, 2] = self.peg_half_sizes[env_idx, 1]
            quat = [0.0,0.707,0.0,0.707]
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 4 / 8,
                    0,
                    np.pi * 3 / 4, 
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = self.peg_half_sizes[env_idx,1:3].cpu()-0.01
            self.agent.robot.set_qpos(qpos)
            # ensure all updates to object poses and configurations are applied on GPU after task initialization
            
            try:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()
            except:
                pass

            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            self.peg.set_pose(
                self.agent.tcp.pose[env_idx,:] *
                sapien.Pose([0.0,0,-0.02], [0.0,-0.707,0.0,0.707]) *
                self.peg_head_offsets[env_idx,:].inv() 
            )

    def evaluate(self):
        """
            Success: All objs have moved + robot arm is still w/o failure
            Failure: Too many steps w/o success or Table force > max table force
        """
        success = self.has_peg_inserted()
        out_dic = dict(success=success)
        out_dic['fail'], out_dic['dmg_force'], out_dic['fail_cause'] = self.pegBroke()
        #out_dic['fail'] = torch.logical_or(~self.agent.is_grasping(self.peg), out_dic['fail'])
        self.max_peg_force = torch.maximum(out_dic['dmg_force'], self.max_peg_force)
        out_dic['max_dmg_force'] = self.max_peg_force
        out_dic['success'] *= ~out_dic['fail']
        #self.termed[torch.logical_or(out_dic['fail'], out_dic['success'])] = True
        return out_dic

    def _get_obs_extra(self, info: Dict):
        """ 
            Add force data if in obs space
            In state mode:
            - obj locations
            - TCP location
        """
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )

        if self.return_force_data:
            obs['force'] = self.agent.tcp.get_net_contact_forces()
        return obs

    def compute_dense_reward(self, 
            obs: Any, 
            action: torch.Tensor, 
            info: Dict):
        """
            Reward types:
            - reaching to pre insertion max: 0-1
            - actual insertion 0-1
            total max reward: 1
        """
        is_grasped = self.agent.is_grasping(self.peg)
        #is_over_box = self.over_box()
        # reach reward (if robot drops peg still gets a reward signal)
        dist_tcp = torch.linalg.norm(
            self.box_hole_pose.p[:,:2] - self.agent.tcp.pose.p[:,:2],
            axis = 1
        )

        reward = 1 - torch.tanh(5 * dist_tcp)

        # this reward encourage holding onto the peg and 
        # keeping the peg aligned vertically
        dist_xy = torch.linalg.norm(
            self.box_hole_pose.p[:,:2] - self.peg_head_pose.p[:,:2], 
            axis=1
        )

        dist2_xy = torch.linalg.norm(
            self.box_hole_pose.p[:,:2] - self.peg.pose.p[:,:2],
            axis=1
        )

        pre_insert_reward = 4 * (
            1 - torch.tanh(
                0.5 * (dist2_xy + dist_xy) + 
                4.5 * torch.maximum(dist2_xy, dist_xy)
            )
        )

        reward += pre_insert_reward * is_grasped 

        # finally reward it for pushing it down
        pre_inserted = torch.logical_and(dist_xy < 0.01, dist2_xy < 0.01)
        #dist = torch.linalg.norm(
        #    self.box_hole_pose.p - self.peg_head_pose.p, 
        #    axis=1
        #)
        dist = self.peg_head_pose.p[:,2] 
        #print(f"r:{5 * (1 - torch.tanh(5*dist))}\tz-dist:{dist}\tpi:{pre_inserted}")
        reward += 5 * (1 - torch.tanh(5*dist)) * pre_inserted

        reward[info["success"]] = 10.0
        #print(f"g:{is_grasped}\to:{is_over_box}\td:{dist}\tr:{reward}")
        return reward
    
    def compute_normalized_dense_reward(self, 
            obs: Any, 
            action: torch.Tensor, 
            info: Dict):
        return self.compute_dense_reward(obs, action, info)/10.0
    
    def over_box(self):
        bp = self.box.pose.p
        pp = self.peg.pose.p
        over = pp[:,0] < self.lengths + bp[:,0]
        over = torch.logical_and(over, pp[:,0] > (-self.lengths + bp[:,0]))
        over = torch.logical_and(over, pp[:,1] < (self.lengths + bp[:,1]))
        over = torch.logical_and(over, self.peg.pose.p[:,1] > (-self.lengths + bp[:,1]))
        return over
    
    def getPegForce(self, object: Actor):

        contact_force = self.scene.get_pairwise_contact_forces(
            self.peg, object
        )
        force = torch.linalg.norm(contact_force, axis=1)
        return force

    def pegBroke(self):
        """ Calculates the maximum force on the peg and returns it """
        max_forces = torch.zeros((self.num_envs), device=self.device)
        resp_actor = torch.zeros((self.num_envs), device=self.device)
        for i, obs in enumerate(self.obsticles):
            obs_forces = self.getPegForce(obs)
            update = obs_forces > max_forces
            idx = 1 # robot
            if i == 2:
                idx = 2 # box
                #print("\tForce on box:", obs_forces)
            resp_actor += idx * update - update * resp_actor   
            
            max_forces = torch.maximum( max_forces, obs_forces)

        brokeFlag = self.PEG_MAX_FORCE <= max_forces
        return brokeFlag, max_forces, resp_actor

    
