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

from mani_skill.envs.utils.observations import (
    parse_visual_obs_mode_to_struct,
    sensor_data_to_pointcloud,
)

from mani_skill.utils.scene_builder.table import TableSceneBuilder


@register_env("ScrapBarnicle-v1", max_episode_steps=50)
class ScrapBarnicleEnv(BaseEnv):

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

    barnicles: List[Actor] = []
    BARNICLE_SPAWN_CENTER: List[float] = [0.0,0.0]
    BARNICLE_SPAWN_DELTA: float = 0.1

    BARNICLE_FRICTION: float = 1.0
    BARNICLE_FRICTION_DELTA: float = 0.0

    BARNICLE_DENSITY: float = 13850.0
    BARNICLE_DENSITY_DELTA: float = 150.0

    BARNICLE_RADIUS: float = 0.061# 0.04  # 0.02 - .102
    BARNICLE_RADIUS_DELTA: float = 0.041

    BARNICLE_HEIGHT: float = 0.06
    BARNICLE_HEIGHT_DELTA: float = 0.041

    NUM_BARNICLES: int = 4
    BARNICLE_HEIGHT_FRAC: torch.Tensor = torch.Tensor([0.1, 0.30, 0.30, 0.30])

    FORCE_DMG_TABLE: float
    def __init__(self, *args, 
                 robot_uids="panda", 
                 dmg_table_force = 25.0,  
                 barnicle_friction = 1000.0, 
                 obs_mode= 'state',
                 **kwargs):
        
        if not obs_mode in self.SUPPORTED_OBS_MODES:
            raise NotImplementedError(obs_mode)
        self.return_force_data = True
        if 'no_ft' in obs_mode:
            self.return_force_data = False
            obs_mode=obs_mode[:-6]

        super().__init__(*args, obs_mode=obs_mode, robot_uids=robot_uids, **kwargs)


        self.FORCE_DMG_TABLE = dmg_table_force
        self.BARNICLE_FRICTION = barnicle_friction

        self.max_table_force = torch.zeros(self.num_envs)
    

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # randomly select "radius" and total height for barnicles
        maxH = self.BARNICLE_HEIGHT + self.BARNICLE_HEIGHT_DELTA
        minH = self.BARNICLE_HEIGHT - self.BARNICLE_HEIGHT_DELTA
        
        self.barnicle_hs = randomization.uniform(
                low=minH, 
                high=maxH,  
                size=(self.num_envs,)
            )

        maxR = self.BARNICLE_RADIUS + self.BARNICLE_RADIUS_DELTA
        minR = self.BARNICLE_RADIUS - self.BARNICLE_RADIUS_DELTA
        rs = randomization.uniform(
                low=minR, 
                high=maxR,  
                size=(self.num_envs,)
            )

        half_sizes = torch.zeros((self.num_envs, 3))
        half_sizes[:,0] = rs/2.0
        half_sizes[:,1] = rs/2.0

        # get dynamic parameters
        maxD = self.BARNICLE_DENSITY + self.BARNICLE_DENSITY_DELTA
        minD = self.BARNICLE_DENSITY - self.BARNICLE_DENSITY_DELTA
        ds = randomization.uniform(
                low=minD, 
                high=maxD,  
                size=(self.num_envs,)
            )

        maxF = self.BARNICLE_FRICTION + self.BARNICLE_FRICTION_DELTA
        minF = self.BARNICLE_FRICTION - self.BARNICLE_FRICTION_DELTA
        fs = randomization.uniform(
                low=minF, 
                high=maxF,  
                size=(self.num_envs,)
            )

        colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]]
        
        for barn_idx in range(self.NUM_BARNICLES):
            new_barns = []
            color = colors[barn_idx]
            half_sizes[:,2] = self.BARNICLE_HEIGHT_FRAC[barn_idx] * self.barnicle_hs/2.0
            for i in range(self.num_envs):
                scene_idxs = [i]

                # get random size for the barnicle
                physical_material: sapien.PhysicalMaterial = self.scene.sub_scenes[0].create_physical_material(
                    static_friction=fs[i]/(1+barn_idx),
                    dynamic_friction=fs[i]/(1+barn_idx),
                    restitution=0.5,
                )
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    half_size=half_sizes[i,:],
                    material=None, #physical_material,
                    density=ds[i]#*(1-self.BARNICLE_HEIGHT_FRAC[i])
                )

                builder.add_box_visual(
                    half_size=half_sizes[i,:],
                    material=sapien.render.RenderMaterial(
                        base_color=color,
                    )
                )
                builder.set_scene_idxs(scene_idxs)
                barn = builder.build(name=f"barnicle_{barn_idx}_{i}")
                #print(barn.angular_damping, barn.linear_damping)
                barn.set_angular_damping(10.0)
                #barn.set_linear_damping(10.0)
                #print(barn.angular_damping, barn.linear_damping)

                new_barns.append(barn)
            self.barnicles.append(Actor.merge(new_barns, f"barnicle_{barn_idx}"))
        

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
            ) * 0
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # initialize the barnicles
            xy = randomization.uniform(
                low=torch.tensor(self.BARNICLE_SPAWN_CENTER) - self.BARNICLE_SPAWN_DELTA, 
                high=torch.tensor(self.BARNICLE_SPAWN_CENTER) + self.BARNICLE_SPAWN_DELTA,  
                size=(b, 2)
            )
            pos = torch.zeros((b,3), dtype=torch.float32, device=self.device)
            pos[:,:2] = xy

            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            h_offsets = torch.zeros((b), dtype=torch.float32, device=self.device)
            self.barnicle_hs = self.barnicle_hs.to(device=self.device)
            
            for i, barns in enumerate(self.barnicles):
                pos[:,2] = h_offsets + self.barnicle_hs * self.BARNICLE_HEIGHT_FRAC[i]/2.0
                barns.set_pose(Pose.create_from_pq(pos, quat))
                h_offsets += self.barnicle_hs * self.BARNICLE_HEIGHT_FRAC[i]

    def evaluate(self):
        """ Returns info with:
                fail (table damaged or barnicle attached on final step)
                success (all barnicles detatched and is_robot_static)
                attach_barnicles 
                is_robot_static
                max_table_force
                table_force: current force on table (for reward)
        """
        is_robot_static = self.agent.is_static(0.2)

        return {
            "fail": torch.zeros(self.num_envs),
            "success": torch.ones(self.num_envs),
            'attached_barnicles':  torch.zeros(self.num_envs),
            "is_robot_static": is_robot_static,
            "max_table_force": torch.zeros(self.num_envs),
            "table_force":  torch.zeros(self.num_envs)
        }

    def get_obs(self, info: Optional[Dict]=None):

        """
        Return the current observation of the environment. User may call this directly to get the current observation
        as opposed to taking a step with actions in the environment.

        Note that some tasks use info of the current environment state to populate the observations to avoid having to
        compute slow operations twice. For example a state based observation may wish to include a boolean indicating
        if a robot is grasping an object. Computing this boolean correctly is slow, so it is preferable to generate that
        data in the info object by overriding the `self.evaluate` function.

        Args:
            info (Dict): The info object of the environment. Generally should always be the result of `self.get_info()`.
                If this is None (the default), this function will call `self.get_info()` itself
        """
        if info is None:
            info = self.get_info()
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return dict()
        elif self._obs_mode == "state" or self._obs_mode == "state_no_ft":
            state_dict = self._get_obs_state_dict(info)
            obs = common.flatten_state_dict(state_dict, use_torch=True, device=self.device)
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict(info)
        elif self._obs_mode == "pointcloud":
            # TODO support more flexible pcd obs mode with new render system
            obs = self._get_obs_with_sensor_data(info)
            obs = sensor_data_to_pointcloud(obs, self._sensors)
        elif self._obs_mode == "sensor_data":
            # return raw texture data dependent on choice of shader
            obs = self._get_obs_with_sensor_data(info, apply_texture_transforms=False)
        elif self._obs_mode in ["rgb","rgb_no_ft", "depth", "segmentation", "rgbd", "rgb+depth", "rgb+depth+segmentation", "depth+segmentation", "rgb+segmentation"]:
            obs = self._get_obs_with_sensor_data(info)
        else:
            raise NotImplementedError(self._obs_mode)
        return obs

    def _get_obs_extra(self, info: Dict):
        """ Adds force-torque if no_ft not in obs_mode """
        data = dict()
        if self.return_force_data:
            data['force'] = torch.zeros((self.num_envs, 3), device=self.device)
        return data

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """ Dense reward has following elements:
            - reaching (dist to barnicle) max: 0-1
            - number of barnicles attached max: 0-1
            - if moving while no barnicles attached max: 0-1
            - current force on table: max: 0-1 1 => no force 0 => >max force
            total max reward: 4 
        """
        return torch.ones(self.num_envs) * 4.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action,  info) / 4.0
