from dataclasses import dataclass
from typing import Sequence, Union
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode

from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig, DictController, CombinedController

from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.agents.base_agent import *
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Articulation
import sapien.physx as physx
from controllers.torque import *

class HybridForcePosController(TorqueJointController):
    config: "HybridForcePosControllerConfig"
    _target = None
    _start_pos = None
    _old_f_error = None
    _old_pos_error = None
    
    def __init__(
        self,
        config: "ControllerConfig",
        articulation: Articulation,
        control_freq: int,
        sim_freq: int = None,
        scene: ManiSkillScene = None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq, scene)
    
    def _initialize_joints(self):
        # init the kinematics for jacobian
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )

    @property 
    def jacobian(self):
        return self.kinematics.pk_chain.jacobian(self.qpos)

    def _initialize_action_space(self):
        # action space is n floats and n bools
        # if true => float is force to control
        # else => float is a position controller
        if len(self.config.force_lower) == 1:
            low = np.ones((3,)) * self.config.force_lower
            high = np.ones((3,)) * self.config.force_higher
        elif len(self.config.force_lower) > 1:
            low = self.config.force_lower
            high = self.config.force_higher
        else:
            raise NotImplementedError, 'Size of force limits must be >= 1'
        
        if self.config.use_force_only:
            self.single_action_space = spaces.Box(low, high, dtype=np.float32)
        else:
            low =  np.append(low,  [0.0, 0.0, 0.0])
            high = np.append(high, [1.0, 1.0, 1.0])
            self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        # calls joint torque driver property
        # handles hybrid specific properties
        super.set_drive_property()
        self.dt = self.config.dt
        self.pos_stiffness = torch.diag(self.config.pos_stiffness)
        self.pos_damping = torch.diag(self.config.pos_damping)
        self.force_stiffness = torch.diag(self.config.force_stiffness)
        self.force_damping = torch.diag(self.config.force_damping)

    def set_action(self, action: Array):
        """
            Convert action to tensor
            Any preprocessing of action 
            use jacobian to convert action to target_qf
        """
        self._old_f_error*=0.0
        if  self.config.use_force_only:
            self._target = action
            self._old_f_error = torch.zeros_like(self._target)
            self.S = torch.eye(3)
            self.s_idxs = [1,1,1]
        else:
            self._old_pos_error*=0.0
            self.s_idxs = action[:,:3] > 0.5
            self.S = torch.diag(action[:,:3] > 0.5)
            self._target = action[:,3:]
            self._old_f_error = torch.zeros_like(self._target[:,3:])[:, self.s_idxs]
            self._old_pos_error = torch.zeros_like(self._target[:,3:])[:, 1-self.s_idxs]
            
    def before_simulation_step(self):
        # update errors
        f_error = (self._target[:, self.s_idxs] 
                    - self.config.agent.get_ee_force()[:, self.s_idxs]
        )
        df_error = (f_error - self._old_f_error)/self.dt
        self._old_f_error = f_error.clone()

        # update goal qf
        J = self.jacobian
        self._target_qf = J[:,self.s_idxs,:].T * self.S * (
                self.force_stiffness * f_error + 
                self.force_damping * df_error
        )
        if not self.config.use_force_only:
            pos_error = (self._target[:, 1 - self.s_idxs] 
                        - self.config.agent.tcp.pose.p[:, 1-self.s_idxs]
            )

            dpos_error = (pos_error - self._old_pos_error)/self.dt 
            self._old_pos_error = pos_error.clone()
            
            self._target_qf += J[:,1-self.s_idxs,:].T * (torch.eye(3) - self.S) * (
                    self.pos_stiffness * pos_error + 
                    self.pos_damping * dpos_error
            )
        
        super().before_simulation_step()

@dataclass
class HybridForcePosControllerConfig(ControllerConfig):
    ee_link: str = None
    """The name of the end-effector link to control. Note that it does not have to be a end-effector necessarily and could just be any link."""
    urdf_path: str = None
    """Path to the URDF file defining the robot to control."""

    """ (Hunter): The following are for joint torque control"""
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    max_saturation: float = 10.0
    dt: float = 0.01
    """ (Hunter): The following is PD control for the ee force """
    force_stiffness: Union[float, Sequence[float]]
    force_damping: Union[float, Sequence[float]]
    force_lower: Union[None, float, Sequence[float]] = -500.0
    force_higher: Union[None, float, Sequence[float]] = 500.0

    """ (Hunter): The following is PD control for position goal to joint torque command"""
    pos_stiffness: Union[float, Sequence[float]]
    pos_damping: Union[float, Sequence[float]]

    """ (Hunter): Following defines controller properties """
    use_delta: bool = False
    use_target: bool = False
    use_force_only: bool = True # if false, is a ee force controller only 
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = HybridForcePosController
    agent: BaseAgent = None