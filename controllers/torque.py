from dataclasses import dataclass
from typing import Sequence, Union
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import BaseController, ControllerConfig

from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

@register_agent()
class PandaForce(Panda):
    uid = "panda_force"
    def _controller_configs(self):
        base_controller_configs = super()._controller_configs()

        arm_joint_torque = TorqueJointControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,

        )

        base_controller_configs['joint_torque'] = dict(
            arm = arm_joint_torque,
            gripper = base_controller_configs['pd_joint_delta_pos']['gripper'] 
        )

        return deepcopy_dict(base_controller_configs)



class TorqueJointController(BaseController):
    config: "TorqueJointControllerConfig"
    _start_qf = None
    _target_qf = None

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices].cpu().numpy()
        )
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits
    
    def _initialize_action_space(self):
        """ 
            Creates an action space for Gym env obj
            must be a spaces.box
        """
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        """
            Set drive properties
            - stiffness
            - damping
            - force_limits
            - friction
            Then sets each joints properties
        """
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            drive_mode = self.config.drive_mode
            if not isinstance(drive_mode, str):
                drive_mode = drive_mode[i]
            joint.set_drive_properties(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=drive_mode
            )
            joint.set_friction(friction[i])

    def reset(self):
        """
            Resets the controller to an initial state. 
            This is called upon environment creation 
            and each environment reset
        """
        pass

    def set_action(self, action: Array):
        """
            Convert action to tensor
            Any preprocessing of action 
            befor setting the drive targets
        """
        action = self._preprocess_action(action)
        action = common.to_tensor(action)
        pass

    def get_state(self) -> dict:
        """ Returns the targets """
        pass

    def set_state(self, state: dict):
        """ Sets the targets """
        pass

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices].cpu().numpy()
        )
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def set_drive_targets(self, targets):
        """ Set target for each joint """
        pass

    def before_simulation_step(self):
        self._step += 1


@dataclass
class TorqueJointControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = TorqueJointController