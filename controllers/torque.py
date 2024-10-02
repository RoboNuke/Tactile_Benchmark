from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import BaseController, ControllerConfig



class TorqueJointController(BaseController):
    config: "TorqueJointControllerConfig"
    _start_qf = None
    _target_qf = None

    def _initialize_action_space(self):
        """ 
            Creates an action space for Gym env obj
            must be a spaces.box
        """
        pass

    def set_drive_property(self):
        """
            Set drive properties
            - stiffness
            - damping
            - force_limits
            - friction
            Then sets each joints properties
        """
        pass

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