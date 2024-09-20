from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv, _build_box_with_hole
from mani_skill.utils.registration import register_env
import torch
import sapien
import numpy as np
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils import common, sapien_utils
from typing import Any, Dict, Union
from mani_skill.utils.structs.actor import Actor

@register_env("FragilePegInsert-v1", max_episode_steps=150)
class FragilePegInsert(PegInsertionSideEnv):
    maximum_peg_force = 35.0
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    def __init__(self, *args, obs_mode='state', **kwargs):
        # handle the ft stuff
        self.return_force_data = True
        if 'no_ft' in obs_mode:
            self.return_force_data = False
            obs_mode=obs_mode[:-6]

        super().__init__(*args, obs_mode=obs_mode, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.obsticles = [
            self.agent.finger1_link,
            self.agent.finger2_link,
            self.box
        ]  

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        #print(
            #torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.peg, self.box), axis=1)
        #    self.scene.get_pairwise_contact_forces(self.peg, self.box)
            #print(self.box),
            #print(self.peg)
        #)
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)
        #out_dic = super().evaluate()
        #out_dic['fail'], out_dic['max_force'], out_dic['fail_cause'] = self.pegBroke()
        #return out_dic
    
    def _get_obs_extra(self, info: Dict):
        """ 
            Adds force-torque if no_ft not in obs_mode
            handles state calculations
        """
        data = super()._get_obs_extra(info)

        if self.return_force_data:
            data['force'] = self.agent.tcp.get_net_contact_forces()
        return data
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
        #print(f"Net Peg forces: {torch.linalg.norm(self.peg.get_net_contact_forces())}")
        #print(f"Net Box: {self.box.get_net_contact_forces()}")
        for i, obs in enumerate(self.obsticles):
            obs_forces = self.getPegForce(obs)
            update = obs_forces > max_forces
            idx = 1 # robot
            if i == 2:
                idx = 2 # box
                #print("\tForce on box:", obs_forces)
            resp_actor += idx * update - update * resp_actor   
            max_forces = torch.maximum( max_forces, obs_forces)

        brokeFlag = self.maximum_peg_force <= max_forces
        return brokeFlag, max_forces, resp_actor

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        #print("Computing dense reward")
        r = super().compute_dense_reward(obs, action, info)
        r *= torch.logical_not(info['fail']) # zero out all failed cases
        r -= 10 * info['fail'] # make them all -10!
        #print("dense reward:", r[0])
        return r
        