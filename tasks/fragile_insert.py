from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv, _build_box_with_hole
from mani_skill.utils.registration import register_env
import torch
import sapien
import numpy as np
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils import common, sapien_utils

from mani_skill.utils.structs.actor import Actor

@register_env("FragilePegInsert-v1", max_episode_steps=150)
class FragilePegInsert(PegInsertionSideEnv):
    maximum_peg_force = 35.0
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.obsticles = [
            self.agent.finger1_link,
            self.agent.finger2_link,
            self.box
        ]  

    def evaluate(self):
        out_dic = super().evaluate()
        out_dic['fail'], out_dic['max_force'] = self.pegBroke()
        out_dic['fail_cause'] = 2 * out_dic['fail']

        return out_dic
    
    def getPegForce(self, object: Actor):

        contact_force = self.scene.get_pairwise_contact_forces(
            self.peg, object
        )
        force = torch.linalg.norm(contact_force, axis=1)
        return force

    def pegBroke(self):
        """ Calculates the maximum force on the peg and returns it """
        max_forces = torch.zeros((self.num_envs), device=self.device)
        all_forces = torch.zeros((len(self.obsticles)))
        #print(self.scene.sub_scenes[0].get_contacts())
        for i, obs in enumerate(self.obsticles):
            obs_forces = self.getPegForce(obs)
            all_forces[i] = obs_forces[0]
            max_forces = torch.maximum( max_forces, obs_forces)
        #print(all_forces)
        brokeFlag = self.maximum_peg_force <= max_forces
        return brokeFlag, max_forces

