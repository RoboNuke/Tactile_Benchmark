from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env
import torch

@register_env("FragilePegInsert-v1", max_episode_steps=50)
class FragilePegInsert(PegInsertionSideEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)   

    def evaluate(self):
        out_dic = super().evaluate()
        return out_dic
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

