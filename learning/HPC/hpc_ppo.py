import sys

# Add the folder path
folder_path = "/nfs/stak/users/brownhun/hpc-share/Tactile_Benchmark"
#folder_path = "/home/hunter/Tactile_Benchmark"
sys.path.append(folder_path)

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_optimizer import Shampoo
import tyro
from torch.distributions.normal import Normal

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# hunter stuff
from learning.agent import *
from data.data_manager import DataManager
from tasks.wiping import *
from tasks.barnicle_scrap import *
from tasks.fragile_insert import *
from tasks.simple_fPiH import *
from tasks.for_env import *
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.closed_gripper import *

@dataclass
class Args:
    exp_name: Optional[str] = "Test_Run"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb_project_name: str = "Tester"
    """the wandb's project name"""
    wandb_entity: str = "hur"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # exp specific info
    obs_mode: str = 'state_dict_no_ft'
    """the observation mode for the robot"""
    control_mode: str = 'pd_joint_delta_pos'
    """the action space or control mode"""
    include_force: bool = False
    """if the robot recieves information about force observations"""
    force_encoding: str = 'FFN'
    """How to encode the force information"""
    reward_mode: str = 'normalized_dense'
    """Reward mode to use during training"""
    robot: str = "panda"
    """Which robot to use """
    lock_gripper: bool = True
    """To make the gripper always closed"""
    exp_max_dmg_force: float = 500.0
    """ Force to break the peg or the table """
    norm_force: list = field(
        default_factory=lambda :[500.0, 1000])
    """ Force to remove an attached object """
    use_bro_agent: bool = True
    """ Use BroNet architecture for critic/agent or old one"""
    critic_n: int = 2
    """ How many layers of bronet for the bro agent """
    critic_hidden_layer_size: int = 512
    """ How many hidden units in each layer of the bro agent critic BroNet """
    use_shampoo: bool = False
    """ Use shampoo optimizer instead of ADAM"""

    # Algorithm specific arguments
    env_id: str = "ForeignObjectRemoval-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    weight_decay: float = 0.0#0.01
    """the weight decay precent for optimizer"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    num_steps: int = 150
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 150
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 #0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.99 #0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0 #0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.05 #0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = True

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

def log_smothness(data: Dict, logger: DataManager, step, fold='train'):
    for k,v in data.items():
        raw = v.float().cpu()
        logger.add_histogram(f'{fold}/{k}_hist', raw, step)
        logger.add_scalar({f'{fold}/{k}': raw.mean()}, step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #print("Using device:", device)
    # env setup
    env_kwargs = dict(
        robot_uids=args.robot,
        obs_mode=args.obs_mode, 
        control_mode=args.control_mode, 
        render_mode=args.render_mode, 
        sim_backend="gpu",
        dmg_force=args.exp_max_dmg_force
    )
    if args.env_id == "ForeignObjectRemoval-v1":
        env_kwargs['obj_norm_force'] = args.norm_force

    eval_envs = gym.make(
        args.env_id, 
        num_envs=args.num_eval_envs, 
        **env_kwargs
    )
    
    envs = gym.make(
        args.env_id, 
        num_envs=args.num_envs if not args.evaluate else 1, 
        **env_kwargs
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
    logger = None
    
    if not args.evaluate:
        print("Running training")
        config = vars(args)
        config["env_cfg"] = dict(
            **env_kwargs, 
            num_envs=args.num_envs, 
            env_id=args.env_id, 
            reward_mode=args.reward_mode, 
            env_horizon=max_episode_steps, 
            partial_reset=args.partial_reset
        )
        config["eval_env_cfg"] = dict(
            **env_kwargs, 
            num_envs=args.num_eval_envs, 
            env_id=args.env_id, 
            reward_mode=args.reward_mode, 
            env_horizon=max_episode_steps, 
            partial_reset=args.partial_reset
        )
        tags = [
            args.env_id, # which problem we solving
            args.reward_mode, 
            "state" if args.include_state else "without_state", # did we include state information 
            args.obs_mode, 
            "force" if args.include_force else "without_force",
            args.control_mode, # action space 
            str(args.exp_max_dmg_force),
            'removed_clip'
        ]

        logger = DataManager(
            project=args.wandb_project_name,
            entity=args.wandb_entity)
        
        logger.init_new_run(
            run_name, 
            config,
            tags = tags
        )
        if args.save_model:
            os.makedirs(logger.get_dir() + "/ckpts")
        assert logger is not None, "Logger didn't init"
    else:
        print("Running evaluation")

    envs = SmoothnessObservationWrapper(envs)
    eval_envs = SmoothnessObservationWrapper(eval_envs)
    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDFTObservationWrapper(
        envs, 
        rgb= 'rgb' in args.obs_mode, 
        depth=False, 
        state=args.include_state,
        force=args.include_force
    )
    eval_envs = FlattenRGBDFTObservationWrapper(
        eval_envs,  
        rgb= 'rgb' in args.obs_mode, 
        depth=False, 
        state=args.include_state,
        force=args.include_force
    )

    if args.lock_gripper:
        envs = CloseGripperActionSpaceWrapper(envs)
        eval_envs = CloseGripperActionSpaceWrapper(eval_envs)

    if args.capture_video:
        
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        else:
            eval_output_dir = f"{logger.get_dir()}/videos"
        #print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs, 
                output_dir=f"{logger.get_dir()}/train_videos", 
                save_trajectory=False, 
                save_video_trigger=save_video_trigger, 
                max_steps_per_video=args.num_steps, 
                video_fps=30
            )
        eval_envs = RecordEpisode(
            eval_envs, 
            output_dir=eval_output_dir, 
            save_trajectory=args.evaluate, 
            trajectory_name="trajectory", 
            max_steps_per_video=args.num_eval_steps, 
            video_fps=30
        )
    
    
    envs = ManiSkillVectorEnv(
        envs, 
        args.num_envs, 
        ignore_terminations=False, 
        record_metrics=True
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, 
        args.num_eval_envs, 
        ignore_terminations=False, 
        record_metrics=True
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    
    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    print(next_obs['state'].size())
    eval_obs, _ = eval_envs.reset()#seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    eps_lens = np.zeros(args.num_envs)
    place_rew = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")

    # add some stuff for logging
    eval_returns = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device)
    eval_reward = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device) # avg reward
    eval_episode_lens = torch.zeros(args.num_eval_envs, dtype=torch.int32, device=device)
    eval_succs = torch.zeros(args.num_eval_envs, dtype=torch.bool, device=device)
    eval_fails = torch.zeros(args.num_eval_envs, dtype=torch.bool, device=device)
    eval_dones = torch.zeros(args.num_eval_envs, dtype=torch.bool, device=device)

    eval_ssv = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device)
    eval_ssj = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device)
    eval_max_dmg_force = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device)
    eval_metrics = {
        'eval/returns':eval_returns,
        'eval/reward':eval_reward,
        'eval/episode_length':eval_episode_lens,
        'eval/success_rate':eval_succs,
        'eval/failure_rate':eval_fails,
        'eval_smoothness/sum_squared_velocity': eval_ssv,
        'eval_smoothness/sum_squared_jerk': eval_ssj,
        'eval_smoothness/avg_max_force': eval_max_dmg_force,
    }
    # add some stuff for logging
    train_returns = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    train_reward = torch.zeros(args.num_envs, dtype=torch.float32, device=device) # avg reward
    train_episode_lens = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
    train_succs = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    train_fails = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    train_dones = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    train_ssv = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    train_ssj = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    train_max_dmg_force = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    
    train_metrics = {
        'train/returns':train_returns,
        'train/reward':train_reward,
        'train/episode_length':train_episode_lens,
        'train/success_rate':train_succs,
        'train/failure_rate':train_fails,
        'train_smoothness/sum_squared_velocity': train_ssv,
        'train_smoothness/sum_squared_jerk': train_ssj,
        'train_smoothness/avg_max_force': train_max_dmg_force,
    }

    if args.use_bro_agent:
        agent = BroAgent(
            envs, 
            sample_obs=next_obs,
            force_type=args.force_encoding,
            device = device,
            critic_n = args.critic_n,
            critic_latent = args.critic_hidden_layer_size
        ).to(device)
    else:
        agent = Agent(
            envs, 
            sample_obs=next_obs, 
            force_type=args.force_encoding
        ).to(device)

    if not args.use_shampoo:
        optimizer = optim.AdamW(
            agent.parameters(), 
            lr=args.learning_rate, 
            eps=1e-5,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = Shampoo(
                agent.parameters(), 
                lr=args.learning_rate, 
                eps=1e-5, 
                weight_decay=args.weight_decay
        )
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    eval_count = 0
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1 or iteration==args.num_iterations:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            #eval_metrics = defaultdict(list)
            #smooth_metrics = defaultdict(list)
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    #eval_envs._env.render_human()
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        sm = eval_infos['smoothness']
                        eval_ssv[mask] = sm['sum_sqr_qv'][mask]
                        eval_ssj[mask] = sm['sum_sqr_qjerk'][mask]
                        eval_max_dmg_force[mask] = sm['max_dmg_force'][mask]
                        #print("episode:", eval_infos['final_info']['episode'].keys())
                        eval_succs[mask] = eval_infos['success'][mask]
                        eval_fails[mask] = eval_infos['fail'][mask]
                        eval_episode_lens[mask] = eval_infos['elapsed_steps'][mask]
                        eval_dones[mask] = True
                        fin_eps = eval_infos['final_info']['episode']
                        if 'success_once' in fin_eps:
                            # new system
                            eval_returns[mask] = fin_eps['return'][mask]
                            eval_reward[mask] = fin_eps['reward'][mask]
                        else:
                            # old system
                            eval_returns[mask] = fin_eps['r'][mask]
                            eval_reward[mask] = eval_returns[mask] / eval_episode_lens[mask]

            mask = ~eval_dones
            sm = eval_infos['smoothness']
            eval_ssv[mask] = sm['sum_sqr_qv'][mask]
            eval_ssj[mask] = sm['sum_sqr_qjerk'][mask]
            eval_max_dmg_force[mask] = sm['max_dmg_force'][mask]
            eval_succs[mask] = eval_infos['success'][mask]
            eval_fails[mask] = eval_infos['fail'][mask]
            eval_episode_lens[mask] = eval_infos['elapsed_steps'][mask]
            
            fin_eps = eval_infos['episode']
            if 'success_once' in fin_eps:
                # new system
                eval_returns[mask] = fin_eps['return'][mask]
                eval_reward[mask] = fin_eps['reward'][mask]
            else:
                # old system
                eval_returns[mask] = fin_eps['r'][mask]
                eval_reward[mask] = eval_returns[mask] / eval_episode_lens[mask]

            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {len(eps_lens)} episodes")
            for k, v in eval_metrics.items():
                #print(k,v)
                mean = v.cpu().float().mean()
                print(f"\teval_{k}_mean={mean}")
                logger.add_scalar({k:mean}, step=global_step+1)
            
            if args.capture_video:
                #print(f"Saving Video at {global_step}")
                if args.save_train_video_freq is not None:
                    logger.add_save(f"train_videos/{eval_count}.mp4")
                logger.add_save(f"/videos/{eval_count}.mp4")
                logger.add_mp4(
                    f'{logger.get_dir()}/videos/{eval_count}.mp4', 
                    tag = 'eval',
                    step= global_step+1, 
                    cap=f'Evaluation after {global_step} steps', 
                    fps=10
                )
            eval_count += 1

            if args.evaluate:
                break

        if args.save_model and iteration % args.eval_freq == 1:
            #model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            model_path = f'{logger.get_dir()}/ckpts/ckpt_{iteration}.pt'
            torch.save(agent.state_dict(), model_path)
            logger.add_save(f"ckpts/ckpt_{iteration}.pt")
            #logger.add_ckpt(model_path, f'ckpt_{iteration}.pt')
            #print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.time()
        next_obs, _ = envs.reset()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                sm = infos['smoothness']
                train_ssv[done_mask] = sm['sum_sqr_qv'][done_mask]
                train_ssj[done_mask] = sm['sum_sqr_qjerk'][done_mask]
                train_max_dmg_force[done_mask] = sm['max_dmg_force'][done_mask]
                
                #print("episode:", train_infos['final_info']['episode'].keys())
                train_succs[done_mask] = infos['success'][done_mask]
                train_fails[done_mask] = infos['fail'][done_mask]
                train_episode_lens[done_mask] = infos['elapsed_steps'][done_mask]
                train_dones[done_mask] = True
                fin_eps = final_info['episode']
                if 'success_once' in fin_eps:
                    # new system
                    train_returns[done_mask] = fin_eps['return'][done_mask]
                    train_reward[done_mask] = fin_eps['reward'][done_mask]
                else:
                    # old system
                    train_returns[done_mask] = fin_eps['r'][done_mask]
                    train_reward[done_mask] = train_returns[done_mask] / train_episode_lens[done_mask]

                #log_smothness(
                #    final_info['smoothness'],
                #    logger,
                #    global_step, 
                #    'train_smoothness'
                #)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        
        done_mask = ~train_dones
        sm = infos['smoothness']
        train_ssv[done_mask] = sm['sum_sqr_qv'][done_mask]
        train_ssj[done_mask] = sm['sum_sqr_qjerk'][done_mask]
        train_max_dmg_force[done_mask] = sm['max_dmg_force'][done_mask]
        
        #print("episode:", train_infos['final_info']['episode'].keys())
        train_succs[done_mask] = infos['success'][done_mask]
        train_fails[done_mask] = infos['fail'][done_mask]
        train_episode_lens[done_mask] = infos['elapsed_steps'][done_mask]
        fin_eps = infos['episode']
        if 'success_once' in fin_eps:
            # new system
            train_returns[done_mask] = fin_eps['return'][done_mask]
            train_reward[done_mask] = fin_eps['reward'][done_mask]
        else:
            # old system
            train_returns[done_mask] = fin_eps['r'][done_mask]
            train_reward[done_mask] = train_returns[done_mask] / train_episode_lens[done_mask]

        # reset the dones
        train_dones[train_dones] = False

        # log the stuff
        for k, v in train_metrics.items():
            logger.add_scalar(
                {k:v.cpu().float().mean()}, 
                step=global_step
            )

        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.time() - update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar(
            {
                "charts/learning_rate":optimizer.param_groups[0]["lr"],
                "losses/value_loss":v_loss.item(),
                "losses/policy_loss":pg_loss.item(),
                "losses/entropy":entropy_loss.item(),
                "losses/old_approx_kl":old_approx_kl.item(),
                "losses/approx_kl":approx_kl.item(),
                "losses/clipfrac":np.mean(clipfracs),
                "losses/explained_variance":explained_var,
                "charts/SPS":int(global_step / (time.time() - start_time)),
                "time/step":global_step,
                "time/update_time":update_time,
                "time/rollout_time":rollout_time,
                "time/rollout_fps":args.num_envs * args.num_steps / rollout_time
            }, step=global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
    
    if args.save_model and not args.evaluate:
        #model_path = f"runs/{run_name}/final_ckpt.pt"
        model_path = f'{logger.get_dir()}/ckpts/final_ckpt.pt'
        torch.save(agent.state_dict(), model_path)
        logger.add_save("ckpts/final_ckpt.pt")
        #print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: 
        logger.finish()
