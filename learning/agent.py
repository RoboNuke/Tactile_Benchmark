
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Dict

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

import numpy as np
class FlattenRGBDFTObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with three keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
        force (bool): Whether to include force data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=True, state=True, force=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_force = force
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        if 'sensor_param' in observation.keys():
            del observation["sensor_param"]

        if 'sensor_data' in observation:
            sensor_data = observation.pop("sensor_data")
            images = []
            for cam_data in sensor_data.values():
                if self.include_rgb:
                    images.append(cam_data["rgb"])
                if self.include_depth:
                    images.append(cam_data["depth"])
            if self.include_depth or self.include_rgb:
                images = torch.concat(images, axis=-1)

        # flatten the rest of the data which should just be state data
        ret = dict()
        if 'force' in observation['extra']:
            force = observation['extra'].pop('force')
        observation = common.flatten_state_dict(observation, use_torch=True)

        if self.include_state:
            ret["state"] = observation
        if self.include_force:
            ret['force'] = force
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images
        return ret
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, sample_obs, force_type="FFN"):#, nn.ReLU=nn.ReLU):
        super().__init__()

        extractors = {}
        self.out_features = 0
        feature_size = 256
        if 'rgb' in sample_obs:
            in_channels=sample_obs["rgb"].shape[-1]
            image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.ReLU()
            )
            self.out_features += 256

        if "force" in sample_obs:
            if force_type == "FFN":
                force_size = sample_obs['force'].shape[-1]
                extractors["force"] = nn.Sequential(
                    nn.Linear(force_size, 256),
                    nn.ReLU()
                )
                self.out_features += 256
                
            elif force_type == "1D-CNN":
                raise NotImplementedError(f"1D-CNN not implemented yet")
            else:
                raise NotImplementedError(f"Unexpected force type:{force_type}")
            
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            if key == 'force':
                obs = torch.tanh( obs.float() * 0.0011 )
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class CustomGLU(nn.Module):
    def __init__(self, input_size):
        super(CustomGLU, self).__init__()
        self.linear_gate = nn.Linear(input_size, input_size)
        self.linear_input = nn.Linear(input_size, input_size)

    def forward(self, x):
        #print(x.size())
        #print(self.linear_gate, self.linear_input)
        gate = torch.sigmoid(self.linear_gate(x))
        return self.linear_input(x) * gate 

class BroNetLayer(nn.Module):
    def __init__(self, size, device):
        super().__init__()

        self.path = nn.Sequential(
            layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
            #nn.GLU(),
            #CustomGLU(size),
            nn.ReLU(),
            layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.path(res)
        return out + res

class BroNet(nn.Module):
    def __init__(self, n, in_size, out_size, latent_size, device, tanh_out=False):
        super().__init__()
        self.layers = []
        self.n = n
        self.input = nn.Sequential(
            layer_init(nn.Linear(in_size, latent_size)),
            nn.LayerNorm(latent_size),
            #nn.GLU(),
            #CustomGLU(size),
            nn.ReLU(),
        ).to(device)

        self.layers = nn.ModuleList([BroNetLayer(latent_size, device) for i in range(self.n)])

        if not tanh_out:
            self.output = layer_init(nn.Linear(latent_size, out_size)).to(device)
        else:
            self.output = nn.Sequential(
                layer_init(nn.Linear(latent_size, out_size)).to(device),
                nn.Tanh()
            ).to(device)

    def forward(self, x):
        #print("input forward:", self.input)
        out =  self.input(x)
        #print("net forward out:", out.size())
        for i in range(self.n):
            out = self.layers[i](out)
            #print(f"{i} forward:", out.size())
        #print("about to out")
        out = self.output(out)
        #print(out.size())
        return out


class Agent(nn.Module):
    def __init__(self, envs, sample_obs, force_type='FFN', passover=False):
        super().__init__()
        if passover:
            return
        self.feature_net = NatureCNN(sample_obs=sample_obs, force_type=force_type)
        
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        #print("in x:", [x[k].size() for k in x])
        x = self.feature_net(x)
        #print(x.size())
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class BroAgent(Agent):
    def __init__(self, 
            envs, 
            sample_obs, 
            force_type='FNN', 
            critic_n = 1, 
            actor_n = 2,
            critic_latent=128,
            actor_latent=512,
            tot_actors=1,
            device='cpu'
        ):

        super().__init__(envs, sample_obs, force_type, True)
        self.feature_net = NatureCNN(sample_obs=sample_obs, force_type=force_type)#, act_func=nn.ReLU)
        self.feature_net.to(device)
        #print("feat net\n", self.feature_net)
        in_size = self.feature_net.out_features
        out_size = np.prod(envs.unwrapped.single_action_space.shape)
        
        self.critic = BroNet(critic_n, in_size, 1, critic_latent, device, tanh_out=False).to(device)
        
        self.actors = [BroNet(actor_n, in_size, out_size, actor_latent, device, tanh_out=True) for i in range(tot_actors)]
        
        for actor in self.actors:
            layer_init(actor.output[-2], std=0.01*np.sqrt(2)) 
            actor.to(device)
        self.actor_logstds = [nn.Parameter(torch.ones(1, out_size) * -0.5).to(device) for i in range(tot_actors)]
        for logstd in self.actor_logstds:
            logstd.to(device)
        if tot_actors == 1:
            self.actor_mean = self.actors[0]
            self.actor_logstd = self.actor_logstds[0]
        
