import rllib

import copy
import numpy as np
import time
import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal

from rllib.buffer import ReplayBuffer
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper


class SAC(MethodSingleAgent):
    gamma = 0.9

    reward_scale = 50
    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 1e-4
    lr_tune = 1e-4

    tau = 0.005

    buffer_size = 1000000
    batch_size = 128

    start_timesteps = 30000
    # start_timesteps = 100  ### !warning

    save_model_interval = 200

    def __init__(self, config: rllib.basic.YamlConfig, writer, tag_name='method'):
        super().__init__(config, writer, tag_name)

        self.critic = config.get('net_critic', rllib.sac.Critic)(config, model_id=0).to(self.device)
        self.actor = config.get('net_actor', rllib.sac.Actor)(config, model_id=0).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        config_adv = copy.copy(config)
        config_adv.set('dim_action', 1)
        self.critic_adv = config_adv.get('net_critic', rllib.sac.Critic)(config_adv, model_id=1).to(self.device)
        self.actor_adv = config_adv.get('net_actor', rllib.sac.Actor)(config_adv, model_id=1).to(self.device)
        self.critic_adv_target = copy.deepcopy(self.critic_adv)

        self.models_to_save = [self.critic, self.actor]
        self.models_to_save_adv = [self.critic_adv, self.actor_adv]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic_adv_optimizer= Adam(self.critic_adv.parameters(), lr=self.lr_critic)
        self.actor_adv_optimizer = Adam(self.actor_adv.parameters(), lr=self.lr_actor)

        self.critic_loss = nn.MSELoss()

        ### automatic entropy tuning
        if self.target_entropy == None:
            self.target_entropy = -np.prod((self.dim_action,)).item()
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=self.dtype, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_tune)

        self.log_alpha_adv = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=self.dtype, device=self.device)
        self.alpha_adv = self.log_alpha_adv.exp().detach()
        self.alpha_adv_optimizer = Adam([self.log_alpha_adv], lr=self.lr_tune)

        self.buffer: ReplayBuffer = config.get('buffer', ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)

        self.step_update_adv = -1



    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps:
            return
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward *self.reward_scale
        done = experience.done

        '''critic'''
        with torch.no_grad():
            next_action, next_logprob, _ = self.actor.sample(next_state)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logprob
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor.sample(state)
        actor_loss = (-self.critic.q1(state, action) + self.alpha * logprob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        '''automatic entropy tuning'''
        alpha_loss = self.log_alpha.exp() * (-logprob.mean() - self.target_entropy).detach()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        self.writer.add_scalar(f'{self.tag_name}/loss_critic', critic_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/loss_actor', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/alpha', self.alpha.detach().item(), self.step_update)

        self._update_model()
        if self.step_update % self.save_model_interval == 0:
            self._save_model()
        
        self.update_callback(locals())
        return


    def update_parameters_adv(self):
        if len(self.buffer) < self.start_timesteps:
            return
        self.step_update_adv += 1

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action_adv
        next_state = experience.next_state
        reward = -experience.reward *self.reward_scale
        done = experience.done

        '''critic'''
        with torch.no_grad():
            next_action, next_logprob, _ = self.actor_adv.sample(next_state)

            target_q1, target_q2 = self.critic_adv_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha_adv * next_logprob
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic_adv(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_adv_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_adv_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor_adv.sample(state)
        actor_loss = (-self.critic_adv.q1(state, action) + self.alpha_adv * logprob).mean()
        self.actor_adv_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_adv_optimizer.step()

        '''automatic entropy tuning'''
        alpha_loss = self.log_alpha_adv.exp() * (-logprob.mean() - self.target_entropy).detach()
        self.alpha_adv_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_adv_optimizer.step()

        self.alpha_adv = self.log_alpha_adv.exp().detach()

        self.writer.add_scalar(f'{self.tag_name}/loss_critic_adv', critic_loss.detach().item(), self.step_update_adv)
        self.writer.add_scalar(f'{self.tag_name}/loss_actor_adv', actor_loss.detach().item(), self.step_update_adv)
        self.writer.add_scalar(f'{self.tag_name}/alpha_adv', self.alpha.detach().item(), self.step_update_adv)

        self._update_model_adv()
        if self.step_update_adv % self.save_model_interval == 0:
            self._save_model_adv()
        
        return




    def update_parameters_adv_(self, index, n_iters=1000):
        t1 = time.time()

        # for i in range(n_iters):
        #     if i % (n_iters //10) == 0:
        #         print(prefix(self) + 'update_parameters index / total: ', i, n_iters)
        #     self.update_parameters()

        for i in tqdm.tqdm(range(n_iters)):
        # for i in range(n_iters):
            self.update_parameters_adv()
        
        t2 = time.time()
        self.writer.add_scalar(f'{self.tag_name}/update_adv_time', t2-t1, index)
        self.writer.add_scalar(f'{self.tag_name}/update_adv_iters', n_iters, index)
        self.writer.add_scalar(f'{self.tag_name}/update_adv_time_per_iter', (t2-t1) /n_iters, index)
        # print()
        return


    



    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            action, _, _ = self.actor.sample(state.to(self.device))
            action = action.cpu()
        return action

    @torch.no_grad()
    def select_action_adv(self, state):
        if self.step_select < self.start_timesteps:
            action = torch.Tensor(1,1).uniform_(-1,1)
        else:
            action, _, _ = self.actor_adv.sample(state.to(self.device))
            action = action.cpu()
        return action



    def _update_model(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def _update_model_adv(self):
        soft_update(self.critic_adv_target, self.critic_adv, self.tau)


    def _save_model_adv(self, iter_num=None):
        if iter_num == None:
            iter_num = self.step_update_adv
        [model.save_model(self.model_dir, iter_num) for model in self.models_to_save_adv]

