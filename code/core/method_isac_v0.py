import rllib

import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam


class IndependentSAC_v0(rllib.template.MethodSingleAgent):
    dim_reward = 2
    
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
    # start_timesteps = 1000  ## ! warning
    before_training_steps = 0

    save_model_interval = 200


    def __init__(self, config: rllib.basic.YamlConfig, writer, tag_name='method'):
        super().__init__(config, writer, tag_name)

        self.critic = config.get('net_critic', rllib.sac.Critic)(config).to(self.device)
        self.actor = config.get('net_actor', rllib.sac.Actor)(config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()

        ### automatic entropy tuning
        if self.target_entropy == None:
            self.target_entropy = -np.prod((self.dim_action,)).item()
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=self.dtype, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_tune)

        self.buffer: rllib.buffer.ReplayBuffer = config.get('buffer', rllib.buffer.ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)

    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps + self.before_training_steps:
            return
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
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

        return


    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(len(state), self.dim_action).uniform_(-1,1)
        else:
            # print('select: ', self.step_select)
            states = rllib.buffer.stack_data(state)
            self.buffer.pad_state(states)
            states = states.cat(dim=0)
            action, _, _ = self.actor.sample(states.to(self.device))
            action = action.cpu()
        return action
    
    def _update_model(self):
        # print('[update_parameters] soft update')
        rllib.utils.soft_update(self.critic_target, self.critic, self.tau)


