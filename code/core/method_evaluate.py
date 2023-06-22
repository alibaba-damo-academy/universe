import rllib
import numpy as np
import matplotlib
import torch
import copy



class Evaluate(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name
        if method_name == 'TD3':
            from rllib import td3
            self.critic = config.get('net_critic', td3.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', td3.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_td3
        elif method_name == 'SAC':
            from rllib import sac
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_sac

        elif method_name == 'IndependentSAC_v0'.upper():
            from rllib import sac
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            # self.models_to_load = [self.critic, self.actor]
            self.models_to_load = [self.actor]
            self.select_actions = self.select_action_isac
            self.select_action = self.select_action_sac

        else:
            raise NotImplementedError('No such method: ' + str(method_name))
        return



    @torch.no_grad()
    def select_action_td3(self, state):
        self.select_action_start()
        state = state.to(self.device)
        action = self.actor(state)


        # print('action: ', action)

        return action


    @torch.no_grad()
    def select_action_sac(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action



    @torch.no_grad()
    def select_action_isac(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .models import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return actions


    def store(self, experience, **kwargs):
        return








class EvaluateIndependentSAC(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
        self.models_to_load = [self.critic, self.actor]

        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()



    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .models import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return actions.cpu()


    def store(self, experience, **kwargs):
        return





class EvaluateIndependentSACMean(EvaluateIndependentSAC):

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return mean.cpu()



    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .models import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return mean.cpu()












class EvaluateSAC(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
        self.models_to_load = [self.critic, self.actor]

        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()



    def store(self, experience, **kwargs):
        return







class EvaluateSACAdv(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config, model_id=0).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config, model_id=0).to(self.device)

        config_adv = copy.copy(config)
        config_adv.set('dim_action', 1)
        self.critic_adv = config.get('net_critic', sac.Critic)(config_adv, model_id=1).to(self.device)
        self.actor_adv = config.get('net_actor', sac.Actor)(config_adv, model_id=1).to(self.device)

        self.models_to_load = [self.critic, self.actor, self.critic_adv, self.actor_adv]

        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()


    @torch.no_grad()
    def select_action_adv(self, state):
        state = state.to(self.device)

        action, logprob, mean = self.actor_adv.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()




    def store(self, experience, **kwargs):
        return





class EvaluateSACAdvDecouple(EvaluateSACAdv):

    def select_method(self):
        config, method_name = self.config, self.method_name
        config_adv = config.config_adv

        config_adv.set('evaluate', config.evaluate)
        config_adv.set('device', config.device)
        config_adv.set('method_name', config.method_name)
        config_adv.set('net_actor_fe', config.net_actor_fe)
        config_adv.set('net_critic_fe', config.net_critic_fe)
        config_adv.set('dim_state', config.dim_state)
        config_adv.set('dim_action', 1)


        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config, model_id=0).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config, model_id=0).to(self.device)

        self.critic_adv = config.get('net_critic', sac.Critic)(config_adv, model_id=1).to(self.device)
        self.actor_adv = config.get('net_actor', sac.Actor)(config_adv, model_id=1).to(self.device)

        self.models_to_load = [self.critic, self.actor, self.critic_adv, self.actor_adv]

        return


