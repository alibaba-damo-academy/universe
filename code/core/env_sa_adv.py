import rllib
import universe

import copy
import torch


class EnvSingleAgentAdv(universe.EnvInteractiveSingleAgent):
    def step(self, action, action_adv):
        self.time_step += 1

        adv_character = (action_adv.item() + 1) *0.5
        print('adv_character: ', adv_character)

        episode_info = self.check_episode()
        reward = self.reward_func.run_step(self.state, action, self.agents_master, episode_info)
        done = episode_info.done

        ### metric step
        self.metric.on_episode_step(self.time_step, reward, episode_info)
        self.recorder.record_vehicles(self.time_step, self.agents_master, episode_info)

        ### step
        self.world_tick(action)
        self.agents_master.vehicles_neural[0].character = adv_character

        ### reset done agents, only for multi-agent
        self.reset_done_agents(episode_info)

        ### next_state
        next_state = self.agents_master.observe(self.step_reset, self.time_step)
        state = self.state
        self.state = copy.copy(next_state)

        ### pad next_state, only for multi-agent
        next_state_vis = [s.vi for s in next_state]
        for s in state:
            if s.vi not in next_state_vis:
                next_state.append(s)
        next_state = sorted(next_state, key=lambda x: x.vi)

        ### experience
        experience = []
        for s, a, ad, ns, r, d in zip(state, action, action_adv, next_state, reward, episode_info.finish):
            e = rllib.basic.Data(vi=s.vi,
                state=s.to_tensor().unsqueeze(0),
                action=torch.from_numpy(a).unsqueeze(0),
                action_adv=torch.from_numpy(ad).unsqueeze(0),
                next_state=ns.to_tensor().unsqueeze(0),
                reward=torch.tensor([r], dtype=torch.float32),
                done=torch.tensor([d], dtype=torch.float32),
            )
            experience.append(e)

        ### metric end
        if done:
            self.metric.on_episode_end()
            self.recorder.save()
        return experience[self.slice()], done, episode_info.to_dict()


