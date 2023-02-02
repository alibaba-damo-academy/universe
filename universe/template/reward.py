
import numpy as np

from .. import AgentsMaster
from .. import RewardFunc




class RewardFunctionNoCharacter(RewardFunc):
    def run_step(self, state, action, agents_master: AgentsMaster, episode_info):
        REWARD_C = -500
        REWARD_B = -500
        collision = episode_info.collision
        off_road = episode_info.off_road
        off_route = episode_info.off_route
        wrong_lane = episode_info.wrong_lane

        reward = []
        for i, agent in enumerate(agents_master.vehicles_neural):
            max_velocity = agent.max_velocity

            ### 1. collision
            reward_collision = int(collision[i]) * REWARD_C /100

            ### 2. boundary
            reward_boundary = int(off_road[i] | off_route[i] | wrong_lane[i]) * REWARD_B /100

            ### 3. velocity
            reward_v = (agent.get_state().v - max_velocity / 2) / (max_velocity / 2)
            reward_v = np.clip(reward_v, -1, 1) *7 /100

            reward.append(reward_collision + reward_v + reward_boundary)
        return reward







class RewardFunctionWithCharacter(RewardFunc):
    def run_step(self, state, action, agents_master: AgentsMaster, episode_info):
        reward = RewardFunctionNoCharacter.run_step(self, state, action, agents_master, episode_info)
        reward = np.array(reward, dtype=np.float32)

        masks = np.zeros_like(state[0].vehicle_masks)
        vis = [agent.vi for agent in agents_master.vehicles_neural]
        masks[vis] = 1

        indexs = np.where(masks > 0)
        reward_padded = np.zeros((masks.shape[0],), dtype=reward.dtype)
        reward_padded[indexs] = reward   ### ! warning: corner case: indexs[0].shape != reward.shape

        reward_with_character = []
        for (agent, s, r) in zip(agents_master.vehicles_neural, state, reward):
            character = agent.character
            agent_masks = s.agent_masks *masks
            num_neighbours = agent_masks.sum() -1  ### ! warning: corner case: num_neighbours < surronding vehicles

            r_others = (reward_padded * agent_masks).sum(axis=0) - r
            # import pdb; pdb.set_trace()
            if num_neighbours > 0:
                r_others /= num_neighbours

            reward_with_character.append( np.cos(character*np.pi/2)* r + np.sin(character*np.pi/2)* r_others )

        return reward_with_character







