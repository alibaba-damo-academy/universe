
import numpy as np
from typing import List

from .common.actor import check_collision
from .dataset import Dataset
from .env_sa_v0 import AgentMode, SingleAgentBoxSpace
from .env_sa_v0 import EnvInteractiveSingleAgent, EnvReplaySingleAgent




class MultiAgentBoxSpace(object):
    def __init__(self, spaces: List[SingleAgentBoxSpace]):
        self.spaces = spaces

    def sample(self):
        return [space.sample() for space in self.spaces]





class EnvInteractiveMultiAgent(EnvInteractiveSingleAgent):
    sa = False

    @property
    def action_space(self):
        return MultiAgentBoxSpace([SingleAgentBoxSpace(-1.0, 1.0, shape=(self.dim_action,), dtype=np.float32) for _ in range(len(self.agents_master.vehicles_neural))])



    def reset_done_agents(self, episode_info):
        finish_agents = []
        for agent, finish in zip(self.agents_master.vehicles_neural, episode_info.finish):
            if finish:
                finish_agents.append(agent)
        for agent in finish_agents:
            self.agents_master.remove(agent)
        return




class EnvReplayMultiAgent(EnvReplaySingleAgent):
    sa = False

    @property
    def action_space(self):
        return MultiAgentBoxSpace([SingleAgentBoxSpace(-1.0, 1.0, shape=(self.dim_action,), dtype=np.float32) for _ in range(len(self.agents_master.vehicles_neural))])



    def reset_done_agents(self, episode_info):
        finish_agents = []
        for agent, finish in zip(self.agents_master.vehicles_neural, episode_info.finish):
            if finish:
                finish_agents.append(agent)
        for agent in finish_agents:
            self.agents_master.remove(agent)

        for vehicle in self.data.vehicles:
            if not self.agents_master.has(vehicle):
                self.agents_master.register_vehicle(vehicle)
        return

