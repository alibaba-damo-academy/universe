
from .agents_master import AgentsMaster


class RewardFunc(object):
    def __init__(self, *args, **kwargs):
        return


    def run_step(self, state, action, agents_master: AgentsMaster, episode_info):
        return [0.0] *len(agents_master.vehicles_neural)


