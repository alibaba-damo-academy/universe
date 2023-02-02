
from .scenario import Scenario
from .agents_master import AgentsMaster


class Dataset(object):
    def __init__(self, config, env_index, dataset_dir):
        self.scenario = Scenario(config, None)
        self.agents_master = AgentsMaster(config)
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

