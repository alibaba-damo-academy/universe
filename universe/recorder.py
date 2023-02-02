import rldev

import numpy as np
import pickle
import os
import copy

from .scenario import Scenario
from .agents_master import AgentsMaster



class Recorder(object):
    def __init__(self, config: rldev.YamlConfig, env_name, sa, dir_path):
        self.env_name = env_name
        self.sa = sa
        self.dir_path = dir_path
        self.records = {}

        config.save(self.dir_path)
        return


    def record_scenario(self, step_reset, scenario: Scenario, agents_master: AgentsMaster):
        self.step_reset = step_reset
        self.records['step_reset'] = step_reset
        self.records['scenario'] = copy.deepcopy(scenario)
        self.records['agents_master'] = copy.deepcopy(agents_master)


    def record_vehicles(self, time_step, agents_master: AgentsMaster, episode_info: rldev.Data):
        """
        
        Args:
            timestamp: time.time()
            agents: list of BaseAgent and BaseAgentObstacle
        
        Returns:
            
        """

        self.records[time_step] = {}
        for vehicle in agents_master.vehicles_neural + agents_master.vehicles_rule:
            self.records[time_step][vehicle.vi] = copy.deepcopy(vehicle)

        self.records[time_step]['episode_info'] = copy.deepcopy(episode_info)
        return


    def save(self):
        file_path = os.path.join(self.dir_path, f'{self.step_reset}.txt')
        with open(file_path, 'wb') as f:
            pickle.dump(self.records, f)
        
        del self.records
        self.records = {}
        return


    @staticmethod
    def load(file_path):
        record = None
        with open(file_path, 'rb') as f:
            record = pickle.load(f)
        return record





class PseudoRecorder(object):
    def __init__(self, *args, **kwargs):
        return

    def record_scenario(self, *args, **kwargs):
        return
    def record_vehicles(self, *args, **kwargs):
        return
    def record_experience(self, *args, **kwargs):
        return
    def save(self, *args, **kwargs):
        return
    def clear(self, *args, **kwargs):
        return

