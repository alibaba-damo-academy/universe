import rldev

import os
import numpy as np
import pickle
import time

from .. import Scenario
from .. import AgentsMaster
from .. import Recorder



class DatasetInteractive(object):
    def __init__(self, config: rldev.YamlConfig, env_index, dataset_dir=None):
        self.config = config
        self.env_index = env_index

        map_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'maps')

        t1 = time.time()
        centerline = np.load(os.path.join(map_dir, config.map_name, 'topo.npy')).astype(np.float32)
        sideline = np.concatenate([
            np.load(os.path.join(map_dir, config.map_name, 'solid.npy')),
            np.load(os.path.join(map_dir, config.map_name, 'ssolid.npy')),
        ], axis=0).astype(np.float32)
        t2 = time.time()
        print(rldev.prefix(self) + 'load map time: ', t2-t1)

        
        scenario_cls = config.get('scenario_cls', Scenario)
        self.scenario = scenario_cls(config, env_index, centerline, sideline)

        agents_master_cls = config.get('agents_master_cls', AgentsMaster)
        self.agents_master = agents_master_cls(config, self.scenario.topology_map)
        return



    def __len__(self):
        return self.config.num_steps






class DatasetReplay(object):
    def __init__(self, config: rldev.YamlConfig, env_index, dataset_dir=None):
        self.config = config
        self.env_index = env_index

        # t1 = time.time()
        # centerline = np.load(os.path.join(map_dir, config.map_name, 'topo.npy')).astype(np.float32)
        # sideline = np.concatenate([
        #     np.load(os.path.join(map_dir, config.map_name, 'solid.npy')),
        #     np.load(os.path.join(map_dir, config.map_name, 'ssolid.npy')),
        # ], axis=0).astype(np.float32)
        # t2 = time.time()
        # print(rldev.prefix(self) + 'load map time: ', t2-t1)

        t1 = time.time()
        self.record = Recorder.load(dataset_dir)
        t2 = time.time()
        print(rldev.prefix(self) + f'env {env_index} load data time: ', t2-t1)

        self.scenario = self.record['scenario']
        self.agents_master = self.record['agents_master']
        self.time_steps = [key for key in self.record.keys() if isinstance(key, int)]
        return

    def __len__(self):
        return len(self.time_steps) +1
    

    def __getitem__(self, time_step):
        episode_info = self.record[time_step].pop('episode_info')

        vehicles = self.record[time_step].values()
        vehicle_states = {vehicle.vi: vehicle.get_state() for vehicle in vehicles}

        current_vis = list(vehicle_states.keys())
        vehicle_future_states = {vi: [] for vi in current_vis}
        for i in range(time_step, np.clip(time_step+10, None, len(self))):
            future_record = self.record[i]
            for vi in current_vis:
                if vi in future_record.keys():
                    vehicle_future_states[vi].append(future_record[vi].get_state())

        return rldev.BaseData(episode_info=episode_info,
            vehicles=vehicles, vehicle_states=vehicle_states, vehicle_future_states=vehicle_future_states,
        )

