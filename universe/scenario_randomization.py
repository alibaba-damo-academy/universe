import rldev

import numpy as np
import random
import pickle
import copy
from typing import List, Dict


from .common.geo import Vector, Transform
from .common.actor import ActorBoundingBox, check_collision
from .common.global_path import GlobalPath



class ScenarioRandomization(object):
    def __init__(self, spawn_transforms: Dict[Transform, GlobalPath], bbx_vectors: List[Vector], num_vehicles):
        self._spawn_transforms = spawn_transforms
        self.num_vehicles = num_vehicles

        if num_vehicles > len(spawn_transforms):
            msg = 'requested {} vehicles, but could only find {} spawn points'.format(num_vehicles, len(self._spawn_transforms))
            print(rldev.prefix(self) + 'warning: {}'.format(msg))
        
        self.spawn_transforms: List[Transform] = np.random.choice(list(self._spawn_transforms.keys()), size=num_vehicles, replace=False)
        self.bbx_vectors = np.random.choice(bbx_vectors, size=num_vehicles, replace=False)
        self.global_paths = np.array([copy.deepcopy(self._spawn_transforms[sp]) for sp in self.spawn_transforms])
        
        bbxs = np.array([ActorBoundingBox(t, bbx.x, bbx.y) for t, bbx in zip(self.spawn_transforms, self.bbx_vectors)])
        valid_flag = ~check_collision(bbxs)
        self.spawn_transforms = self.spawn_transforms[valid_flag]
        self.global_paths = self.global_paths[valid_flag]
        self.num_vehicles = len(self.global_paths)

        if self.num_vehicles == 0:
            num_vehicles = 1
            self.spawn_transforms = np.random.choice(list(self._spawn_transforms.keys()), size=num_vehicles, replace=False)
            self.bbx_vectors = np.random.choice(bbx_vectors, size=num_vehicles, replace=False)
            self.global_paths = np.array([copy.deepcopy(self._spawn_transforms[sp]) for sp in self.spawn_transforms])
            self.num_vehicles = len(self.global_paths)
        return


    def __getitem__(self, vi):
        return rldev.BaseData(vi=vi, global_path=self.global_paths[vi], transform=self.spawn_transforms[vi])




    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            scenario_randomization = pickle.load(f)
        return scenario_randomization


