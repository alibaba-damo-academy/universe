
import numpy as np
import os
from typing import Dict


from ..common import Vector
from ..common import GlobalPath, HashableTransform
from .. import Scenario



class ScenarioMerge(Scenario):
    def generate_global_paths(self):
        self.global_paths = [
            self.topology_map.route_planning(Vector(159.4, -118.4), Vector(191.2, -22)),
            self.topology_map.route_planning(Vector(160.5, -121.6), Vector(194.6, -22)),

            self.topology_map.route_planning(Vector(191.2, -138.0), Vector(191.2, -22)),
            self.topology_map.route_planning(Vector(194.6, -138.0), Vector(194.6, -22)),
            self.topology_map.route_planning(Vector(198.1, -138.0), Vector(198.1, -22)),
        ]
        return


    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[0:0+13:4]
            special_spawn_transforms[sts[0]] = global_path
            for st in sts:
                spawn_transforms[st] = global_path

        ### @todo: remove future
        def shuffle_dict(a: Dict):
            import random
            key = list(a.keys())
            random.shuffle(key)
            b = {}
            for key_i in key:
                b[key_i] = a[key_i]
            return b
        spawn_transforms = shuffle_dict(spawn_transforms)
        special_spawn_transforms = shuffle_dict(special_spawn_transforms)
        
        _spawn_transforms = [HashableTransform(sp) for sp in spawn_transforms.keys()]
        _spawn_transforms = [hsp.transform for hsp in list(set(_spawn_transforms))]
        self.spawn_transforms = {st: spawn_transforms[st] for st in _spawn_transforms}

        _special_spawn_transforms = [HashableTransform(sp) for sp in special_spawn_transforms.keys()]
        _special_spawn_transforms = [hsp.transform for hsp in list(set(_special_spawn_transforms))]
        self.special_spawn_transforms = {st: special_spawn_transforms[st] for st in _special_spawn_transforms}
        return

