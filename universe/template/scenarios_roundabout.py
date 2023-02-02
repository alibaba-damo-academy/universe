
import numpy as np
import os
from typing import Dict

from ..common import Vector
from ..common import GlobalPath, HashableTransform
from .. import Scenario



class ScenarioRoundabout(Scenario):
    def generate_global_paths(self):
        a = Vector(27.0, -35.3)
        b = Vector(29.2, -37.9)
        # c = Vector(15.7, 42.1)
        c = Vector(27.2, 35.3)
        # d = Vector(18.4, 44.6)
        d = Vector(29.0, 38.6)
        self.global_paths = [
            self.topology_map.route_planning(c, a),
            self.topology_map.route_planning(d, b),

            self.topology_map.route_planning(Vector(95, -1.7), a),
            self.topology_map.route_planning(Vector(95, -5.2), b),

            self.topology_map.route_planning(c, Vector(95, 1.7)),
            self.topology_map.route_planning(d, Vector(95, 5.2)),
        ]
        return


    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[0:0+20:4]
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

