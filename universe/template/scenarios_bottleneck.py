
import numpy as np
import os
from typing import Dict
import random

from ..common import GlobalPath, HashableTransform
from .. import Scenario



class ScenarioBottleneck(Scenario):
    def generate_global_paths(self):
        path_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'maps/global_path_bottleneck')
        global_paths_numpy = np.concatenate([
            np.load(os.path.join(path_path, 'bottom_all.npy')),
            np.load(os.path.join(path_path, 'upper_all.npy')),
        ], axis=0)

        self.global_paths = []
        for global_path_numpy in global_paths_numpy:
            global_path = GlobalPath(x=global_path_numpy[:,0], y=global_path_numpy[:,1])
            self.global_paths.append(global_path)

        return


    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[10:10+13:4]
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



