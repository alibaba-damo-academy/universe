import rldev
from .env_sa_v0 import EnvInteractiveSingleAgent as Env
import ray

import os
import copy
from typing import List




class EnvMaster(object):
    def __init__(self, config: rldev.YamlConfig, writer, env_cls: Env):
        self.config, self.path_pack = config, config.path_pack
        self.writer = writer
        self.env_cls = env_cls

        self.config_envs: List[rldev.YamlConfig] = config.envs
        self.num_envs = len(self.config_envs)
        self.env_indices = list(range(self.num_envs))

        self.cfgs = []
        for env_index, config_env in zip(self.env_indices, self.config_envs):
            cfg = copy.copy(config_env)
            cfg.set('path_pack', config.path_pack)
            cfg.set('dataset_name', config.dataset_name)
            cfg.set('env_cls', self.env_cls)
            cfg.set('env_index', env_index)
            cfg.set('seed', config.seed + env_index)
            cfg.set('render', config.render)
            cfg.set('invert', config.invert)
            cfg.set('render_save', config.render_save)
            cfg.set('dim_state', config_env.perception_cls.dim_state)
            cfg.set('dim_action', config_env.neural_vehicle_cls.dim_action)
            self.cfgs.append(cfg)
        
        config.set('dim_state', self.cfgs[0].perception_cls.dim_state)
        config.set('dim_action', self.cfgs[0].neural_vehicle_cls.dim_action)
        return


    def create_envs(self, func):
        worker_cls = ray.remote(num_cpus=0.1, num_gpus=0.1)(RemoteWorker).remote
        self.envs = [worker_cls(cfg, func) for cfg in self.cfgs]
        return





class RemoteWorker(object):
    def __init__(self, cfg, func):
        self.func = func

        suffix = 'env' + str(cfg.env_index)
        log_dir = os.path.join(cfg.path_pack.log_path, suffix)
        writer = rldev.Writer(log_dir=log_dir, comment=cfg.dataset_name, filename_suffix='--'+suffix)
        env_cls: Env = cfg.env_cls
        self.env = env_cls(cfg, writer, cfg.env_index)
    
    
    def run(self, n_iters=5):
        for i in range(n_iters):
            self.func(self.env)
        return




