import rllib
from universe import EnvInteractiveSingleAgent as Env
import ray

import os
import copy
import numpy as np
import psutil
import torch
from typing import List




class Runner_v0(object):
    def __init__(self, config: rllib.basic.YamlConfig, writer, env_cls: Env, func):
        self.config, self.path_pack = config, config.path_pack
        self.writer = writer
        self.env_cls = env_cls
        self.func = func

        ### resource
        self.num_cpus = ray.cluster_resources()['CPU']
        self.num_gpus = ray.cluster_resources()['GPU']

        self.used_cpus = 0.01
        self.used_gpus = 0.01
        for cfg_method in config.methods:
            self.used_cpus += cfg_method.num_cpus
            self.used_gpus += cfg_method.num_gpus

        ### env
        self.config_envs: List[rllib.basic.YamlConfig] = config.envs
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
            cfg.set('step_reset', config.step_reset)
            cfg.set('render', config.render)
            cfg.set('invert', config.invert)
            cfg.set('render_save', config.render_save)
            cfg.set('dim_state', config_env.perception_cls.dim_state)
            cfg.set('dim_action', config_env.neural_vehicle_cls.dim_action)
            self.cfgs.append(cfg)
        
        config.set('dim_state', self.cfgs[0].perception_cls.dim_state)
        config.set('dim_action', self.cfgs[0].neural_vehicle_cls.dim_action)

        ### method
        self.config_methods: List[rllib.basic.YamlConfig] = []
        for config_method in config.methods:
            cfg_method: rllib.basic.YamlConfig = copy.copy(config_method)
            cfg_method.set('num_workers', self.num_envs)
            cfg_method.set('path_pack', config.path_pack)
            cfg_method.set('dataset_name', config.dataset_name)
            cfg_method.set('dim_state', config.dim_state)
            cfg_method.set('dim_action', config.dim_action)
            cfg_method.set('evaluate', config.evaluate)
            cfg_method.set('model_dir', config.model_dir)
            cfg_method.set('model_num', config.model_num)
            cfg_method.set('method', config.method)
            self.config_methods.append(cfg_method)
        return


    def create_tasks(self, method):
        if self.num_gpus == 1:
            num_envs = self.num_envs
        elif self.num_gpus > 1:
            num_envs = self.num_envs + np.ceil(self.num_gpus) - (self.num_envs % np.ceil(self.num_gpus))
        else:
            raise NotImplementedError
        num_cpus = (self.num_cpus - self.used_cpus) /num_envs
        num_gpus = (self.num_gpus - self.used_gpus) /num_envs
        if num_cpus > 1:
            num_cpus = int(num_cpus)
        worker_cls = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(RemoteWorker).remote
        self.tasks = [worker_cls(cfg, method, self.func) for cfg in self.cfgs]
        return

    def create_envs(self):
        self.envs = [RemoteWorker(cfg, None, None).env for cfg in self.cfgs]
        return

    def execute(self, index, method):
        total_steps = ray.get([t.run.remote() for t in self.tasks])
        print('update index: ', index)
        ray.get(method.update_parameters_.remote(index, n_iters=sum(total_steps)))






class RemoteWorker(object):
    def __init__(self, cfg, method, func):
        self.method = method
        self.func = func

        suffix = 'env' + str(cfg.env_index)
        log_dir = os.path.join(cfg.path_pack.log_path, suffix)
        writer = rllib.basic.Writer(log_dir=log_dir, comment=cfg.dataset_name, filename_suffix='--'+suffix)
        env_cls: Env = cfg.env_cls
        self.env = env_cls(cfg, writer, cfg.env_index)
    
    
    def run(self, n_iters=5):
        total_steps = 0
        for i in range(n_iters):
            self.func(self.env, self.method)
            total_steps += self.env.time_step
        return total_steps









class Runner_v1(Runner_v0):
    def __init__(self, config: rllib.basic.YamlConfig, writer, env_cls: Env, method_cls: rllib.template.MethodSingleAgent):
        super().__init__(config, writer, env_cls)

        for config_method in self.config_methods:
            config_method.set('method_cls', method_cls)
        return

    def create_tasks(self, func):
        num_cpus = (self.num_cpus - self.used_cpus) /self.num_envs
        num_gpus = (self.num_gpus - self.used_gpus) /self.num_envs
        if num_cpus > 1:
            num_cpus = int(num_cpus)
        worker_cls = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(RemoteWorkerWithMethod).remote
        self.tasks = [worker_cls(cfg, self.config_methods[0], func) for cfg in self.cfgs]
        return



class RemoteWorkerWithMethod(RemoteWorker):
    def __init__(self, cfg, cfg_method, func):
        self.func = func

        suffix = 'env' + str(cfg.env_index)
        log_dir = os.path.join(cfg.path_pack.log_path, suffix)
        writer = rllib.basic.Writer(log_dir=log_dir, comment=cfg.dataset_name, filename_suffix='--'+suffix)
        env_cls: Env = cfg.env_cls
        self.env = env_cls(cfg, writer, cfg.env_index)
        self.method = cfg_method.method_cls(cfg_method, writer)




