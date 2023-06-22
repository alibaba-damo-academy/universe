import rllib
import universe
import ray

import os
import copy
from typing import Tuple

import torch


from universe import EnvInteractiveMultiAgent as Env
from core.method_isac_v0 import IndependentSAC_v0 as Method



def init(config, mode, Env=Env, Method=Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/zdk/universe', '~/github/zdk/duplicity-rarl']
    config.set('github_repos', repos)

    from core.models import SAC as Method
    from core.method_isac_v0 import IndependentSAC_v0 as MethodAdv

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)

    config_method = env_master.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    config_method_adv = env_master.config_methods[1]
    MethodAdv = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(MethodAdv).remote
    method_adv = MethodAdv(config_method_adv, writer, tag_name='method_adv')
    method_adv.reset_writer.remote()

    return writer, env_master, method, method_adv







############################################################################
#### bottleneck ############################################################
############################################################################


def ray_sac__multi_scenario__explicit_adv(config, mode='train', scale=2):
    from core.env_ma_adv import EnvMultiAgentAdv as Env
    from core.env_ma_adv import MetricAdv

    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character_adv, bottleneck.config_env__with_character_share_adv,
        intersection.config_env__with_character_adv, intersection.config_env__with_character_share_adv,
        merge.config_env__with_character_adv, merge.config_env__with_character_share_adv,
        roundabout.config_env__with_character_adv, roundabout.config_env__with_character_share_adv,
    ] *scale)

    for cfg in config.envs:
        cfg.set('metric_cls', MetricAdv)
        # cfg.set('num_vehicles_range', rllib.basic.BaseData(min=1, max=1))

    # import pdb; pdb.set_trace()

    ### method param
    from config.method import config_sac as config_method
    from config.method import config_isac__adaptive_character as config_method_adv
    config.set('methods', [config_method, config_method_adv])

    return init(config, mode, Env, Method)





def ray_sac__multi_scenario__explicit_adv__small_scale(config, mode='train', scale=2):
    from core.env_ma_adv import EnvMultiAgentAdv as Env
    from core.env_ma_adv import MetricAdv

    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character_adv, bottleneck.config_env__with_character_share_adv,
        intersection.config_env__with_character_adv, intersection.config_env__with_character_share_adv,
        merge.config_env__with_character_adv, merge.config_env__with_character_share_adv,
        roundabout.config_env__with_character_adv, roundabout.config_env__with_character_share_adv,
    ] *scale)

    for cfg in config.envs:
        cfg.set('metric_cls', MetricAdv)
        cfg.set('num_vehicles_range', rllib.basic.BaseData(min=2, max=5))

    # import pdb; pdb.set_trace()

    ### method param
    from config.method import config_sac as config_method
    from config.method import config_isac__adaptive_character as config_method_adv
    config.set('methods', [config_method, config_method_adv])

    return init(config, mode, Env, Method)







def ray_sac__multi_scenario__target_adv(config, mode='train', scale=2):
    from core.env_ma_adv import EnvMultiAgentAdvTarget as Env
    from core.env_ma_adv import MetricAdv
    from utils.reward import RewardFunctionAdvTarget as reward_func

    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character_adv, bottleneck.config_env__with_character_share_adv,
        intersection.config_env__with_character_adv, intersection.config_env__with_character_share_adv,
        merge.config_env__with_character_adv, merge.config_env__with_character_share_adv,
        roundabout.config_env__with_character_adv, roundabout.config_env__with_character_share_adv,
    ] *scale)

    for cfg in config.envs:
        cfg.set('metric_cls', MetricAdv)
        cfg.set('reward_func', reward_func)
        # cfg.set('num_vehicles_range', rllib.basic.BaseData(min=1, max=1))

    # import pdb; pdb.set_trace()

    ### method param
    from config.method import config_sac as config_method
    from config.method import config_isac__adaptive_character as config_method_adv
    config.set('methods', [config_method, config_method_adv])

    return init(config, mode, Env, Method)



