import rllib
import universe
import ray

import os
import copy
from typing import Tuple

import torch


from universe import EnvInteractiveMultiAgent as Env
from core.method_isac_v0 import IndependentSAC_v0 as Method
from core.run_one_episode import MultiAgent
from runners.runner import Runner_v0


def init(config, mode, Env=Env, Method=Method, func=MultiAgent.v0) -> Tuple[Runner_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/zdk/universe', '~/github/zdk/duplicity-rarl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from runners.runner import Runner_v0 as Runner
    runner = Runner(config, writer, env_cls=Env, func=func)

    config_method = runner.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    return runner, method






def isac__social_comm(config, mode='train', scale=2):
    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_svo, bottleneck.config_env__with_svo_share,
        intersection.config_env__with_svo, intersection.config_env__with_svo_share,
        merge.config_env__with_svo, merge.config_env__with_svo_share,
        roundabout.config_env__with_svo, roundabout.config_env__with_svo_share,
    ] *scale)
    config.set('envs', [bottleneck.config_env__with_svo, intersection.config_env__with_svo, merge.config_env__with_svo, roundabout.config_env__with_svo])
    # config.set('envs', [bottleneck.config_env__with_svo])

    ### method param
    from config.method import config_isac__fully_svo as config_method
    config.set('methods', [config_method])

    return init(config, mode)









################################################################################################
##### multi scenario robust, cooperative #######################################################
################################################################################################





# from gallery_sa import get_sac__bottleneck__robust_character_config
# from gallery_sa import get_sac__intersection__robust_character_config
# from gallery_sa import get_sac__merge__robust_character_config
# from gallery_sa import get_sac__roundabout__robust_character_config




# def ray_isac_robust_character_copo__multi_scenario(config, mode='train', scale=2):
#     ### env param
#     from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
#     from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
#     from utils.reward import RewardFunctionGlobalCoordination as reward_func

#     from config.bottleneck import config_env as config_bottleneck
#     config_bottleneck.set('neural_vehicle_cls', neural_vehicle_cls)
#     config_bottleneck.set('agents_master_cls', agents_master_cls)
#     config_bottleneck.set('reward_func', reward_func)
#     config_bottleneck.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

#     from config.intersection import config_env as config_intersection
#     config_intersection.set('neural_vehicle_cls', neural_vehicle_cls)
#     config_intersection.set('agents_master_cls', agents_master_cls)
#     config_intersection.set('reward_func', reward_func)
#     config_intersection.set('config_neural_policy', get_sac__intersection__robust_character_config(config))

#     from config.merge import config_env as config_merge
#     config_merge.set('neural_vehicle_cls', neural_vehicle_cls)
#     config_merge.set('agents_master_cls', agents_master_cls)
#     config_merge.set('reward_func', reward_func)
#     config_merge.set('config_neural_policy', get_sac__merge__robust_character_config(config))

#     from config.roundabout import config_env as config_roundabout
#     config_roundabout.set('neural_vehicle_cls', neural_vehicle_cls)
#     config_roundabout.set('agents_master_cls', agents_master_cls)
#     config_roundabout.set('reward_func', reward_func)
#     config_roundabout.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))


#     config.set('envs', [
#         config_bottleneck, config_bottleneck,
#         config_intersection, config_intersection,
#         config_merge, config_merge,
#         config_roundabout, config_roundabout,
#     ] *scale)

#     ### method param
#     from config.method import config_isac__no_character as config_method
#     config.set('methods', [config_method])

#     return init(config, mode)







# ################################################################################################
# ##### multi scenario, explicit adv #############################################################
# ################################################################################################




# def ray_isac_robust_character_explicit_adv__multi_scenario__old(config, mode='train', scale=2):
#     ### env param
#     from config import bottleneck, intersection, merge, roundabout
#     config.set('envs', [
#         bottleneck.config_env__with_character_adv, bottleneck.config_env__with_character_share_adv,
#         intersection.config_env__with_character_adv, intersection.config_env__with_character_share_adv,
#         merge.config_env__with_character_adv, merge.config_env__with_character_share_adv,
#         roundabout.config_env__with_character_adv, roundabout.config_env__with_character_share_adv,
#     ] *scale)
    
#     ### method param
#     from config.method import config_isac__robust_character as config_method
#     config.set('methods', [config_method])

#     return init(config, mode)




# def ray_isac_robust_character_explicit_adv__multi_scenario__small_scale(config, mode='train', scale=2):
#     ### env param
#     from config import bottleneck, intersection, merge, roundabout
#     config.set('envs', [
#         bottleneck.config_env__with_character_adv, bottleneck.config_env__with_character_share_adv,
#         intersection.config_env__with_character_adv, intersection.config_env__with_character_share_adv,
#         merge.config_env__with_character_adv, merge.config_env__with_character_share_adv,
#         roundabout.config_env__with_character_adv, roundabout.config_env__with_character_share_adv,
#     ] *scale)

#     for cfg in config.envs:
#         cfg.set('num_vehicles_range', rllib.basic.BaseData(min=2, max=4))
    
#     ### method param
#     from config.method import config_isac__robust_character as config_method
#     config.set('methods', [config_method])

#     return init(config, mode)








