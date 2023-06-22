import rllib
import universe
import ray

import os
import copy

import psutil
import torch






def init(config, mode, Env, Method):
    repos = ['~/github/zdk/rl-lib', '~/github/zdk/universe', '~/github/zdk/duplicity-rarl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)

    config_method = env_master.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).options(max_concurrency=2).remote
    # Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    return writer, env_master, method




from gallery_sa import get_sac__bottleneck__adaptive_character_config
from gallery_sa import get_sac__intersection__adaptive_character_config
from gallery_sa import get_sac__merge__adaptive_character_config
from gallery_sa import get_sac__roundabout__adaptive_character_config



from gallery_sa import get_sac__bottleneck__robust_character_config
from gallery_sa import get_sac__intersection__robust_character_config
from gallery_sa import get_sac__merge__robust_character_config
from gallery_sa import get_sac__roundabout__robust_character_config






def ray_sac__multi_scenario__adaptive_adv_background(config, mode='train', scale=2):
    from core.env_sa_adv import EnvSingleAgentAdv as Env
    from core.method_sac_adv import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__intersection__adaptive_character_config(config))

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('config_neural_policy', get_sac__merge__adaptive_character_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('config_neural_policy', get_sac__roundabout__adaptive_character_config(config))

    config.set('envs', [
        config_bottleneck, config_bottleneck,
        config_intersection, config_intersection,
        config_merge, config_merge,
        config_roundabout, config_roundabout,
    ] *scale)

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)






def ray_sac__multi_scenario__explicit_adv_background(config, mode='train', scale=2):
    from core.env_sa_adv import EnvSingleAgentAdv as Env
    from core.method_sac_adv import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__intersection__robust_character_config(config))

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))

    config.set('envs', [
        config_bottleneck, config_bottleneck,
        config_intersection, config_intersection,
        config_merge, config_merge,
        config_roundabout, config_roundabout,
    ] *scale)

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)


