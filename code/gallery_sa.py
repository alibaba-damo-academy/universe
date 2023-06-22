import rllib
import universe
import ray

import os
import copy
from typing import Tuple

import torch

import models_ma


def init(config, mode, Env, Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/zdk/universe', '~/github/zdk/duplicity-rarl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)

    config_method = env_master.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    return writer, env_master, method





################################################################################################
##### model adaptive ###########################################################################
################################################################################################




def get_sac__bottleneck__adaptive_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharactersAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        # model_dir='~/github/zdk/duplicity-rarl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        model_dir='~/github/zdk/duplicity-rarl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        model_num=865800,


        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy


def get_sac__intersection__adaptive_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharactersAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        # model_dir='~/github/zdk/duplicity-rarl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-14-10:25:41----ray_isac_adaptive_character__intersection/saved_models_method',
        model_dir='~/github/zdk/duplicity-rarl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-14-10:25:41----ray_isac_adaptive_character__intersection/saved_models_method',
        model_num=422600,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__merge__adaptive_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharactersAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        # model_dir='~/github/zdk/duplicity-rarl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        model_dir='~/github/zdk/duplicity-rarl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        model_num=866200,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__roundabout__adaptive_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharactersAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        # model_dir='~/github/zdk/duplicity-rarl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        model_dir='~/github/zdk/duplicity-rarl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        # model_num=869400,
        model_num=930000,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy







################################################################################################
##### model robust #############################################################################
################################################################################################






def get_sac__bottleneck__robust_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharacterAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_robust_character__bottleneck.model_dir,
        model_num=models_ma.isac_robust_character__bottleneck.model_num,


        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy


def get_sac__intersection__robust_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharacterAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_robust_character__intersection.model_dir,
        model_num=models_ma.isac_robust_character__intersection.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__merge__robust_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharacterAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_robust_character__merge.model_dir,
        model_num=models_ma.isac_robust_character__merge.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__roundabout__robust_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharacterAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_robust_character__roundabout.model_dir,
        model_num=models_ma.isac_robust_character__roundabout.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy







################################################################################################
##### model no character #######################################################################
################################################################################################





def get_sac__bottleneck__no_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_no_character__bottleneck.model_dir,
        model_num=models_ma.isac_no_character__bottleneck.model_num,


        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy


def get_sac__intersection__no_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_no_character__multi_scenario.model_dir,
        model_num=models_ma.isac_no_character__multi_scenario.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__merge__no_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_no_character__multi_scenario.model_dir,
        model_num=models_ma.isac_no_character__multi_scenario.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy



def get_sac__roundabout__no_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir=models_ma.isac_no_character__multi_scenario.model_dir,
        model_num=models_ma.isac_no_character__multi_scenario.model_num,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy











################################################################################################
##### model adv ################################################################################
################################################################################################







def get_sac__multi_scenario__robust_character_adv_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.models import PointNetWithCharacterAgentHistory as FeatureExtractor
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        model_dir='~/github/zdk/duplicity-rarl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-19-22:34:56----ray_isac_robust_character_explicit_adv__multi_scenario/saved_models_method',
        model_num=661400,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy
















################################################################################################
##### bottleneck ###############################################################################
################################################################################################




def ray_sac__bottleneck__idm_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env
    config.set('envs', [config_env] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)



def ray_sac__bottleneck__adaptive_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)






def ray_ppo__bottleneck__idm_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import PPO as Method

    ### env param
    from config.bottleneck import config_env
    config.set('envs', [config_env] *scale)
    
    ### method param
    from config.method import config_ppo as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)















def ray_sac__merge__adaptive_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.merge import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__merge__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)






def ray_sac__roundabout__adaptive_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.roundabout import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__roundabout__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)









################################################################################################
##### multi scenario ###########################################################################
################################################################################################





def ray_sac__idm_background__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env, bottleneck.config_env,
        intersection.config_env, intersection.config_env,
        merge.config_env, merge.config_env,
        roundabout.config_env, roundabout.config_env,
    ] *scale)

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)







def ray_sac__no_character__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__intersection__no_character_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('config_neural_policy', get_sac__merge__no_character_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('config_neural_policy', get_sac__roundabout__no_character_config(config))

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






def ray_sac__robust_character_copo__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls

    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('agents_master_cls', agents_master_cls)
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('agents_master_cls', agents_master_cls)
    config_intersection.set('config_neural_policy', get_sac__intersection__robust_character_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('agents_master_cls', agents_master_cls)
    config_merge.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('agents_master_cls', agents_master_cls)
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




def ray_sac__robust_character_copo_adv__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv as agents_master_cls

    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('agents_master_cls', agents_master_cls)
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('agents_master_cls', agents_master_cls)
    config_intersection.set('config_neural_policy', get_sac__intersection__robust_character_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('agents_master_cls', agents_master_cls)
    config_merge.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('agents_master_cls', agents_master_cls)
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








def ray_sac__robust_character_adv__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    config.set('envs', [
        config_bottleneck, config_bottleneck,
        config_intersection, config_intersection,
        config_merge, config_merge,
        config_roundabout, config_roundabout,
    ] *scale)

    from utils.scenarios_template import ScenarioRandomizationNegtiveSVO as scenario_randomization_cls
    for cfg in config.envs:
        cfg.set('scenario_randomization_cls', scenario_randomization_cls)

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)














def ray_sac__adaptive_character__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__intersection__adaptive_character_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

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








def ray_sac__adaptive_character__know_others_svo__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.models import SAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__intersection__adaptive_character_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

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
    from config.method import config_sac_know_svo as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)








def debug__ray_sac__robust_character_adv__multi_scenario(config, mode='train', scale=2):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_evaluate import EvaluateSAC as Method

    ### env param
    from config.bottleneck import config_env__neural_background as config_bottleneck
    config_bottleneck.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    from config.intersection import config_env__neural_background as config_intersection
    config_intersection.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))
    config_intersection.set('scenario_name', 'intersection_v2')

    from config.merge import config_env__neural_background as config_merge
    config_merge.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    from config.roundabout import config_env__neural_background as config_roundabout
    config_roundabout.set('config_neural_policy', get_sac__multi_scenario__robust_character_adv_config(config))

    config.set('envs', [
        config_bottleneck,
        # config_intersection,
        # config_merge,
        # config_roundabout,
    ])

    from utils.scenarios_template import ScenarioRandomizationAllNegtiveSVO as scenario_randomization_cls
    for cfg in config.envs:
        cfg.set('scenario_randomization_cls', scenario_randomization_cls)
        cfg.set('num_vehicles_range', rllib.basic.BaseData(min=20, max=20))

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)







