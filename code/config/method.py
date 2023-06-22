import rllib


from core.models import ReplayBufferMultiAgentMultiWorker as ReplayBufferMultiAgent
from core.models import ReplayBufferSingleAgentMultiWorker as ReplayBufferSingleAgent
from core.models import PointNetWithAgentHistory  ### no_character
from core.models import PointNetWithCharacterAgentHistory  ### robust_character
from core.models import PointNetWithCharactersAgentHistory  ### adaptive_character


config_meta = rllib.basic.YamlConfig(
    device='cuda',
    num_cpus=1.0,
    num_gpus=0.2,
)


########################################################################
#### IndependentSAC ####################################################
########################################################################


config_isac__no_svo = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithAgentHistory,
    net_critic_fe=PointNetWithAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)


config_isac__self_svo = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharacterAgentHistory,
    net_critic_fe=PointNetWithCharacterAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)


config_isac__fully_svo = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistory,
    net_critic_fe=PointNetWithCharactersAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)




########################################################################
#### SAC ###############################################################
########################################################################


config_sac = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithAgentHistory,
    net_critic_fe=PointNetWithAgentHistory,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)


