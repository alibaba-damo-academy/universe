import rllib

import numpy as np
import os
import copy


import universe
from universe.template.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_merge
from utils import agents_master
from utils import perception
from utils import reward



from config.bottleneck import config_vehicle


config_env = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='merge',
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_merge.ScenarioMerge,
    boundary=rllib.basic.BaseData(x_min=120, y_min=-140, x_max=220, y_max=-20),
    center=rllib.basic.BaseData(x=170, y=-80),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),

    num_steps=100,
    decision_frequency=4, control_frequency=40,
    perception_range=50.0,

    bbx_x=config_vehicle.bbx_x, bbx_y=config_vehicle.bbx_y,
    config_vehicle=config_vehicle,

    agents_master_cls=agents_master.AgentListMaster,
    neural_vehicle_cls=agents_master.EndToEndVehicleWithCharacter,
    rule_vehicle_cls=agents_master.IdmVehicleWithCharacter,
    perception_cls=perception.PerceptionPointNet,
    reward_func=reward.RewardFunctionNoCharacter,

)



####################################################################################
### multi agent ####################################################################
####################################################################################



config_env__with_svo = copy.copy(config_env)
config_env__with_svo.reward_func = reward.RewardFunctionWithCharacter

config_env__with_svo_share = copy.copy(config_env__with_svo)
config_env__with_svo_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character





config_env__with_svo_adv = copy.copy(config_env__with_svo)
config_env__with_svo_adv.reward_func = reward.RewardFunctionAdv
config_env__with_svo_adv.scenario_randomization_cls = scenarios_template.ScenarioRandomizationNegtiveSVO

config_env__with_svo_share_adv = copy.copy(config_env__with_svo_adv)
config_env__with_svo_share_adv.scenario_randomization_cls = scenarios_template.ScenarioRandomizationNegtiveSVO_share_character





####################################################################################
### single agent ###################################################################
####################################################################################

config_env__neural_background = copy.copy(config_env)
config_env__neural_background.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch
config_env__neural_background.agents_master_cls = agents_master.AgentListMasterNeuralBackground
config_env__neural_background.rule_vehicle_cls = agents_master.EndToEndVehicleWithCharacterBackground



