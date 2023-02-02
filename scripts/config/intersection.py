import rldev

import numpy as np
import os
import copy


import universe
from universe.template.dataset import DatasetInteractive as dataset_cls
from universe.template import scenarios_template
from universe.template import scenarios_intersection
from universe.template import agents_master
from universe.template import perception
from universe.template import reward



from config.bottleneck import config_vehicle



config_env = rldev.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='intersection',
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_intersection.ScenarioIntersection,
    boundary=rldev.BaseData(x_min=-74, y_min=-74, x_max=74, y_max=74),
    center=rldev.BaseData(x=0, y=0),
    num_vehicles_range=rldev.BaseData(min=8, max=20),

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




config_env__with_character = copy.copy(config_env)
config_env__with_character.reward_func = reward.RewardFunctionWithCharacter
