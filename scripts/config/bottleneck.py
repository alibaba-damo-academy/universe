import rldev

import numpy as np
import os
import copy


import universe
from universe.template.dataset import DatasetInteractive as dataset_cls
from universe.template import scenarios_template
from universe.template import scenarios_bottleneck
from universe.template import agents_master
from universe.template import perception
from universe.template import reward


config_vehicle = rldev.YamlConfig(
    min_velocity=0.0,
    max_velocity=6.0,
    max_acceleration=5.0,
    min_acceleration=-5.0,
    max_steer=np.deg2rad(45),

    wheelbase=2.6,
    bbx_x=2.1, bbx_y=1.0,

    idm_scale_x=1.0,
    idm_scale_y=1.6,

)


config_env = rldev.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='bottleneck',
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_bottleneck.ScenarioBottleneck,
    boundary=rldev.BaseData(x_min=-150, y_min=180, x_max=150, y_max=220),
    center=rldev.BaseData(x=0, y=200),
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


